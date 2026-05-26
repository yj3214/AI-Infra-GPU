#include "segmentor.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path g_model_path{"unet.onnx"};
std::filesystem::path g_image_path{"imgs/car.jpg"};
std::filesystem::path g_config_path{"config/seg_config.json"};

bool file_available(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) > 0;
}

bool runtime_assets_available() {
    return file_available(g_model_path) && file_available(g_image_path) && file_available(g_config_path);
}

void skip_if_runtime_assets_missing() {
    if (!runtime_assets_available()) {
        GTEST_SKIP() << "segmentor runtime tests require model, image and config files. model=" << g_model_path
                     << ", image=" << g_image_path << ", config=" << g_config_path;
    }
}

cv::Mat read_test_image() {
    cv::Mat image = cv::imread(g_image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("failed to read test image: " + g_image_path.string());
    }
    return image;
}

myai_gpu::SegmentorConfig default_test_config() {
    myai_gpu::SegmentorConfig config;
    config.input_color_format = myai_gpu::InputColorFormat::Bgr;
    config.keep_ratio = false;
    config.batch_size = 1;
    config.height = 512;
    config.width = 512;
    config.use_fp16 = false;
    config.mask_threshold = 0.5F;
    return config;
}

void expect_valid_mask(const myai_gpu::SegmentationResult& result, const cv::Size& image_size) {
    ASSERT_FALSE(result.empty());
    ASSERT_NE(result.mask, nullptr);
    EXPECT_EQ(result.width, image_size.width);
    EXPECT_EQ(result.height, image_size.height);
    EXPECT_EQ(result.size(), static_cast<std::size_t>(image_size.width) * static_cast<std::size_t>(image_size.height));

    const auto* begin = result.mask.get();
    const auto* end = begin + result.size();
    const auto [min_iter, max_iter] = std::minmax_element(begin, end);
    EXPECT_GE(static_cast<int>(*min_iter), 0);
    EXPECT_LE(static_cast<int>(*max_iter), 255);
}

std::filesystem::path write_temp_config(std::string_view name, std::string_view content) {
    const auto path = std::filesystem::temp_directory_path() / std::filesystem::path{name};
    std::ofstream output{path};
    if (!output) {
        throw std::runtime_error("failed to create temp config: " + path.string());
    }
    output << content;
    return path;
}

}  // namespace

TEST(SegmentationResultTest, EmptyResultHasZeroSize) {
    const myai_gpu::SegmentationResult result;

    EXPECT_TRUE(result.empty());
    EXPECT_EQ(result.size(), 0U);
}

TEST(SegmentationResultTest, CopyPerformsDeepCopy) {
    auto mask = std::make_unique<unsigned char[]>(4U);
    mask[0] = 1;
    mask[1] = 2;
    mask[2] = 3;
    mask[3] = 4;
    const myai_gpu::SegmentationResult source{2, 2, std::move(mask)};

    myai_gpu::SegmentationResult copied = source;

    ASSERT_FALSE(copied.empty());
    ASSERT_NE(copied.mask, source.mask);
    EXPECT_EQ(copied.size(), source.size());
    EXPECT_EQ(copied.mask[0], 1);
    copied.mask[0] = 42;
    EXPECT_EQ(source.mask[0], 1);
}

TEST(SegmentorConstructionTest, MissingConfigPathThrows) {
    EXPECT_THROW((myai_gpu::Segmentor{g_model_path.string(), "missing_segmentor_config.json"}), std::runtime_error);
}

TEST(SegmentorConstructionTest, InvalidConfigValuesThrow) {
    skip_if_runtime_assets_missing();

    auto invalid_batch = default_test_config();
    invalid_batch.batch_size = 0;
    EXPECT_THROW((myai_gpu::Segmentor{g_model_path.string(), invalid_batch}), std::runtime_error);

    auto invalid_height = default_test_config();
    invalid_height.height = 0;
    EXPECT_THROW((myai_gpu::Segmentor{g_model_path.string(), invalid_height}), std::runtime_error);

    auto invalid_width = default_test_config();
    invalid_width.width = 0;
    EXPECT_THROW((myai_gpu::Segmentor{g_model_path.string(), invalid_width}), std::runtime_error);

    auto invalid_std = default_test_config();
    invalid_std.std[0] = 0.0F;
    EXPECT_THROW((myai_gpu::Segmentor{g_model_path.string(), invalid_std}), std::runtime_error);

    auto invalid_threshold = default_test_config();
    invalid_threshold.mask_threshold = 1.5F;
    EXPECT_THROW((myai_gpu::Segmentor{g_model_path.string(), invalid_threshold}), std::runtime_error);
}

TEST(SegmentorConstructionTest, JsonConfigWithCommentsLoads) {
    skip_if_runtime_assets_missing();

    const auto config_path = write_temp_config(
        "myai_gpu_segmentor_gtest_config.json",
        R"json(
        {
          // Verifies JSON comment stripping.
          "input_color_format": "BGR",
          "keep_ratio": false,
          "mean": [0.0, 0.0, 0.0],
          "std": [1.0, 1.0, 1.0],
          "batch_size": 1,
          "height": 512,
          "width": 512,
          "use_fp16": false,
          "mask_threshold": 0.5
        }
        )json");

    myai_gpu::Segmentor segmentor{g_model_path.string(), config_path.string()};
    EXPECT_TRUE(segmentor.is_ready());
    EXPECT_EQ(segmentor.get_batch_size(), 1);

    std::filesystem::remove(config_path);
}

TEST(SegmentorRuntimeTest, ConfigFileConstructorLoadsSegmentor) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), g_config_path.string()};

    EXPECT_TRUE(segmentor.is_ready());
    EXPECT_EQ(segmentor.get_batch_size(), 1);
}

TEST(SegmentorRuntimeTest, ObjectConfigConstructorLoadsSegmentor) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), default_test_config()};

    EXPECT_TRUE(segmentor.is_ready());
    EXPECT_EQ(segmentor.get_batch_size(), 1);
}

TEST(SegmentorRuntimeTest, SingleImageInferenceProducesMask) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), default_test_config()};
    const auto image = read_test_image();

    const auto result = segmentor.inference(image);

    expect_valid_mask(result, image.size());
}

TEST(SegmentorRuntimeTest, VectorAndOutputParameterInferenceMatchSingleImage) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), default_test_config()};
    const auto image = read_test_image();
    const auto single_result = segmentor.inference(image);

    const auto vector_results = segmentor.inference(std::vector<cv::Mat>{image});
    ASSERT_EQ(vector_results.size(), 1U);
    expect_valid_mask(vector_results.front(), image.size());
    EXPECT_EQ(vector_results.front().size(), single_result.size());

    std::vector<myai_gpu::SegmentationResult> output_results;
    segmentor.inference(std::vector<cv::Mat>{image}, output_results);
    ASSERT_EQ(output_results.size(), 1U);
    expect_valid_mask(output_results.front(), image.size());
    EXPECT_EQ(output_results.front().size(), single_result.size());
}

TEST(SegmentorRuntimeTest, ImageProxyInferenceMatchesCvMatInference) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), default_test_config()};
    auto image = read_test_image();
    const auto single_result = segmentor.inference(image);

    std::vector<myai_gpu::ImageProxy> proxies;
    proxies.push_back(myai_gpu::ImageProxy{image.rows, image.cols, image.channels(), image.data});

    std::vector<myai_gpu::SegmentationResult> proxy_results;
    segmentor.inference(proxies, proxy_results);

    ASSERT_EQ(proxy_results.size(), 1U);
    expect_valid_mask(proxy_results.front(), image.size());
    EXPECT_EQ(proxy_results.front().size(), single_result.size());
}

TEST(SegmentorRuntimeTest, EmptyBatchReturnsEmptyResults) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), default_test_config()};
    std::vector<myai_gpu::SegmentationResult> results;

    segmentor.inference(std::vector<cv::Mat>{}, results);

    EXPECT_TRUE(results.empty());
}

TEST(SegmentorRuntimeTest, InvalidInferenceInputsThrow) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor segmentor{g_model_path.string(), default_test_config()};
    const auto image = read_test_image();

    EXPECT_THROW(static_cast<void>(segmentor.inference(cv::Mat{})), std::runtime_error);
    EXPECT_THROW(static_cast<void>(segmentor.inference(std::vector<cv::Mat>{image, image})), std::runtime_error);

    std::vector<myai_gpu::SegmentationResult> proxy_results;
    EXPECT_THROW(
        segmentor.inference(std::vector<myai_gpu::ImageProxy>{myai_gpu::ImageProxy{image.rows, image.cols, 3, nullptr}}, proxy_results),
        std::runtime_error);
    EXPECT_THROW(
        segmentor.inference(std::vector<myai_gpu::ImageProxy>{myai_gpu::ImageProxy{image.rows, image.cols, 2, image.data}}, proxy_results),
        std::runtime_error);
}

TEST(SegmentorRuntimeTest, MoveConstructorTransfersReadySegmentor) {
    skip_if_runtime_assets_missing();

    myai_gpu::Segmentor source{g_model_path.string(), default_test_config()};
    ASSERT_TRUE(source.is_ready());

    myai_gpu::Segmentor moved{std::move(source)};

    EXPECT_TRUE(moved.is_ready());
    EXPECT_FALSE(source.is_ready());
    EXPECT_EQ(moved.get_batch_size(), 1);
}

int main(int argc, char** argv) {
    std::vector<char*> gtest_args;
    gtest_args.reserve(static_cast<std::size_t>(argc));
    gtest_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        constexpr std::string_view model_path_prefix{"--model_path="};
        constexpr std::string_view image_path_prefix{"--image_path="};
        constexpr std::string_view config_path_prefix{"--config_path="};
        if (arg.starts_with(model_path_prefix)) {
            g_model_path = arg.substr(model_path_prefix.size());
            continue;
        }
        if (arg.starts_with(image_path_prefix)) {
            g_image_path = arg.substr(image_path_prefix.size());
            continue;
        }
        if (arg.starts_with(config_path_prefix)) {
            g_config_path = arg.substr(config_path_prefix.size());
            continue;
        }
        gtest_args.push_back(argv[i]);
    }

    int gtest_argc = static_cast<int>(gtest_args.size());
    ::testing::InitGoogleTest(&gtest_argc, gtest_args.data());
    return RUN_ALL_TESTS();
}
