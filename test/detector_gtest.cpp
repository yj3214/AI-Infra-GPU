#include "detector.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path g_model_path{"yolov8s.onnx"};
std::filesystem::path g_image_path{"imgs/cat.jpg"};
std::filesystem::path g_config_path{"config/det_config.json"};

bool file_available(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) > 0;
}

bool runtime_assets_available() {
    return file_available(g_model_path) && file_available(g_image_path) && file_available(g_config_path);
}

void skip_if_runtime_assets_missing() {
    if (!runtime_assets_available()) {
        GTEST_SKIP() << "detector runtime tests require model, image and config files. model=" << g_model_path
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

myai_gpu::DetectorConfig default_test_config() {
    myai_gpu::DetectorConfig config;
    config.input_color_format = myai_gpu::InputColorFormat::Bgr;
    config.keep_ratio = true;
    config.batch_size = 1;
    config.height = 640;
    config.width = 640;
    config.use_fp16 = false;
    config.runtime.iou_threshold = 0.6F;
    config.runtime.confidence_threshold = 0.5F;
    return config;
}

void expect_consistent_result(const myai_gpu::DetectionResult& result, const cv::Size& image_size) {
    ASSERT_EQ(result.top_lefts.size(), result.size());
    ASSERT_EQ(result.bottom_rights.size(), result.size());
    ASSERT_EQ(result.confidence.size(), result.size());
    ASSERT_EQ(result.label_id.size(), result.size());

    for (std::size_t i = 0; i < result.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_TRUE(std::isfinite(result.confidence[i]));
        EXPECT_GE(result.confidence[i], 0.0F);
        EXPECT_LE(result.confidence[i], 1.0F);
        EXPECT_GE(result.label_id[i], 0);
        EXPECT_GE(result.top_lefts[i].x, 0);
        EXPECT_GE(result.top_lefts[i].y, 0);
        EXPECT_LE(result.bottom_rights[i].x, image_size.width);
        EXPECT_LE(result.bottom_rights[i].y, image_size.height);
        EXPECT_LE(result.top_lefts[i].x, result.bottom_rights[i].x);
        EXPECT_LE(result.top_lefts[i].y, result.bottom_rights[i].y);
        if (i > 0) {
            EXPECT_GE(result.confidence[i - 1], result.confidence[i]);
        }
    }
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

TEST(DetectionResultTest, EmptyResultHasZeroSize) {
    const myai_gpu::DetectionResult result;

    EXPECT_TRUE(result.empty());
    EXPECT_EQ(result.size(), 0U);
}

TEST(DetectionResultTest, SizeFollowsLabelCount) {
    const myai_gpu::DetectionResult result{{cv::Point{1, 2}}, {cv::Point{3, 4}}, {0.9F}, {7}};

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 1U);
}

TEST(DetectorConstructionTest, MissingConfigPathThrows) {
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), "missing_detector_config.json"}), std::runtime_error);
}

TEST(DetectorConstructionTest, InvalidConfigValuesThrow) {
    skip_if_runtime_assets_missing();

    auto invalid_batch = default_test_config();
    invalid_batch.batch_size = 0;
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), invalid_batch}), std::runtime_error);

    auto invalid_height = default_test_config();
    invalid_height.height = 0;
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), invalid_height}), std::runtime_error);

    auto invalid_width = default_test_config();
    invalid_width.width = 0;
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), invalid_width}), std::runtime_error);

    auto invalid_std = default_test_config();
    invalid_std.std[2] = 0.0F;
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), invalid_std}), std::runtime_error);

    auto invalid_iou = default_test_config();
    invalid_iou.runtime.iou_threshold = 1.2F;
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), invalid_iou}), std::runtime_error);

    auto invalid_confidence = default_test_config();
    invalid_confidence.runtime.confidence_threshold = -0.1F;
    EXPECT_THROW((myai_gpu::Detector{g_model_path.string(), invalid_confidence}), std::runtime_error);
}

TEST(DetectorConstructionTest, JsonConfigWithCommentsLoads) {
    skip_if_runtime_assets_missing();

    const auto config_path = write_temp_config(
        "myai_gpu_detector_gtest_config.json",
        R"json(
        {
          // Verifies JSON comment stripping.
          "input_color_format": "BGR",
          "keep_ratio": false,
          "iou_threshold": 0.4,
          "confidence_threshold": 0.25,
          "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
          "batch_size": 1,
                    "height": 640,
                    "width": 640,
          "use_fp16": false
        }
        )json");

    myai_gpu::Detector detector{g_model_path.string(), config_path.string()};
    EXPECT_TRUE(detector.is_ready());
    EXPECT_EQ(detector.get_batch_size(), 1);
    EXPECT_FLOAT_EQ(detector.get_runtime_config().iou_threshold, 0.4F);
    EXPECT_FLOAT_EQ(detector.get_runtime_config().confidence_threshold, 0.25F);

    std::filesystem::remove(config_path);
}

TEST(DetectorRuntimeTest, ConfigFileConstructorLoadsDetector) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), g_config_path.string()};

    EXPECT_TRUE(detector.is_ready());
    EXPECT_EQ(detector.get_batch_size(), 1);
}

TEST(DetectorRuntimeTest, ObjectConfigConstructorLoadsDetector) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};

    EXPECT_TRUE(detector.is_ready());
    EXPECT_EQ(detector.get_batch_size(), 1);
}

TEST(DetectorRuntimeTest, RuntimeConfigCanBeUpdated) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};
    detector.set_runtime_config(myai_gpu::DetRuntimeConfig{0.3F, 0.2F});

    const auto runtime = detector.get_runtime_config();
    EXPECT_FLOAT_EQ(runtime.iou_threshold, 0.3F);
    EXPECT_FLOAT_EQ(runtime.confidence_threshold, 0.2F);
    EXPECT_THROW(detector.set_runtime_config(myai_gpu::DetRuntimeConfig{-0.1F, 0.2F}), std::runtime_error);
    EXPECT_THROW(detector.set_runtime_config(myai_gpu::DetRuntimeConfig{0.3F, 1.2F}), std::runtime_error);
}

TEST(DetectorRuntimeTest, SingleImageInferenceProducesConsistentResult) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};
    const auto image = read_test_image();

    const auto result = detector.inference(image);

    expect_consistent_result(result, image.size());
    EXPECT_FALSE(result.empty());
}

TEST(DetectorRuntimeTest, VectorAndOutputParameterInferenceMatchSingleImage) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};
    const auto image = read_test_image();
    const auto single_result = detector.inference(image);

    const auto vector_results = detector.inference(std::vector<cv::Mat>{image});
    ASSERT_EQ(vector_results.size(), 1U);
    expect_consistent_result(vector_results.front(), image.size());
    EXPECT_EQ(vector_results.front().size(), single_result.size());

    std::vector<myai_gpu::DetectionResult> output_results;
    detector.inference(std::vector<cv::Mat>{image}, output_results);
    ASSERT_EQ(output_results.size(), 1U);
    expect_consistent_result(output_results.front(), image.size());
    EXPECT_EQ(output_results.front().size(), single_result.size());
}

TEST(DetectorRuntimeTest, ImageProxyInferenceMatchesCvMatInference) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};
    auto image = read_test_image();
    const auto single_result = detector.inference(image);

    std::vector<myai_gpu::ImageProxy> proxies;
    proxies.push_back(myai_gpu::ImageProxy{image.rows, image.cols, image.channels(), image.data});

    std::vector<myai_gpu::DetectionResult> proxy_results;
    detector.inference(proxies, proxy_results);

    ASSERT_EQ(proxy_results.size(), 1U);
    expect_consistent_result(proxy_results.front(), image.size());
    EXPECT_EQ(proxy_results.front().size(), single_result.size());
}

TEST(DetectorRuntimeTest, EmptyBatchReturnsEmptyResults) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};
    std::vector<myai_gpu::DetectionResult> results;

    detector.inference(std::vector<cv::Mat>{}, results);

    EXPECT_TRUE(results.empty());
}

TEST(DetectorRuntimeTest, InvalidInferenceInputsThrow) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector detector{g_model_path.string(), default_test_config()};
    const auto image = read_test_image();

    EXPECT_THROW(static_cast<void>(detector.inference(cv::Mat{})), std::runtime_error);
    EXPECT_THROW(static_cast<void>(detector.inference(std::vector<cv::Mat>{image, image})), std::runtime_error);

    std::vector<myai_gpu::DetectionResult> proxy_results;
    EXPECT_THROW(
        detector.inference(std::vector<myai_gpu::ImageProxy>{myai_gpu::ImageProxy{image.rows, image.cols, 3, nullptr}}, proxy_results),
        std::runtime_error);
    EXPECT_THROW(
        detector.inference(std::vector<myai_gpu::ImageProxy>{myai_gpu::ImageProxy{image.rows, image.cols, 2, image.data}}, proxy_results),
        std::runtime_error);
}

TEST(DetectorRuntimeTest, MoveConstructorTransfersReadyDetector) {
    skip_if_runtime_assets_missing();

    myai_gpu::Detector source{g_model_path.string(), default_test_config()};
    ASSERT_TRUE(source.is_ready());

    myai_gpu::Detector moved{std::move(source)};

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
