#include "classifier.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::filesystem::path g_model_path{"model.plan"};
std::filesystem::path g_image_path{"imgs/cat.jpg"};
std::filesystem::path g_config_path{"config/cls_config.json"};

bool file_available(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) > 0;
}

bool runtime_assets_available() {
    return file_available(g_model_path) && file_available(g_image_path) && file_available(g_config_path);
}

void skip_if_runtime_assets_missing() {
    if (!runtime_assets_available()) {
        GTEST_SKIP() << "classifier runtime tests require model, image and config files. model=" << g_model_path
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

myai_gpu::ClassifierConfig default_test_config(int topk = 5) {
    myai_gpu::ClassifierConfig config;
    config.input_color_format = myai_gpu::InputColorFormat::Bgr;
    config.batch_size = 1;
    config.height = 224;
    config.width = 224;
    config.topk = topk;
    config.use_fp16 = false;
    return config;
}

void expect_valid_result(const myai_gpu::ClassificationResult& result, std::size_t expected_topk) {
    ASSERT_EQ(result.confidence.size(), expected_topk);
    ASSERT_EQ(result.label_id.size(), expected_topk);
    for (std::size_t i = 0; i < result.confidence.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_TRUE(std::isfinite(result.confidence[i]));
        EXPECT_GE(result.confidence[i], 0.0F);
        EXPECT_LE(result.confidence[i], 1.0F);
        EXPECT_GE(result.label_id[i], 0);
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

TEST(ClassificationResultTest, EmptyResultHasZeroClassIdAndScore) {
    const myai_gpu::ClassificationResult result;

    EXPECT_EQ(result.class_id(), 0U);
    EXPECT_FLOAT_EQ(result.score(), 0.0F);
}

TEST(ClassificationResultTest, ResultAccessorsReturnTop1) {
    const myai_gpu::ClassificationResult result{{0.2F, 0.1F}, {42, 7}};

    EXPECT_EQ(result.class_id(), 42U);
    EXPECT_FLOAT_EQ(result.score(), 0.2F);
}

TEST(ClassifierConstructionTest, MissingConfigPathThrows) {
    EXPECT_THROW(
        (myai_gpu::Classifier{g_model_path.string(), "missing_classifier_config.json"}), std::runtime_error);
}

TEST(ClassifierConstructionTest, InvalidConfigValuesThrow) {
    skip_if_runtime_assets_missing();

    auto invalid_batch = default_test_config();
    invalid_batch.batch_size = 0;
    EXPECT_THROW((myai_gpu::Classifier{g_model_path.string(), invalid_batch}), std::runtime_error);

    auto invalid_height = default_test_config();
    invalid_height.height = 0;
    EXPECT_THROW((myai_gpu::Classifier{g_model_path.string(), invalid_height}), std::runtime_error);

    auto invalid_width = default_test_config();
    invalid_width.width = 0;
    EXPECT_THROW((myai_gpu::Classifier{g_model_path.string(), invalid_width}), std::runtime_error);

    auto invalid_topk = default_test_config();
    invalid_topk.topk = 0;
    EXPECT_THROW((myai_gpu::Classifier{g_model_path.string(), invalid_topk}), std::runtime_error);

    auto invalid_std = default_test_config();
    invalid_std.std[1] = 0.0F;
    EXPECT_THROW((myai_gpu::Classifier{g_model_path.string(), invalid_std}), std::runtime_error);
}

TEST(ClassifierConstructionTest, JsonConfigWithCommentsAndRgbFormatLoads) {
    skip_if_runtime_assets_missing();

    const auto config_path = write_temp_config(
        "myai_gpu_classifier_gtest_config.json",
        R"json(
        {
          // This comment verifies that the lightweight parser strips JSON comments.
          "input_color_format": "RGB",
          "mean": [0.485, 0.456, 0.406],
          "std": [0.229, 0.224, 0.225],
          "batch_size": 1,
          "height": 224,
          "width": 224,
          "topk": 3,
          "use_fp16": false
        }
        )json");

    myai_gpu::Classifier classifier{g_model_path.string(), config_path.string()};
    EXPECT_TRUE(classifier.is_ready());
    EXPECT_EQ(classifier.get_batch_size(), 1);

    std::filesystem::remove(config_path);
}

TEST(ClassifierRuntimeTest, ConfigFileConstructorLoadsClassifier) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};

    EXPECT_TRUE(classifier.is_ready());
    EXPECT_EQ(classifier.get_batch_size(), 1);
}

TEST(ClassifierRuntimeTest, ObjectConfigConstructorLoadsClassifier) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), default_test_config(5)};

    EXPECT_TRUE(classifier.is_ready());
    EXPECT_EQ(classifier.get_batch_size(), 1);
}

TEST(ClassifierRuntimeTest, SingleImageInferenceProducesExpectedTopKCatResult) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    const auto image = read_test_image();

    const auto result = classifier.inference(image);

    expect_valid_result(result, 5);
    EXPECT_EQ(result.class_id(), 281U);
    EXPECT_GT(result.score(), 0.6F);
}

TEST(ClassifierRuntimeTest, VectorAndOutputParameterInferenceMatchSingleImage) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    const auto image = read_test_image();
    const auto single_result = classifier.inference(image);

    const auto vector_results = classifier.inference(std::vector<cv::Mat>{image});
    ASSERT_EQ(vector_results.size(), 1U);
    expect_valid_result(vector_results.front(), 5);
    EXPECT_EQ(vector_results.front().class_id(), single_result.class_id());
    EXPECT_NEAR(vector_results.front().score(), single_result.score(), 1.0e-4F);

    std::vector<myai_gpu::ClassificationResult> output_results;
    classifier.inference(std::vector<cv::Mat>{image}, output_results);
    ASSERT_EQ(output_results.size(), 1U);
    expect_valid_result(output_results.front(), 5);
    EXPECT_EQ(output_results.front().class_id(), single_result.class_id());
    EXPECT_NEAR(output_results.front().score(), single_result.score(), 1.0e-4F);
}

TEST(ClassifierRuntimeTest, ImageProxyInferenceMatchesCvMatInference) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    auto image = read_test_image();
    const auto single_result = classifier.inference(image);

    std::vector<myai_gpu::ImageProxy> proxies;
    proxies.push_back(myai_gpu::ImageProxy{image.rows, image.cols, image.channels(), image.data});

    std::vector<myai_gpu::ClassificationResult> proxy_results;
    classifier.inference(proxies, proxy_results);

    ASSERT_EQ(proxy_results.size(), 1U);
    expect_valid_result(proxy_results.front(), 5);
    EXPECT_EQ(proxy_results.front().class_id(), single_result.class_id());
    EXPECT_NEAR(proxy_results.front().score(), single_result.score(), 1.0e-4F);
}

TEST(ClassifierRuntimeTest, GrayAndRgbaImagesAreSupported) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    const auto image = read_test_image();

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const auto gray_result = classifier.inference(gray);
    expect_valid_result(gray_result, 5);

    cv::Mat bgra;
    cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
    const auto bgra_result = classifier.inference(bgra);
    expect_valid_result(bgra_result, 5);
    EXPECT_EQ(bgra_result.class_id(), 281U);
}

TEST(ClassifierRuntimeTest, EmptyBatchReturnsEmptyResults) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    std::vector<myai_gpu::ClassificationResult> results;

    classifier.inference(std::vector<cv::Mat>{}, results);

    EXPECT_TRUE(results.empty());
}

TEST(ClassifierRuntimeTest, InvalidInferenceInputsThrow) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    const auto image = read_test_image();

    EXPECT_THROW(static_cast<void>(classifier.inference(cv::Mat{})), std::runtime_error);
    EXPECT_THROW(static_cast<void>(classifier.inference(std::vector<cv::Mat>{image, image})), std::runtime_error);

    std::vector<myai_gpu::ClassificationResult> proxy_results;
    EXPECT_THROW(
        classifier.inference(std::vector<myai_gpu::ImageProxy>{myai_gpu::ImageProxy{image.rows, image.cols, 3, nullptr}}, proxy_results),
        std::runtime_error);
    EXPECT_THROW(
        classifier.inference(std::vector<myai_gpu::ImageProxy>{myai_gpu::ImageProxy{image.rows, image.cols, 2, image.data}}, proxy_results),
        std::runtime_error);
}

TEST(ClassifierRuntimeTest, BenchmarkReturnsPositiveLatencyAndFps) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    const auto image = read_test_image();

    const auto perf = classifier.benchmark(image, 2, 5);

    EXPECT_EQ(perf.warmup_runs, 2);
    EXPECT_EQ(perf.benchmark_runs, 5);
    EXPECT_GT(perf.total_ms, 0.0);
    EXPECT_GT(perf.average_latency_ms, 0.0);
    EXPECT_GT(perf.fps, 0.0);
}

TEST(ClassifierRuntimeTest, BenchmarkRejectsInvalidRunCounts) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier classifier{g_model_path.string(), g_config_path.string()};
    const auto image = read_test_image();

    EXPECT_THROW(static_cast<void>(classifier.benchmark(image, -1, 5)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(classifier.benchmark(image, 1, 0)), std::runtime_error);
}

TEST(ClassifierRuntimeTest, MoveConstructorTransfersReadyClassifier) {
    skip_if_runtime_assets_missing();

    myai_gpu::Classifier source{g_model_path.string(), g_config_path.string()};
    ASSERT_TRUE(source.is_ready());

    myai_gpu::Classifier moved{std::move(source)};

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