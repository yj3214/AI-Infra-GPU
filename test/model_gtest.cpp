#include "model.hpp"

#include <gtest/gtest.h>

#include <NvInferRuntime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::filesystem::path g_model_path{"model.plan"};
constexpr std::size_t kInvalidBufferSize = std::numeric_limits<std::size_t>::max();

bool model_file_available() {
    return std::filesystem::exists(g_model_path) && std::filesystem::is_regular_file(g_model_path) &&
           std::filesystem::file_size(g_model_path) > 0;
}

myai_gpu::ModelConfig default_test_config() {
    myai_gpu::ModelConfig config;
    config.batch_size = 1;
    config.input_height = 224;
    config.input_width = 224;
    config.use_fp16 = false;
    return config;
}

myai_gpu::Model make_test_model() {
    return myai_gpu::Model{g_model_path.string(), myai_gpu::TaskType::Classification, default_test_config()};
}

void skip_if_model_missing() {
    if (!model_file_available()) {
        GTEST_SKIP() << "model.plan is required for TensorRT runtime tests: " << g_model_path.string();
    }
}

const myai_gpu::TensorInfo* find_tensor(const std::vector<myai_gpu::TensorInfo>& tensors, const std::string& name) {
    const auto iter = std::ranges::find_if(tensors, [&name](const myai_gpu::TensorInfo& tensor) {
        return tensor.name == name;
    });
    return iter == tensors.end() ? nullptr : std::addressof(*iter);
}

}  // namespace

TEST(ModelMetadataTest, VersionAndHelloReturnExpectedText) {
    EXPECT_EQ(myai_gpu::version(), "myai-gpu 0.1.0");
    EXPECT_EQ(myai_gpu::hello(), "hello from myai_gpu SDK");
}

TEST(ModelMetadataTest, DefaultModelConfigIsValidForImageClassification) {
    const auto config = myai_gpu::ModelConfig{};

    EXPECT_EQ(config.batch_size, 1);
    EXPECT_EQ(config.input_height, 224);
    EXPECT_EQ(config.input_width, 224);
    EXPECT_FALSE(config.use_fp16);
    EXPECT_GT(config.workspace_size_bytes, 0U);
}

TEST(ModelConstructionTest, EmptyModelPathThrows) {
    EXPECT_THROW(
        (myai_gpu::Model{"", myai_gpu::TaskType::Classification, default_test_config()}), std::runtime_error);
}

TEST(ModelConstructionTest, UnsupportedModelFormatThrows) {
    EXPECT_THROW(
        (myai_gpu::Model{"unsupported_model.txt", myai_gpu::TaskType::Classification, default_test_config()}),
        std::runtime_error);
}

TEST(ModelConstructionTest, MissingPlanFileThrows) {
    EXPECT_THROW(
        (myai_gpu::Model{"missing_model.plan", myai_gpu::TaskType::Classification, default_test_config()}),
        std::runtime_error);
}

TEST(ModelRuntimeTest, PlanModelLoadsAndExposesRuntimeState) {
    skip_if_model_missing();

    auto model = make_test_model();

    EXPECT_TRUE(model.is_ready());
    EXPECT_EQ(model.task_type(), myai_gpu::TaskType::Classification);
    EXPECT_EQ(model.get_batch_size(), 1);
    EXPECT_NE(model.cuda_stream(), nullptr);
    EXPECT_FALSE(model.tensors().empty());
    EXPECT_FALSE(model.input_names().empty());
    EXPECT_FALSE(model.output_names().empty());
}

TEST(ModelRuntimeTest, PlanModelAllocatesHostAndDeviceBuffersForEveryTensor) {
    skip_if_model_missing();

    auto model = make_test_model();

    for (const auto& tensor : model.tensors()) {
        SCOPED_TRACE(tensor.name);

        EXPECT_GT(tensor.dims.nbDims, 0);
        EXPECT_NE(tensor.io_mode, nvinfer1::TensorIOMode::kNONE);
        EXPECT_NE(model.host_buffer(tensor.name), nullptr);
        EXPECT_NE(model.device_buffer(tensor.name), nullptr);
        EXPECT_NE(model.buffer_size(tensor.name), kInvalidBufferSize);
        EXPECT_GT(model.buffer_size(tensor.name), 0U);
    }
}

TEST(ModelRuntimeTest, InputAndOutputTensorNamesMapToTensorMetadata) {
    skip_if_model_missing();

    auto model = make_test_model();

    for (const auto& input_name : model.input_names()) {
        SCOPED_TRACE(input_name);
        const auto* tensor = find_tensor(model.tensors(), input_name);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->io_mode, nvinfer1::TensorIOMode::kINPUT);
    }

    for (const auto& output_name : model.output_names()) {
        SCOPED_TRACE(output_name);
        const auto* tensor = find_tensor(model.tensors(), output_name);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->io_mode, nvinfer1::TensorIOMode::kOUTPUT);
    }
}

TEST(ModelRuntimeTest, InferZeroInputRunsAndProducesFiniteFloatOutput) {
    skip_if_model_missing();

    auto model = make_test_model();
    ASSERT_FALSE(model.output_names().empty());

    ASSERT_NO_THROW(model.infer_zero_input());

    const auto& output_name = model.output_names().front();
    const auto* output_tensor = find_tensor(model.tensors(), output_name);
    ASSERT_NE(output_tensor, nullptr);
    ASSERT_EQ(output_tensor->dtype, nvinfer1::DataType::kFLOAT);

    const auto output_bytes = model.buffer_size(output_name);
    ASSERT_NE(output_bytes, kInvalidBufferSize);
    ASSERT_GE(output_bytes, sizeof(float));

    const auto* output = static_cast<const float*>(model.host_buffer(output_name));
    ASSERT_NE(output, nullptr);

    const auto output_count = output_bytes / sizeof(float);
    const auto values_to_check = std::min<std::size_t>(output_count, 32U);
    for (std::size_t i = 0; i < values_to_check; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST(ModelRuntimeTest, MoveConstructorTransfersOwnership) {
    skip_if_model_missing();

    auto source = make_test_model();
    ASSERT_TRUE(source.is_ready());

    auto moved = myai_gpu::Model{std::move(source)};

    EXPECT_TRUE(moved.is_ready());
    EXPECT_FALSE(source.is_ready());
    EXPECT_EQ(moved.task_type(), myai_gpu::TaskType::Classification);
    EXPECT_FALSE(moved.input_names().empty());
    EXPECT_FALSE(moved.output_names().empty());
}

int main(int argc, char** argv) {
    std::vector<char*> gtest_args;
    gtest_args.reserve(static_cast<std::size_t>(argc));
    gtest_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        constexpr std::string_view model_path_prefix{"--model_path="};
        if (arg.starts_with(model_path_prefix)) {
            g_model_path = arg.substr(model_path_prefix.size());
            continue;
        }
        gtest_args.push_back(argv[i]);
    }

    int gtest_argc = static_cast<int>(gtest_args.size());
    ::testing::InitGoogleTest(&gtest_argc, gtest_args.data());
    return RUN_ALL_TESTS();
}