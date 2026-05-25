#include "model.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ClassificationResult {
    std::size_t class_id{};
    float score{};
};

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

void check_cuda(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + " failed: " + cudaGetErrorString(status));
    }
}

cv::Mat make_gradient_image(int width, int height) {
    cv::Mat image(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b{
                static_cast<std::uint8_t>(128),
                static_cast<std::uint8_t>((y * 255) / std::max(height - 1, 1)),
                static_cast<std::uint8_t>((x * 255) / std::max(width - 1, 1)),
            };
        }
    }

    return image;
}

cv::Mat read_image_bgr(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("failed to read image with OpenCV: " + path);
    }
    return image;
}

void preprocess_bgr_to_rgb_chw_device(const cv::Mat& image_bgr, myai_gpu::Model& model, int dst_width, int dst_height) {
    if (image_bgr.empty()) {
        throw std::runtime_error("input image is empty");
    }
    if (image_bgr.channels() != 3) {
        throw std::runtime_error("classification demo expects a 3-channel image");
    }
    if (model.input_names().empty()) {
        throw std::runtime_error("model has no input tensor");
    }

    const std::string& input_name = model.input_names().front();
    void* input_device = model.device_buffer(input_name);
    const std::size_t input_bytes = model.buffer_size(input_name);
    const std::size_t plane_elements = static_cast<std::size_t>(dst_width) * static_cast<std::size_t>(dst_height);
    const std::size_t expected_bytes = 3U * plane_elements * sizeof(float);
    if (input_device == nullptr) {
        throw std::runtime_error("failed to get device input buffer: " + input_name);
    }
    if (input_bytes < expected_bytes) {
        throw std::runtime_error("device input buffer is smaller than preprocessed image");
    }

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(model.cuda_stream());

    cv::cuda::GpuMat gpu_bgr;
    gpu_bgr.upload(image_bgr, cv_stream);

    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::cvtColor(gpu_bgr, gpu_rgb, cv::COLOR_BGR2RGB, 0, cv_stream);

    cv::cuda::GpuMat resized_rgb;
    cv::cuda::resize(gpu_rgb, resized_rgb, cv::Size{dst_width, dst_height}, 0.0, 0.0, cv::INTER_LINEAR, cv_stream);

    cv::cuda::GpuMat resized_rgb_fp32;
    resized_rgb.convertTo(resized_rgb_fp32, CV_32FC3, 1.0 / 255.0, 0.0, cv_stream);

    cv::cuda::GpuMat normalized_rgb;
    cv::cuda::subtract(resized_rgb_fp32, cv::Scalar{0.485, 0.456, 0.406}, normalized_rgb, cv::noArray(), -1, cv_stream);
    cv::cuda::divide(normalized_rgb, cv::Scalar{0.229, 0.224, 0.225}, normalized_rgb, 1.0, -1, cv_stream);

    std::vector<cv::cuda::GpuMat> chw_planes;
    chw_planes.reserve(3);
    auto* input_float = static_cast<float*>(input_device);
    for (int c = 0; c < 3; ++c) {
        chw_planes.emplace_back(dst_height, dst_width, CV_32FC1, input_float + static_cast<std::size_t>(c) * plane_elements);
    }
    cv::cuda::split(normalized_rgb, chw_planes, cv_stream);
    check_cuda(cudaStreamSynchronize(model.cuda_stream()), "cudaStreamSynchronize after OpenCV CUDA preprocess");
}

std::vector<float> softmax(const float* logits, std::size_t count) {
    if (logits == nullptr || count == 0) {
        throw std::runtime_error("empty classification output");
    }

    const float max_logit = *std::max_element(logits, logits + count);
    std::vector<float> probabilities(count);
    float sum = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    for (auto& probability : probabilities) {
        probability /= sum;
    }
    return probabilities;
}

ClassificationResult classify_output(const myai_gpu::Model& model) {
    if (model.output_names().empty()) {
        throw std::runtime_error("model has no output tensor");
    }

    const std::string& output_name = model.output_names().front();
    const auto* logits = static_cast<const float*>(model.host_buffer(output_name));
    const std::size_t count = model.buffer_size(output_name) / sizeof(float);
    auto probabilities = softmax(logits, count);

    const auto best = std::max_element(probabilities.begin(), probabilities.end());
    return ClassificationResult{
        static_cast<std::size_t>(std::distance(probabilities.begin(), best)),
        *best,
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "model.plan";
        const std::string image_path = argc > 2 ? argv[2] : "";

        constexpr int input_width = 224;
        constexpr int input_height = 224;

        myai_gpu::ModelConfig config;
        config.batch_size = 1;
        config.use_fp16 = false;

        const auto load_begin = Clock::now();
        myai_gpu::Model model{model_path, myai_gpu::TaskType::Classification, config};
        const auto load_end = Clock::now();

        std::cout << myai_gpu::version() << '\n';
        std::cout << "model: " << model_path << '\n';
        std::cout << "model ready: " << std::boolalpha << model.is_ready() << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "model load/init time: " << elapsed_ms(load_begin, load_end) << " ms\n";

        const auto preprocess_begin = Clock::now();
        const cv::Mat image_bgr = image_path.empty() ? make_gradient_image(input_width, input_height) : read_image_bgr(image_path);
        preprocess_bgr_to_rgb_chw_device(image_bgr, model, input_width, input_height);
        const auto preprocess_end = Clock::now();

        constexpr int warmup_runs = 10;
        constexpr int benchmark_runs = 100;

        const auto cold_inference_begin = Clock::now();
        model.inference_device_input();
        const auto cold_inference_end = Clock::now();

        for (int i = 0; i < warmup_runs; ++i) {
            model.inference_device_input();
        }

        const auto benchmark_begin = Clock::now();
        for (int i = 0; i < benchmark_runs; ++i) {
            model.inference_device_input();
        }
        const auto benchmark_end = Clock::now();

        const double cold_inference_ms = elapsed_ms(cold_inference_begin, cold_inference_end);
        const double benchmark_total_ms = elapsed_ms(benchmark_begin, benchmark_end);
        const double average_latency_ms = benchmark_total_ms / static_cast<double>(benchmark_runs);
        const double fps = average_latency_ms > 0.0 ? 1000.0 * static_cast<double>(model.get_batch_size()) / average_latency_ms : 0.0;

        const auto postprocess_begin = Clock::now();
        const auto result = classify_output(model);
        const auto postprocess_end = Clock::now();

        std::cout << "image: " << (image_path.empty() ? "generated-gradient" : image_path) << '\n';
        std::cout << "original image size: " << image_bgr.cols << "x" << image_bgr.rows << '\n';
        std::cout << "input size: " << input_width << "x" << input_height << " NCHW FP32 RGB ImageNet normalized\n";
        std::cout << "classification result\n";
        std::cout << "  class_id: " << result.class_id << '\n';
        std::cout << "  confidence: " << result.score << '\n';
        std::cout << "timing\n";
        std::cout << "  preprocess: " << elapsed_ms(preprocess_begin, preprocess_end) << " ms\n";
        std::cout << "  cold inference: " << cold_inference_ms << " ms\n";
        std::cout << "  warmup runs: " << warmup_runs << '\n';
        std::cout << "  benchmark runs: " << benchmark_runs << '\n';
        std::cout << "  benchmark total: " << benchmark_total_ms << " ms\n";
        std::cout << "  average latency after warmup: " << average_latency_ms << " ms\n";
        std::cout << "  FPS after warmup: " << fps << " images/s\n";
        std::cout << "  postprocess: " << elapsed_ms(postprocess_begin, postprocess_end) << " ms\n";
        std::cout << "  total without load: " << elapsed_ms(preprocess_begin, postprocess_end) << " ms\n";

        return model.is_ready() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "classification_demo failed: " << error.what() << '\n';
        return 1;
    }
}
