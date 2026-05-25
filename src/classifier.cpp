#include "classifier_imp.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace myai_gpu {
namespace {

void check_cuda(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + " failed: " + cudaGetErrorString(status));
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open classifier config: " + path);
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

std::string strip_json_comments(std::string text) {
    std::string output;
    output.reserve(text.size());

    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_string) {
            output.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            output.push_back(ch);
            continue;
        }
        if (ch == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            while (i < text.size() && text[i] != '\n') {
                ++i;
            }
            if (i < text.size()) {
                output.push_back('\n');
            }
            continue;
        }
        output.push_back(ch);
    }
    return output;
}

std::string parse_json_string(const std::string& json, std::string_view key, const std::string& fallback) {
    const std::string quoted_key = '"' + std::string{key} + '"';
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t colon_pos = json.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t begin_quote = json.find('"', colon_pos + 1);
    if (begin_quote == std::string::npos) {
        return fallback;
    }
    const std::size_t end_quote = json.find('"', begin_quote + 1);
    if (end_quote == std::string::npos) {
        return fallback;
    }
    return json.substr(begin_quote + 1, end_quote - begin_quote - 1);
}

int parse_json_int(const std::string& json, std::string_view key, int fallback) {
    const std::string quoted_key = '"' + std::string{key} + '"';
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t colon_pos = json.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t value_begin = json.find_first_of("-0123456789", colon_pos + 1);
    if (value_begin == std::string::npos) {
        return fallback;
    }
    const std::size_t value_end = json.find_first_not_of("0123456789", value_begin + (json[value_begin] == '-' ? 1 : 0));
    return std::stoi(json.substr(value_begin, value_end - value_begin));
}

bool parse_json_bool(const std::string& json, std::string_view key, bool fallback) {
    const std::string quoted_key = '"' + std::string{key} + '"';
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t colon_pos = json.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t value_begin = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_begin == std::string::npos) {
        return fallback;
    }
    if (json.compare(value_begin, 4, "true") == 0) {
        return true;
    }
    if (json.compare(value_begin, 5, "false") == 0) {
        return false;
    }
    return fallback;
}

std::array<float, 3> parse_json_float3(const std::string& json, std::string_view key, std::array<float, 3> fallback) {
    const std::string quoted_key = '"' + std::string{key} + '"';
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t open = json.find('[', key_pos + quoted_key.size());
    const std::size_t close = json.find(']', open == std::string::npos ? key_pos : open);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return fallback;
    }

    std::array<float, 3> values{};
    std::stringstream stream{json.substr(open + 1, close - open - 1)};
    for (std::size_t i = 0; i < values.size(); ++i) {
        stream >> values[i];
        if (!stream) {
            return fallback;
        }
        stream >> std::ws;
        if (i + 1 < values.size()) {
            if (stream.peek() != ',') {
                return fallback;
            }
            stream.ignore(1);
        }
    }
    return values;
}

InputColorFormat parse_color_format(std::string value, InputColorFormat fallback) {
    std::ranges::transform(value, value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    if (value == "BGR") {
        return InputColorFormat::Bgr;
    }
    if (value == "RGB") {
        return InputColorFormat::Rgb;
    }
    if (value == "GRAY" || value == "GREY") {
        return InputColorFormat::Gray;
    }
    return fallback;
}

ClassifierConfig parse_classifier_config(const std::string& config_path) {
    ClassifierConfig config{};
    if (config_path.empty()) {
        return config;
    }
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("classifier config file does not exist: " + config_path);
    }

    const std::string json = strip_json_comments(read_text_file(config_path));
    config.input_color_format = parse_color_format(parse_json_string(json, "input_color_format", "BGR"), config.input_color_format);
    config.mean = parse_json_float3(json, "mean", config.mean);
    config.std = parse_json_float3(json, "std", config.std);
    config.batch_size = parse_json_int(json, "batch_size", config.batch_size);
    config.height = parse_json_int(json, "height", config.height);
    config.width = parse_json_int(json, "width", config.width);
    config.topk = parse_json_int(json, "topk", config.topk);
    config.use_fp16 = parse_json_bool(json, "use_fp16", config.use_fp16);
    return config;
}

ModelConfig to_model_config(const ClassifierConfig& config) {
    ModelConfig model_config{};
    model_config.batch_size = std::max(config.batch_size, 1);
    model_config.input_height = std::max(config.height, 1);
    model_config.input_width = std::max(config.width, 1);
    model_config.use_fp16 = config.use_fp16;
    return model_config;
}

void validate_config(const ClassifierConfig& config) {
    if (config.batch_size <= 0) {
        throw std::runtime_error("ClassifierConfig.batch_size must be positive");
    }
    if (config.height <= 0) {
        throw std::runtime_error("ClassifierConfig.height must be positive");
    }
    if (config.width <= 0) {
        throw std::runtime_error("ClassifierConfig.width must be positive");
    }
    if (config.topk <= 0) {
        throw std::runtime_error("ClassifierConfig.topk must be positive");
    }
    for (float value : config.std) {
        if (std::abs(value) <= std::numeric_limits<float>::epsilon()) {
            throw std::runtime_error("ClassifierConfig.std must not contain zero");
        }
    }
}

cv::Mat image_proxy_to_mat(const ImageProxy& image) {
    if (image.data == nullptr) {
        throw std::runtime_error("ImageProxy data is null");
    }
    if (image.rows <= 0 || image.cols <= 0) {
        throw std::runtime_error("ImageProxy size is invalid");
    }

    int type = CV_8UC3;
    if (image.channels == 1) {
        type = CV_8UC1;
    } else if (image.channels == 3) {
        type = CV_8UC3;
    } else if (image.channels == 4) {
        type = CV_8UC4;
    } else {
        throw std::runtime_error("ImageProxy only supports 1, 3, or 4 channels");
    }
    return cv::Mat{image.rows, image.cols, type, image.data};
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
    if (sum <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("invalid classification probability sum");
    }
    for (auto& probability : probabilities) {
        probability /= sum;
    }
    return probabilities;
}

}  // namespace

std::size_t ClassificationResult::class_id() const noexcept {
    return label_id.empty() ? 0U : static_cast<std::size_t>(label_id.front());
}

float ClassificationResult::score() const noexcept {
    return confidence.empty() ? 0.0F : confidence.front();
}

Classifier::Classifier(const std::string& model_path, const std::string& config_path)
    : impl_(std::make_unique<Impl>(model_path, config_path)) {}

Classifier::Classifier(const std::string& model_path, ClassifierConfig config)
    : impl_(std::make_unique<Impl>(model_path, config)) {}

Classifier::~Classifier() = default;
Classifier::Classifier(Classifier&&) noexcept = default;
Classifier& Classifier::operator=(Classifier&&) noexcept = default;

ClassificationResult Classifier::inference(const cv::Mat& image) {
    return impl_->inference(image);
}

std::vector<ClassificationResult> Classifier::inference(const std::vector<cv::Mat>& images) {
    return impl_->inference(images);
}

void Classifier::inference(const std::vector<cv::Mat>& images, std::vector<ClassificationResult>& results) {
    impl_->inference(images, results);
}

void Classifier::inference(const std::vector<ImageProxy>& images, std::vector<ClassificationResult>& results) {
    impl_->inference(images, results);
}

ClassifierPerformance Classifier::benchmark(const cv::Mat& image, int warmup_runs, int benchmark_runs) {
    return impl_->benchmark(image, warmup_runs, benchmark_runs);
}

int Classifier::get_batch_size() const noexcept {
    return impl_ == nullptr ? 0 : impl_->get_batch_size();
}

bool Classifier::is_ready() const noexcept {
    return impl_ != nullptr && impl_->is_ready();
}

Classifier::Impl::Impl(const std::string& model_path, const std::string& config_path)
    : Impl(model_path, parse_classifier_config(config_path)) {}

Classifier::Impl::Impl(const std::string& model_path, ClassifierConfig config)
    : config_(config), model_(model_path, TaskType::Classification, to_model_config(config)) {
    validate_config(config_);
}

ClassificationResult Classifier::Impl::inference(const cv::Mat& image) {
    std::vector<ClassificationResult> results;
    inference(std::vector<cv::Mat>{image}, results);
    return results.empty() ? ClassificationResult{} : results.front();
}

std::vector<ClassificationResult> Classifier::Impl::inference(const std::vector<cv::Mat>& images) {
    std::vector<ClassificationResult> results;
    inference(images, results);
    return results;
}

void Classifier::Impl::inference(const std::vector<cv::Mat>& images, std::vector<ClassificationResult>& results) {
    validate_request(images.size());
    results.clear();
    results.reserve(images.size());

    for (const auto& image : images) {
        preprocess_to_device(image);
        model_.inference_device_input();
        results.push_back(classify_output());
    }
}

void Classifier::Impl::inference(const std::vector<ImageProxy>& images, std::vector<ClassificationResult>& results) {
    validate_request(images.size());
    std::vector<cv::Mat> mats;
    mats.reserve(images.size());
    for (const auto& image : images) {
        mats.push_back(image_proxy_to_mat(image));
    }
    inference(mats, results);
}

ClassifierPerformance Classifier::Impl::benchmark(const cv::Mat& image, int warmup_runs, int benchmark_runs) {
    if (warmup_runs < 0) {
        throw std::runtime_error("warmup_runs must be non-negative");
    }
    if (benchmark_runs <= 0) {
        throw std::runtime_error("benchmark_runs must be positive");
    }

    for (int i = 0; i < warmup_runs; ++i) {
        static_cast<void>(inference(image));
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        static_cast<void>(inference(image));
    }
    const auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>{end - start}.count();
    const double average_ms = total_ms / static_cast<double>(benchmark_runs);
    return ClassifierPerformance{
        warmup_runs,
        benchmark_runs,
        total_ms,
        average_ms,
        average_ms > 0.0 ? 1000.0 / average_ms : 0.0,
    };
}

int Classifier::Impl::get_batch_size() const noexcept {
    return model_.get_batch_size();
}

bool Classifier::Impl::is_ready() const noexcept {
    return model_.is_ready();
}

void Classifier::Impl::preprocess_to_device(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("input image is empty");
    }
    if (model_.input_names().empty()) {
        throw std::runtime_error("model has no input tensor");
    }

    const std::string& input_name = model_.input_names().front();
    void* input_device = model_.device_buffer(input_name);
    const std::size_t input_bytes = model_.buffer_size(input_name);
    const std::size_t input_height = static_cast<std::size_t>(config_.height);
    const std::size_t input_width = static_cast<std::size_t>(config_.width);
    const std::size_t plane_elements = input_width * input_height;
    const std::size_t expected_bytes = 3U * plane_elements * sizeof(float);
    if (input_device == nullptr) {
        throw std::runtime_error("failed to get device input buffer: " + input_name);
    }
    if (input_bytes < expected_bytes) {
        throw std::runtime_error("device input buffer is smaller than preprocessed image");
    }

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(model_.cuda_stream());

    cv::cuda::GpuMat gpu_src;
    gpu_src.upload(image, cv_stream);

    cv::cuda::GpuMat gpu_rgb;
    if (image.channels() == 1) {
        cv::cuda::cvtColor(gpu_src, gpu_rgb, cv::COLOR_GRAY2RGB, 0, cv_stream);
    } else if (image.channels() == 3) {
        if (config_.input_color_format == InputColorFormat::Rgb) {
            gpu_rgb = gpu_src;
        } else {
            cv::cuda::cvtColor(gpu_src, gpu_rgb, cv::COLOR_BGR2RGB, 0, cv_stream);
        }
    } else if (image.channels() == 4) {
        if (config_.input_color_format == InputColorFormat::Rgb) {
            cv::cuda::cvtColor(gpu_src, gpu_rgb, cv::COLOR_RGBA2RGB, 0, cv_stream);
        } else {
            cv::cuda::cvtColor(gpu_src, gpu_rgb, cv::COLOR_BGRA2RGB, 0, cv_stream);
        }
    } else {
        throw std::runtime_error("Classifier expects 1, 3, or 4 channel image");
    }

    cv::cuda::GpuMat resized_rgb;
    cv::cuda::resize(gpu_rgb, resized_rgb, cv::Size{config_.width, config_.height}, 0.0, 0.0, cv::INTER_LINEAR, cv_stream);

    cv::cuda::GpuMat resized_rgb_fp32;
    resized_rgb.convertTo(resized_rgb_fp32, CV_32FC3, 1.0 / 255.0, 0.0, cv_stream);

    cv::cuda::GpuMat normalized_rgb;
    cv::cuda::subtract(resized_rgb_fp32, cv::Scalar{config_.mean[0], config_.mean[1], config_.mean[2]}, normalized_rgb, cv::noArray(), -1, cv_stream);
    cv::cuda::divide(normalized_rgb, cv::Scalar{config_.std[0], config_.std[1], config_.std[2]}, normalized_rgb, 1.0, -1, cv_stream);

    std::vector<cv::cuda::GpuMat> chw_planes;
    chw_planes.reserve(3);
    auto* input_float = static_cast<float*>(input_device);
    for (int c = 0; c < 3; ++c) {
        chw_planes.emplace_back(config_.height, config_.width, CV_32FC1, input_float + static_cast<std::size_t>(c) * plane_elements);
    }
    cv::cuda::split(normalized_rgb, chw_planes, cv_stream);
    check_cuda(cudaStreamSynchronize(model_.cuda_stream()), "cudaStreamSynchronize after classifier preprocess");
}

void Classifier::Impl::validate_request(std::size_t image_count) const {
    if (image_count == 0) {
        return;
    }
    if (image_count > static_cast<std::size_t>(get_batch_size())) {
        throw std::runtime_error("input image count exceeds classifier batch_size");
    }
}

ClassificationResult Classifier::Impl::classify_output() const {
    if (model_.output_names().empty()) {
        throw std::runtime_error("model has no output tensor");
    }

    const std::string& output_name = model_.output_names().front();
    const auto* logits = static_cast<const float*>(model_.host_buffer(output_name));
    const std::size_t count = model_.buffer_size(output_name) / sizeof(float);
    auto probabilities = softmax(logits, count);

    const int topk = std::min<int>(config_.topk, static_cast<int>(probabilities.size()));
    std::vector<int> indices(probabilities.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(), [&probabilities](int lhs, int rhs) {
        return probabilities[static_cast<std::size_t>(lhs)] > probabilities[static_cast<std::size_t>(rhs)];
    });

    ClassificationResult result;
    result.confidence.reserve(static_cast<std::size_t>(topk));
    result.label_id.reserve(static_cast<std::size_t>(topk));
    for (int i = 0; i < topk; ++i) {
        const int label = indices[static_cast<std::size_t>(i)];
        result.label_id.push_back(label);
        result.confidence.push_back(probabilities[static_cast<std::size_t>(label)]);
    }
    return result;
}

}  // namespace myai_gpu
