#include "detector_imp.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace myai_gpu {
namespace {

struct CandidateBox {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float score{};
    int label{};
};

struct YoloV8OutputLayout {
    bool attributes_first{true};
    std::size_t attribute_count{};
    std::size_t box_count{};
    std::size_t class_count{};
};

constexpr std::size_t kYoloV8BoxAttributes = 4;

void check_cuda(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + " failed: " + cudaGetErrorString(status));
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open detector config: " + path);
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

float parse_json_float(const std::string& json, std::string_view key, float fallback) {
    const std::string quoted_key = '"' + std::string{key} + '"';
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t colon_pos = json.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return fallback;
    }
    const std::size_t value_begin = json.find_first_of("-0123456789.", colon_pos + 1);
    if (value_begin == std::string::npos) {
        return fallback;
    }
    const std::size_t value_end = json.find_first_not_of("0123456789.eE+-", value_begin);
    return std::stof(json.substr(value_begin, value_end - value_begin));
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

DetectorConfig parse_detector_config(const std::string& config_path) {
    DetectorConfig config{};
    if (config_path.empty()) {
        return config;
    }
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("detector config file does not exist: " + config_path);
    }

    const std::string json = strip_json_comments(read_text_file(config_path));
    config.input_color_format = parse_color_format(parse_json_string(json, "input_color_format", "BGR"), config.input_color_format);
    config.keep_ratio = parse_json_bool(json, "keep_ratio", config.keep_ratio);
    config.runtime.iou_threshold = parse_json_float(json, "iou_threshold", config.runtime.iou_threshold);
    config.runtime.confidence_threshold = parse_json_float(json, "confidence_threshold", config.runtime.confidence_threshold);
    config.mean = parse_json_float3(json, "mean", config.mean);
    config.std = parse_json_float3(json, "std", config.std);
    config.batch_size = parse_json_int(json, "batch_size", config.batch_size);
    config.height = parse_json_int(json, "height", config.height);
    config.width = parse_json_int(json, "width", config.width);
    config.use_fp16 = parse_json_bool(json, "use_fp16", config.use_fp16);
    return config;
}

ModelConfig to_model_config(const DetectorConfig& config) {
    ModelConfig model_config{};
    model_config.batch_size = std::max(config.batch_size, 1);
    model_config.input_height = std::max(config.height, 1);
    model_config.input_width = std::max(config.width, 1);
    model_config.use_fp16 = config.use_fp16;
    return model_config;
}

void validate_runtime_config(const DetRuntimeConfig& config) {
    if (!std::isfinite(config.iou_threshold) || config.iou_threshold < 0.0F || config.iou_threshold > 1.0F) {
        throw std::runtime_error("DetRuntimeConfig.iou_threshold must be in [0, 1]");
    }
    if (!std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0.0F ||
        config.confidence_threshold > 1.0F) {
        throw std::runtime_error("DetRuntimeConfig.confidence_threshold must be in [0, 1]");
    }
}

void validate_config(const DetectorConfig& config) {
    if (config.batch_size <= 0) {
        throw std::runtime_error("DetectorConfig.batch_size must be positive");
    }
    if (config.height <= 0) {
        throw std::runtime_error("DetectorConfig.height must be positive");
    }
    if (config.width <= 0) {
        throw std::runtime_error("DetectorConfig.width must be positive");
    }
    for (float value : config.std) {
        if (std::abs(value) <= std::numeric_limits<float>::epsilon()) {
            throw std::runtime_error("DetectorConfig.std must not contain zero");
        }
    }
    validate_runtime_config(config.runtime);
}

DetectorConfig validate_and_return(DetectorConfig config) {
    validate_config(config);
    return config;
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

float intersection_over_union(const CandidateBox& lhs, const CandidateBox& rhs) noexcept {
    const float inter_x1 = std::max(lhs.x1, rhs.x1);
    const float inter_y1 = std::max(lhs.y1, rhs.y1);
    const float inter_x2 = std::min(lhs.x2, rhs.x2);
    const float inter_y2 = std::min(lhs.y2, rhs.y2);
    const float inter_w = std::max(0.0F, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0F, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float lhs_area = std::max(0.0F, lhs.x2 - lhs.x1) * std::max(0.0F, lhs.y2 - lhs.y1);
    const float rhs_area = std::max(0.0F, rhs.x2 - rhs.x1) * std::max(0.0F, rhs.y2 - rhs.y1);
    const float union_area = lhs_area + rhs_area - inter_area;
    return union_area <= std::numeric_limits<float>::epsilon() ? 0.0F : inter_area / union_area;
}

std::vector<CandidateBox> nms(std::vector<CandidateBox> candidates, float iou_threshold) {
    std::ranges::sort(candidates, [](const CandidateBox& lhs, const CandidateBox& rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<CandidateBox> kept;
    std::vector<bool> removed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        kept.push_back(candidates[i]);
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (!removed[j] && candidates[i].label == candidates[j].label &&
                intersection_over_union(candidates[i], candidates[j]) > iou_threshold) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

float clamp_float(float value, float low, float high) noexcept {
    return std::max(low, std::min(value, high));
}

CandidateBox restore_yolov8_box(float center_x,
                                float center_y,
                                float width,
                                float height,
                                float score,
                                int label,
                                int original_width,
                                int original_height,
                                float scale_x,
                                float scale_y,
                                float pad_x,
                                float pad_y) noexcept {
    CandidateBox candidate;
    candidate.x1 = (center_x - width / 2.0F - pad_x) / scale_x;
    candidate.y1 = (center_y - height / 2.0F - pad_y) / scale_y;
    candidate.x2 = (center_x + width / 2.0F - pad_x) / scale_x;
    candidate.y2 = (center_y + height / 2.0F - pad_y) / scale_y;
    candidate.x1 = clamp_float(candidate.x1, 0.0F, static_cast<float>(original_width));
    candidate.y1 = clamp_float(candidate.y1, 0.0F, static_cast<float>(original_height));
    candidate.x2 = clamp_float(candidate.x2, 0.0F, static_cast<float>(original_width));
    candidate.y2 = clamp_float(candidate.y2, 0.0F, static_cast<float>(original_height));
    candidate.score = score;
    candidate.label = label;
    return candidate;
}

void append_yolov8_candidate(const float* class_scores,
                             std::size_t class_stride,
                             std::size_t class_count,
                             float center_x,
                             float center_y,
                             float width,
                             float height,
                             int original_width,
                             int original_height,
                             float scale_x,
                             float scale_y,
                             float pad_x,
                             float pad_y,
                             const DetRuntimeConfig& runtime,
                             std::vector<CandidateBox>& candidates) {
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0F || height <= 0.0F) {
        return;
    }

    if (class_count == 0U) {
        return;
    }

    int label = 0;
    float score = class_scores[0];
    for (std::size_t class_id = 1; class_id < class_count; ++class_id) {
        const float class_score = class_scores[class_id * class_stride];
        if (class_score > score) {
            score = class_score;
            label = static_cast<int>(class_id);
        }
    }
    if (!std::isfinite(score) || score < runtime.confidence_threshold) {
        return;
    }

    auto candidate = restore_yolov8_box(center_x,
                                        center_y,
                                        width,
                                        height,
                                        score,
                                        label,
                                        original_width,
                                        original_height,
                                        scale_x,
                                        scale_y,
                                        pad_x,
                                        pad_y);
    if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1) {
        candidates.push_back(candidate);
    }
}

YoloV8OutputLayout infer_yolov8_output_layout(const Model& model, const std::string& output_name, std::size_t output_count) {
    for (const auto& tensor : model.tensors()) {
        if (tensor.name != output_name || tensor.dims.nbDims < 2) {
            continue;
        }

        const int last_dim = tensor.dims.d[tensor.dims.nbDims - 1];
        const int prev_dim = tensor.dims.d[tensor.dims.nbDims - 2];
        if (last_dim <= 0 || prev_dim <= 0) {
            break;
        }

        YoloV8OutputLayout layout{};
        if (last_dim > static_cast<int>(kYoloV8BoxAttributes) && last_dim <= prev_dim) {
            layout.attributes_first = false;
            layout.attribute_count = static_cast<std::size_t>(last_dim);
            layout.box_count = output_count / layout.attribute_count;
            layout.class_count = layout.attribute_count - kYoloV8BoxAttributes;
            if (layout.attribute_count * layout.box_count == output_count && layout.box_count > 0U && layout.class_count > 0U) {
                return layout;
            }
        }

        if (prev_dim > static_cast<int>(kYoloV8BoxAttributes) && prev_dim <= last_dim) {
            layout.attributes_first = true;
            layout.attribute_count = static_cast<std::size_t>(prev_dim);
            layout.box_count = output_count / layout.attribute_count;
            layout.class_count = layout.attribute_count - kYoloV8BoxAttributes;
            if (layout.attribute_count * layout.box_count == output_count && layout.box_count > 0U && layout.class_count > 0U) {
                return layout;
            }
        }

        break;
    }

    throw std::runtime_error("failed to infer YOLOv8 output layout from tensor shape");
}

}  // namespace

std::size_t DetectionResult::size() const noexcept {
    return label_id.size();
}

bool DetectionResult::empty() const noexcept {
    return label_id.empty();
}

Detector::Detector(const std::string& model_path, const std::string& config_path)
    : impl_(std::make_unique<Impl>(model_path, config_path)) {}

Detector::Detector(const std::string& model_path, DetectorConfig config)
    : impl_(std::make_unique<Impl>(model_path, config)) {}

Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

DetectionResult Detector::inference(const cv::Mat& image) {
    return impl_->inference(image);
}

std::vector<DetectionResult> Detector::inference(const std::vector<cv::Mat>& images) {
    return impl_->inference(images);
}

void Detector::inference(const std::vector<cv::Mat>& images, std::vector<DetectionResult>& det_results) {
    impl_->inference(images, det_results);
}

void Detector::inference(const std::vector<ImageProxy>& images, std::vector<DetectionResult>& det_results) {
    impl_->inference(images, det_results);
}

DetRuntimeConfig Detector::get_runtime_config() const noexcept {
    return impl_ == nullptr ? DetRuntimeConfig{} : impl_->get_runtime_config();
}

void Detector::set_runtime_config(const DetRuntimeConfig& config) {
    impl_->set_runtime_config(config);
}

int Detector::get_batch_size() const noexcept {
    return impl_ == nullptr ? 0 : impl_->get_batch_size();
}

bool Detector::is_ready() const noexcept {
    return impl_ != nullptr && impl_->is_ready();
}

Detector::Impl::Impl(const std::string& model_path, const std::string& config_path)
    : Impl(model_path, parse_detector_config(config_path)) {}

Detector::Impl::Impl(const std::string& model_path, DetectorConfig config)
    : config_(validate_and_return(config)), model_(model_path, TaskType::Detection, to_model_config(config_)) {}

DetectionResult Detector::Impl::inference(const cv::Mat& image) {
    std::vector<DetectionResult> results;
    inference(std::vector<cv::Mat>{image}, results);
    return results.empty() ? DetectionResult{} : results.front();
}

std::vector<DetectionResult> Detector::Impl::inference(const std::vector<cv::Mat>& images) {
    std::vector<DetectionResult> results;
    inference(images, results);
    return results;
}

void Detector::Impl::inference(const std::vector<cv::Mat>& images, std::vector<DetectionResult>& det_results) {
    validate_request(images.size());
    det_results.clear();
    det_results.reserve(images.size());

    for (const auto& image : images) {
        PreprocessMeta meta;
        preprocess_to_device(image, meta);
        model_.inference_device_input();
        det_results.push_back(decode_output(meta));
    }
}

void Detector::Impl::inference(const std::vector<ImageProxy>& images, std::vector<DetectionResult>& det_results) {
    validate_request(images.size());
    std::vector<cv::Mat> mats;
    mats.reserve(images.size());
    for (const auto& image : images) {
        mats.push_back(image_proxy_to_mat(image));
    }
    inference(mats, det_results);
}

DetRuntimeConfig Detector::Impl::get_runtime_config() const noexcept {
    return config_.runtime;
}

void Detector::Impl::set_runtime_config(const DetRuntimeConfig& config) {
    validate_runtime_config(config);
    config_.runtime = config;
}

int Detector::Impl::get_batch_size() const noexcept {
    return model_.get_batch_size();
}

bool Detector::Impl::is_ready() const noexcept {
    return model_.is_ready();
}

void Detector::Impl::preprocess_to_device(const cv::Mat& image, PreprocessMeta& meta) {
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

    meta.original_width = image.cols;
    meta.original_height = image.rows;

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
        throw std::runtime_error("Detector expects 1, 3, or 4 channel image");
    }

    cv::cuda::GpuMat resized_rgb;
    if (config_.keep_ratio) {
        const double scale = std::min(static_cast<double>(config_.width) / static_cast<double>(image.cols),
                                      static_cast<double>(config_.height) / static_cast<double>(image.rows));
        const int resized_width = std::max(1, static_cast<int>(std::round(static_cast<double>(image.cols) * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::round(static_cast<double>(image.rows) * scale)));
        meta.scale_x = static_cast<float>(scale);
        meta.scale_y = static_cast<float>(scale);
        meta.pad_x = static_cast<float>(config_.width - resized_width) / 2.0F;
        meta.pad_y = static_cast<float>(config_.height - resized_height) / 2.0F;
        const int top = std::max(0, static_cast<int>(std::round(static_cast<double>(meta.pad_y) - 0.1)));
        const int bottom = std::max(0, config_.height - resized_height - top);
        const int left = std::max(0, static_cast<int>(std::round(static_cast<double>(meta.pad_x) - 0.1)));
        const int right = std::max(0, config_.width - resized_width - left);

        cv::cuda::GpuMat scaled_rgb;
        cv::cuda::resize(gpu_rgb, scaled_rgb, cv::Size{resized_width, resized_height}, 0.0, 0.0, cv::INTER_LINEAR, cv_stream);
        cv::cuda::copyMakeBorder(scaled_rgb,
                                 resized_rgb,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv::BORDER_CONSTANT,
                                 cv::Scalar{114.0, 114.0, 114.0},
                                 cv_stream);
    } else {
        meta.scale_x = static_cast<float>(config_.width) / static_cast<float>(image.cols);
        meta.scale_y = static_cast<float>(config_.height) / static_cast<float>(image.rows);
        meta.pad_x = 0.0F;
        meta.pad_y = 0.0F;
        cv::cuda::resize(gpu_rgb, resized_rgb, cv::Size{config_.width, config_.height}, 0.0, 0.0, cv::INTER_LINEAR, cv_stream);
    }

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
    check_cuda(cudaStreamSynchronize(model_.cuda_stream()), "cudaStreamSynchronize after detector preprocess");
}

void Detector::Impl::validate_request(std::size_t image_count) const {
    if (image_count == 0) {
        return;
    }
    if (image_count > static_cast<std::size_t>(get_batch_size())) {
        throw std::runtime_error("input image count exceeds detector batch_size");
    }
}

DetectionResult Detector::Impl::decode_output(const PreprocessMeta& meta) const {
    if (model_.output_names().empty()) {
        throw std::runtime_error("model has no output tensor");
    }

    const std::string& output_name = model_.output_names().front();
    const auto* output = static_cast<const float*>(model_.host_buffer(output_name));
    const std::size_t output_count = model_.buffer_size(output_name) / sizeof(float);
    if (output == nullptr || output_count <= kYoloV8BoxAttributes) {
        return DetectionResult{};
    }

    const auto layout = infer_yolov8_output_layout(model_, output_name, output_count);
    std::vector<CandidateBox> candidates;
    candidates.reserve(layout.box_count);

    if (layout.attributes_first) {
        for (std::size_t box = 0; box < layout.box_count; ++box) {
            append_yolov8_candidate(output + kYoloV8BoxAttributes * layout.box_count + box,
                                    layout.box_count,
                                    layout.class_count,
                                    output[box],
                                    output[layout.box_count + box],
                                    output[2U * layout.box_count + box],
                                    output[3U * layout.box_count + box],
                                    meta.original_width,
                                    meta.original_height,
                                    meta.scale_x,
                                    meta.scale_y,
                                    meta.pad_x,
                                    meta.pad_y,
                                    config_.runtime,
                                    candidates);
        }
    } else {
        for (std::size_t box = 0; box < layout.box_count; ++box) {
            const std::size_t base = box * layout.attribute_count;
            append_yolov8_candidate(output + base + 4U,
                                    1U,
                                    layout.class_count,
                                    output[base],
                                    output[base + 1U],
                                    output[base + 2U],
                                    output[base + 3U],
                                    meta.original_width,
                                    meta.original_height,
                                    meta.scale_x,
                                    meta.scale_y,
                                    meta.pad_x,
                                    meta.pad_y,
                                    config_.runtime,
                                    candidates);
        }
    }

    DetectionResult result;
    for (const auto& candidate : nms(std::move(candidates), config_.runtime.iou_threshold)) {
        result.top_lefts.emplace_back(static_cast<int>(std::round(candidate.x1)), static_cast<int>(std::round(candidate.y1)));
        result.bottom_rights.emplace_back(static_cast<int>(std::round(candidate.x2)), static_cast<int>(std::round(candidate.y2)));
        result.confidence.push_back(candidate.score);
        result.label_id.push_back(candidate.label);
    }
    return result;
}

}  // namespace myai_gpu
