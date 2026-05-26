#include "segmentor_imp.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

struct SegOutputLayout {
    enum class Kind {
        SingleChannel,
        ChannelFirst,
        ChannelLast,
    };

    Kind kind{Kind::SingleChannel};
    std::size_t height{};
    std::size_t width{};
    std::size_t class_count{1};
    std::size_t batch_stride{};
};

void check_cuda(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + " failed: " + cudaGetErrorString(status));
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open segmentor config: " + path);
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

SegmentorConfig parse_segmentor_config(const std::string& config_path) {
    SegmentorConfig config{};
    if (config_path.empty()) {
        return config;
    }
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("segmentor config file does not exist: " + config_path);
    }

    const std::string json = strip_json_comments(read_text_file(config_path));
    config.input_color_format = parse_color_format(parse_json_string(json, "input_color_format", "BGR"), config.input_color_format);
    config.keep_ratio = parse_json_bool(json, "keep_ratio", config.keep_ratio);
    config.mean = parse_json_float3(json, "mean", config.mean);
    config.std = parse_json_float3(json, "std", config.std);
    config.batch_size = parse_json_int(json, "batch_size", config.batch_size);
    config.height = parse_json_int(json, "height", config.height);
    config.width = parse_json_int(json, "width", config.width);
    config.use_fp16 = parse_json_bool(json, "use_fp16", config.use_fp16);
    config.mask_threshold = parse_json_float(json, "mask_threshold", config.mask_threshold);
    return config;
}

ModelConfig to_model_config(const SegmentorConfig& config) {
    ModelConfig model_config{};
    model_config.batch_size = std::max(config.batch_size, 1);
    model_config.input_height = std::max(config.height, 1);
    model_config.input_width = std::max(config.width, 1);
    model_config.use_fp16 = config.use_fp16;
    return model_config;
}

void validate_config(const SegmentorConfig& config) {
    if (config.batch_size <= 0) {
        throw std::runtime_error("SegmentorConfig.batch_size must be positive");
    }
    if (config.height <= 0) {
        throw std::runtime_error("SegmentorConfig.height must be positive");
    }
    if (config.width <= 0) {
        throw std::runtime_error("SegmentorConfig.width must be positive");
    }
    if (!std::isfinite(config.mask_threshold) || config.mask_threshold < 0.0F || config.mask_threshold > 1.0F) {
        throw std::runtime_error("SegmentorConfig.mask_threshold must be in [0, 1]");
    }
    for (float value : config.std) {
        if (std::abs(value) <= std::numeric_limits<float>::epsilon()) {
            throw std::runtime_error("SegmentorConfig.std must not contain zero");
        }
    }
}

SegmentorConfig validate_and_return(SegmentorConfig config) {
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

std::size_t element_size(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(std::uint16_t);
    case nvinfer1::DataType::kINT8:
        return sizeof(std::int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(std::int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kUINT8:
        return sizeof(std::uint8_t);
#endif
    default:
        return 0U;
    }
}

float read_float_value(const void* data, nvinfer1::DataType dtype, std::size_t index) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return static_cast<const float*>(data)[index];
    case nvinfer1::DataType::kHALF: {
        const auto half_bits = static_cast<const std::uint16_t*>(data)[index];
        const std::uint32_t sign = static_cast<std::uint32_t>(half_bits & 0x8000U) << 16U;
        std::uint32_t exponent = (half_bits & 0x7C00U) >> 10U;
        std::uint32_t mantissa = half_bits & 0x03FFU;
        std::uint32_t float_bits = 0U;
        if (exponent == 0U) {
            if (mantissa == 0U) {
                float_bits = sign;
            } else {
                exponent = 1U;
                while ((mantissa & 0x0400U) == 0U) {
                    mantissa <<= 1U;
                    --exponent;
                }
                mantissa &= 0x03FFU;
                float_bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
            }
        } else if (exponent == 0x1FU) {
            float_bits = sign | 0x7F800000U | (mantissa << 13U);
        } else {
            float_bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
        }
        float value = 0.0F;
        std::memcpy(&value, &float_bits, sizeof(float));
        return value;
    }
    case nvinfer1::DataType::kINT8:
        return static_cast<float>(static_cast<const std::int8_t*>(data)[index]);
    case nvinfer1::DataType::kINT32:
        return static_cast<float>(static_cast<const std::int32_t*>(data)[index]);
    case nvinfer1::DataType::kBOOL:
        return static_cast<const bool*>(data)[index] ? 1.0F : 0.0F;
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kUINT8:
        return static_cast<float>(static_cast<const std::uint8_t*>(data)[index]);
#endif
    default:
        return 0.0F;
    }
}

unsigned char read_label_value(const void* data, nvinfer1::DataType dtype, std::size_t index) {
    switch (dtype) {
    case nvinfer1::DataType::kINT8:
        return static_cast<unsigned char>(std::max(0, static_cast<int>(static_cast<const std::int8_t*>(data)[index])));
    case nvinfer1::DataType::kINT32:
        return static_cast<unsigned char>(std::clamp(static_cast<const std::int32_t*>(data)[index], 0, 255));
    case nvinfer1::DataType::kBOOL:
        return static_cast<const bool*>(data)[index] ? 255U : 0U;
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kUINT8:
        return static_cast<const std::uint8_t*>(data)[index];
#endif
    default:
        return static_cast<unsigned char>(std::clamp(static_cast<int>(std::round(read_float_value(data, dtype, index))), 0, 255));
    }
}

bool is_integral_output(nvinfer1::DataType dtype) {
    return dtype == nvinfer1::DataType::kINT8 || dtype == nvinfer1::DataType::kINT32 || dtype == nvinfer1::DataType::kBOOL
#if NV_TENSORRT_MAJOR >= 8
           || dtype == nvinfer1::DataType::kUINT8
#endif
        ;
}

std::vector<int> positive_dims(const nvinfer1::Dims& dims) {
    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(std::max(dims.nbDims, 0)));
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0) {
            values.push_back(dims.d[i]);
        }
    }
    return values;
}

SegOutputLayout infer_segmentation_output_layout(const Model& model, const std::string& output_name, std::size_t element_count) {
    for (const auto& tensor : model.tensors()) {
        if (tensor.name != output_name) {
            continue;
        }

        const std::vector<int> dims = positive_dims(tensor.dims);
        if (dims.empty()) {
            break;
        }

        const std::size_t batch_size = static_cast<std::size_t>(std::max(model.get_batch_size(), 1));
        SegOutputLayout layout{};
        if (dims.size() >= 4U) {
            const std::size_t c_first = static_cast<std::size_t>(dims[dims.size() - 3U]);
            const std::size_t h_first = static_cast<std::size_t>(dims[dims.size() - 2U]);
            const std::size_t w_first = static_cast<std::size_t>(dims[dims.size() - 1U]);
            const std::size_t h_last = static_cast<std::size_t>(dims[dims.size() - 3U]);
            const std::size_t w_last = static_cast<std::size_t>(dims[dims.size() - 2U]);
            const std::size_t c_last = static_cast<std::size_t>(dims[dims.size() - 1U]);

            if (c_last > 1U && c_last <= 256U && h_last > 1U && w_last > 1U) {
                layout.kind = SegOutputLayout::Kind::ChannelLast;
                layout.height = h_last;
                layout.width = w_last;
                layout.class_count = c_last;
            } else {
                layout.kind = c_first > 1U ? SegOutputLayout::Kind::ChannelFirst : SegOutputLayout::Kind::SingleChannel;
                layout.height = h_first;
                layout.width = w_first;
                layout.class_count = c_first;
            }
        } else if (dims.size() == 3U) {
            const std::size_t d0 = static_cast<std::size_t>(dims[0]);
            const std::size_t d1 = static_cast<std::size_t>(dims[1]);
            const std::size_t d2 = static_cast<std::size_t>(dims[2]);
            if (d0 <= 256U && d0 > 1U && d1 > 1U && d2 > 1U && element_count == d0 * d1 * d2 * batch_size) {
                layout.kind = SegOutputLayout::Kind::ChannelFirst;
                layout.height = d1;
                layout.width = d2;
                layout.class_count = d0;
            } else if (d2 <= 256U && d2 > 1U && d0 > 1U && d1 > 1U) {
                layout.kind = SegOutputLayout::Kind::ChannelLast;
                layout.height = d0;
                layout.width = d1;
                layout.class_count = d2;
            } else {
                layout.kind = SegOutputLayout::Kind::SingleChannel;
                layout.height = d1;
                layout.width = d2;
                layout.class_count = 1U;
            }
        } else if (dims.size() == 2U) {
            layout.kind = SegOutputLayout::Kind::SingleChannel;
            layout.height = static_cast<std::size_t>(dims[0]);
            layout.width = static_cast<std::size_t>(dims[1]);
            layout.class_count = 1U;
        }

        const std::size_t expected_per_batch = layout.height * layout.width * layout.class_count;
        if (layout.height > 0U && layout.width > 0U && layout.class_count > 0U && expected_per_batch > 0U &&
            expected_per_batch * batch_size <= element_count) {
            layout.batch_stride = expected_per_batch;
            return layout;
        }
    }

    const std::size_t batch_size = static_cast<std::size_t>(std::max(model.get_batch_size(), 1));
    const std::size_t per_batch = element_count / batch_size;
    const std::size_t side = static_cast<std::size_t>(std::sqrt(static_cast<double>(per_batch)));
    if (side > 0U && side * side == per_batch) {
        return SegOutputLayout{SegOutputLayout::Kind::SingleChannel, side, side, 1U, per_batch};
    }

    throw std::runtime_error("failed to infer segmentation output layout from tensor shape");
}

std::size_t infer_input_channel_count(const Model& model,
                                      const std::string& input_name,
                                      std::size_t input_height,
                                      std::size_t input_width) {
    const std::size_t plane_elements = input_height * input_width;
    if (plane_elements == 0U) {
        return 0U;
    }

    for (const auto& tensor : model.tensors()) {
        if (tensor.name != input_name) {
            continue;
        }
        const std::vector<int> dims = positive_dims(tensor.dims);
        if (dims.size() >= 3U) {
            const std::size_t candidate = static_cast<std::size_t>(dims[dims.size() - 3U]);
            if (candidate == 1U || candidate == 3U) {
                return candidate;
            }
            const std::size_t channel_last_candidate = static_cast<std::size_t>(dims.back());
            if (channel_last_candidate == 1U || channel_last_candidate == 3U) {
                return channel_last_candidate;
            }
        }
    }
    return 3U;
}

cv::Mat score_map_from_output(const void* output,
                              nvinfer1::DataType dtype,
                              const SegOutputLayout& layout,
                              std::size_t batch_offset,
                              std::size_t class_index) {
    cv::Mat score_map(static_cast<int>(layout.height), static_cast<int>(layout.width), CV_32FC1);
    auto* dst = score_map.ptr<float>();
    const std::size_t mask_elements = layout.height * layout.width;

    if (layout.kind == SegOutputLayout::Kind::ChannelFirst) {
        const std::size_t class_offset = batch_offset + class_index * mask_elements;
        for (std::size_t i = 0; i < mask_elements; ++i) {
            dst[i] = read_float_value(output, dtype, class_offset + i);
        }
        return score_map;
    }

    for (std::size_t y = 0; y < layout.height; ++y) {
        for (std::size_t x = 0; x < layout.width; ++x) {
            const std::size_t src_index = batch_offset + (y * layout.width + x) * layout.class_count + class_index;
            dst[y * layout.width + x] = read_float_value(output, dtype, src_index);
        }
    }
    return score_map;
}

float sigmoid(float value) noexcept {
    return 1.0F / (1.0F + std::exp(-value));
}

}  // namespace

SegmentationResult::SegmentationResult(int mask_height, int mask_width, std::unique_ptr<unsigned char[]> mask_data) noexcept
    : height(mask_height), width(mask_width), mask(std::move(mask_data)) {}

SegmentationResult::SegmentationResult(const SegmentationResult& other) : height(other.height), width(other.width) {
    if (!other.empty()) {
        mask = std::make_unique<unsigned char[]>(other.size());
        std::copy(other.mask.get(), other.mask.get() + other.size(), mask.get());
    }
}

SegmentationResult& SegmentationResult::operator=(const SegmentationResult& other) {
    if (this == &other) {
        return *this;
    }
    height = other.height;
    width = other.width;
    if (other.empty()) {
        mask.reset();
        return *this;
    }
    auto copied_mask = std::make_unique<unsigned char[]>(other.size());
    std::copy(other.mask.get(), other.mask.get() + other.size(), copied_mask.get());
    mask = std::move(copied_mask);
    return *this;
}

std::size_t SegmentationResult::size() const noexcept {
    if (height <= 0 || width <= 0 || mask == nullptr) {
        return 0U;
    }
    return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
}

bool SegmentationResult::empty() const noexcept {
    return size() == 0U;
}

Segmentor::Segmentor(const std::string& model_path, const std::string& config_path)
    : impl_(std::make_unique<Impl>(model_path, config_path)) {}

Segmentor::Segmentor(const std::string& model_path, SegmentorConfig config)
    : impl_(std::make_unique<Impl>(model_path, config)) {}

Segmentor::~Segmentor() = default;
Segmentor::Segmentor(Segmentor&&) noexcept = default;
Segmentor& Segmentor::operator=(Segmentor&&) noexcept = default;

SegmentationResult Segmentor::inference(const cv::Mat& image) {
    return impl_->inference(image);
}

std::vector<SegmentationResult> Segmentor::inference(const std::vector<cv::Mat>& images) {
    return impl_->inference(images);
}

void Segmentor::inference(const std::vector<cv::Mat>& images, std::vector<SegmentationResult>& seg_results) {
    impl_->inference(images, seg_results);
}

void Segmentor::inference(const std::vector<ImageProxy>& images, std::vector<SegmentationResult>& seg_results) {
    impl_->inference(images, seg_results);
}

int Segmentor::get_batch_size() const noexcept {
    return impl_ == nullptr ? 0 : impl_->get_batch_size();
}

bool Segmentor::is_ready() const noexcept {
    return impl_ != nullptr && impl_->is_ready();
}

Segmentor::Impl::Impl(const std::string& model_path, const std::string& config_path)
    : Impl(model_path, parse_segmentor_config(config_path)) {}

Segmentor::Impl::Impl(const std::string& model_path, SegmentorConfig config)
    : config_(validate_and_return(config)), model_(model_path, TaskType::Segmentation, to_model_config(config_)) {}

SegmentationResult Segmentor::Impl::inference(const cv::Mat& image) {
    std::vector<SegmentationResult> results;
    inference(std::vector<cv::Mat>{image}, results);
    return results.empty() ? SegmentationResult{} : std::move(results.front());
}

std::vector<SegmentationResult> Segmentor::Impl::inference(const std::vector<cv::Mat>& images) {
    std::vector<SegmentationResult> results;
    inference(images, results);
    return results;
}

void Segmentor::Impl::inference(const std::vector<cv::Mat>& images, std::vector<SegmentationResult>& seg_results) {
    validate_request(images.size());
    seg_results.clear();
    seg_results.reserve(images.size());

    for (const auto& image : images) {
        PreprocessMeta meta;
        preprocess_to_device(image, meta);
        model_.inference_device_input();
        seg_results.push_back(decode_output(meta));
    }
}

void Segmentor::Impl::inference(const std::vector<ImageProxy>& images, std::vector<SegmentationResult>& seg_results) {
    validate_request(images.size());
    std::vector<cv::Mat> mats;
    mats.reserve(images.size());
    for (const auto& image : images) {
        mats.push_back(image_proxy_to_mat(image));
    }
    inference(mats, seg_results);
}

int Segmentor::Impl::get_batch_size() const noexcept {
    return model_.get_batch_size();
}

bool Segmentor::Impl::is_ready() const noexcept {
    return model_.is_ready();
}

void Segmentor::Impl::preprocess_to_device(const cv::Mat& image, PreprocessMeta& meta) {
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
    const std::size_t input_channels = infer_input_channel_count(model_, input_name, input_height, input_width);
    const std::size_t expected_bytes = input_channels * plane_elements * sizeof(float);
    if (input_device == nullptr) {
        throw std::runtime_error("failed to get device input buffer: " + input_name);
    }
    if (input_channels != 1U && input_channels != 3U) {
        throw std::runtime_error("Segmentor only supports 1-channel or 3-channel float input tensors");
    }
    if (input_bytes < expected_bytes) {
        throw std::runtime_error("device input buffer is smaller than preprocessed image");
    }

    meta.original_width = image.cols;
    meta.original_height = image.rows;

    cv::Mat model_input;
    if (input_channels == 1U) {
        if (image.channels() == 1) {
            model_input = image;
        } else if (image.channels() == 3) {
            cv::cvtColor(image, model_input, cv::COLOR_BGR2GRAY);
        } else if (image.channels() == 4) {
            cv::cvtColor(image, model_input, cv::COLOR_BGRA2GRAY);
        } else {
            throw std::runtime_error("Segmentor expects 1, 3, or 4 channel image");
        }
    } else {
        if (config_.input_color_format == InputColorFormat::Gray) {
            cv::Mat gray_input;
            if (image.channels() == 1) {
                gray_input = image;
            } else if (image.channels() == 3) {
                cv::cvtColor(image, gray_input, cv::COLOR_BGR2GRAY);
            } else if (image.channels() == 4) {
                cv::cvtColor(image, gray_input, cv::COLOR_BGRA2GRAY);
            } else {
                throw std::runtime_error("Segmentor expects 1, 3, or 4 channel image");
            }
            cv::cvtColor(gray_input, model_input, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 1) {
            cv::cvtColor(image, model_input, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 3) {
            if (config_.input_color_format == InputColorFormat::Rgb) {
                model_input = image;
            } else {
                cv::cvtColor(image, model_input, cv::COLOR_BGR2RGB);
            }
        } else if (image.channels() == 4) {
            if (config_.input_color_format == InputColorFormat::Rgb) {
                cv::cvtColor(image, model_input, cv::COLOR_RGBA2RGB);
            } else {
                cv::cvtColor(image, model_input, cv::COLOR_BGRA2RGB);
            }
        } else {
            throw std::runtime_error("Segmentor expects 1, 3, or 4 channel image");
        }
    }

    cv::Mat resized_input;
    if (config_.keep_ratio) {
        const double scale = std::min(static_cast<double>(config_.width) / static_cast<double>(image.cols),
                                      static_cast<double>(config_.height) / static_cast<double>(image.rows));
        meta.resized_width = std::max(1, static_cast<int>(std::round(static_cast<double>(image.cols) * scale)));
        meta.resized_height = std::max(1, static_cast<int>(std::round(static_cast<double>(image.rows) * scale)));
        const int pad_x = config_.width - meta.resized_width;
        const int pad_y = config_.height - meta.resized_height;
        meta.pad_left = std::max(0, pad_x / 2);
        meta.pad_top = std::max(0, pad_y / 2);
        const int right = std::max(0, pad_x - meta.pad_left);
        const int bottom = std::max(0, pad_y - meta.pad_top);

        cv::Mat scaled_input;
        cv::resize(model_input, scaled_input, cv::Size{meta.resized_width, meta.resized_height}, 0.0, 0.0, cv::INTER_CUBIC);
        cv::copyMakeBorder(scaled_input,
                           resized_input,
                           meta.pad_top,
                           bottom,
                           meta.pad_left,
                           right,
                           cv::BORDER_CONSTANT,
                           cv::Scalar{0.0, 0.0, 0.0});
    } else {
        meta.resized_width = config_.width;
        meta.resized_height = config_.height;
        meta.pad_left = 0;
        meta.pad_top = 0;
        cv::resize(model_input, resized_input, cv::Size{config_.width, config_.height}, 0.0, 0.0, cv::INTER_CUBIC);
    }

    if (input_channels == 1U) {
        cv::Mat resized_fp32;
        resized_input.convertTo(resized_fp32, CV_32FC1, 1.0 / 255.0, 0.0);

        cv::Mat input_plane;
        cv::subtract(resized_fp32, cv::Scalar{config_.mean[0]}, input_plane);
        cv::divide(input_plane, cv::Scalar{config_.std[0]}, input_plane);
        check_cuda(cudaMemcpyAsync(input_device, input_plane.ptr<float>(), expected_bytes, cudaMemcpyHostToDevice, model_.cuda_stream()),
                   "cudaMemcpyAsync segmentor input");
    } else {
        cv::Mat resized_fp32;
        resized_input.convertTo(resized_fp32, CV_32FC3, 1.0 / 255.0, 0.0);

        cv::Mat normalized_input;
        cv::subtract(resized_fp32, cv::Scalar{config_.mean[0], config_.mean[1], config_.mean[2]}, normalized_input);
        cv::divide(normalized_input, cv::Scalar{config_.std[0], config_.std[1], config_.std[2]}, normalized_input);

        std::vector<float> chw_data(input_channels * plane_elements);
        std::vector<cv::Mat> chw_planes;
        chw_planes.reserve(3);
        for (int c = 0; c < 3; ++c) {
            chw_planes.emplace_back(config_.height,
                                    config_.width,
                                    CV_32FC1,
                                    chw_data.data() + static_cast<std::size_t>(c) * plane_elements);
        }
        cv::split(normalized_input, chw_planes);
        check_cuda(cudaMemcpyAsync(input_device, chw_data.data(), expected_bytes, cudaMemcpyHostToDevice, model_.cuda_stream()),
                   "cudaMemcpyAsync segmentor input");
    }
    check_cuda(cudaStreamSynchronize(model_.cuda_stream()), "cudaStreamSynchronize after segmentor preprocess");
}

void Segmentor::Impl::validate_request(std::size_t image_count) const {
    if (image_count == 0) {
        return;
    }
    if (image_count > static_cast<std::size_t>(get_batch_size())) {
        throw std::runtime_error("input image count exceeds segmentor batch_size");
    }
}

SegmentationResult Segmentor::Impl::decode_output(const PreprocessMeta& meta) const {
    if (model_.output_names().empty()) {
        throw std::runtime_error("model has no output tensor");
    }

    const std::string& output_name = model_.output_names().front();
    const void* output = model_.host_buffer(output_name);
    if (output == nullptr) {
        return SegmentationResult{};
    }

    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    for (const auto& tensor : model_.tensors()) {
        if (tensor.name == output_name) {
            dtype = tensor.dtype;
            break;
        }
    }
    const std::size_t bytes = model_.buffer_size(output_name);
    const std::size_t elem_size = element_size(dtype);
    if (elem_size == 0U || bytes < elem_size) {
        return SegmentationResult{};
    }
    const std::size_t element_count = bytes / elem_size;
    const auto layout = infer_segmentation_output_layout(model_, output_name, element_count);
    const std::size_t mask_elements = layout.height * layout.width;
    const std::size_t batch_offset = 0U;
    auto result_mask = std::make_unique<unsigned char[]>(static_cast<std::size_t>(meta.original_width) * static_cast<std::size_t>(meta.original_height));
    cv::Mat result_mat(meta.original_height, meta.original_width, CV_8UC1, result_mask.get());
    result_mat.setTo(cv::Scalar{0});

    if (layout.kind == SegOutputLayout::Kind::SingleChannel) {
        if (is_integral_output(dtype)) {
            cv::Mat model_mask(static_cast<int>(layout.height), static_cast<int>(layout.width), CV_8UC1, cv::Scalar{0});
            auto* mask_ptr = model_mask.ptr<unsigned char>();
            for (std::size_t i = 0; i < mask_elements; ++i) {
                mask_ptr[i] = read_label_value(output, dtype, batch_offset + i);
            }
            cv::resize(model_mask, result_mat, cv::Size{meta.original_width, meta.original_height}, 0.0, 0.0, cv::INTER_NEAREST);
        } else {
            cv::Mat score_map(static_cast<int>(layout.height), static_cast<int>(layout.width), CV_32FC1);
            auto* scores = score_map.ptr<float>();
            for (std::size_t i = 0; i < mask_elements; ++i) {
                scores[i] = read_float_value(output, dtype, batch_offset + i);
            }
            cv::Mat resized_scores;
            cv::resize(score_map, resized_scores, cv::Size{meta.original_width, meta.original_height}, 0.0, 0.0, cv::INTER_LINEAR);
            for (int y = 0; y < resized_scores.rows; ++y) {
                const auto* src = resized_scores.ptr<float>(y);
                auto* dst = result_mat.ptr<unsigned char>(y);
                for (int x = 0; x < resized_scores.cols; ++x) {
                    const float probability = sigmoid(src[x]);
                    dst[x] = std::isfinite(probability) && probability > config_.mask_threshold ? 255U : 0U;
                }
            }
        }
    } else {
        std::vector<cv::Mat> resized_scores;
        resized_scores.reserve(layout.class_count);
        for (std::size_t c = 0; c < layout.class_count; ++c) {
            cv::Mat score_map = score_map_from_output(output, dtype, layout, batch_offset, c);
            cv::Mat resized_score;
            cv::resize(score_map, resized_score, cv::Size{meta.original_width, meta.original_height}, 0.0, 0.0, cv::INTER_LINEAR);
            resized_scores.push_back(std::move(resized_score));
        }
        for (int y = 0; y < meta.original_height; ++y) {
            auto* dst = result_mat.ptr<unsigned char>(y);
            for (int x = 0; x < meta.original_width; ++x) {
                std::size_t best_class = 0U;
                float best_score = resized_scores.front().at<float>(y, x);
                for (std::size_t c = 1; c < resized_scores.size(); ++c) {
                    const float score = resized_scores[c].at<float>(y, x);
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }
                dst[x] = static_cast<unsigned char>(std::min<std::size_t>(best_class, 255U));
            }
        }
    }

    return SegmentationResult{meta.original_height, meta.original_width, std::move(result_mask)};
}

}  // namespace myai_gpu
