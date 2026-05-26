#pragma once

#include "detector.hpp"
#include "model.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace myai_gpu {

class Detector::Impl {
public:
    explicit Impl(const std::string& model_path, const std::string& config_path = {});
    Impl(const std::string& model_path, DetectorConfig config);

    [[nodiscard]] DetectionResult inference(const cv::Mat& image);
    [[nodiscard]] std::vector<DetectionResult> inference(const std::vector<cv::Mat>& images);
    void inference(const std::vector<cv::Mat>& images, std::vector<DetectionResult>& det_results);
    void inference(const std::vector<ImageProxy>& images, std::vector<DetectionResult>& det_results);

    [[nodiscard]] DetRuntimeConfig get_runtime_config() const noexcept;
    void set_runtime_config(const DetRuntimeConfig& config);
    [[nodiscard]] int get_batch_size() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;

private:
    struct PreprocessMeta {
        int original_width{};
        int original_height{};
        float scale_x{1.0F};
        float scale_y{1.0F};
        float pad_x{0.0F};
        float pad_y{0.0F};
    };

    void preprocess_to_device(const cv::Mat& image, PreprocessMeta& meta);
    void validate_request(std::size_t image_count) const;
    [[nodiscard]] DetectionResult decode_output(const PreprocessMeta& meta) const;

private:
    DetectorConfig config_{};
    Model model_;
};

}  // namespace myai_gpu
