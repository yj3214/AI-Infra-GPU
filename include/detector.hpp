#pragma once

#include "classifier.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace myai_gpu {

struct DetectionResult {
    std::vector<cv::Point> top_lefts;
    std::vector<cv::Point> bottom_rights;
    std::vector<float> confidence;
    std::vector<int> label_id;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
};

struct DetRuntimeConfig {
    float iou_threshold{0.6F};
    float confidence_threshold{0.5F};
};

struct DetectorConfig {
    InputColorFormat input_color_format{InputColorFormat::Bgr};
    bool keep_ratio{true};
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> std{1.0F, 1.0F, 1.0F};
    int batch_size{1};
    int height{640};
    int width{640};
    bool use_fp16{false};
    DetRuntimeConfig runtime{};
};

class Detector {
public:
    explicit Detector(const std::string& model_path, const std::string& config_path = {});
    Detector(const std::string& model_path, DetectorConfig config);
    ~Detector();

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    [[nodiscard]] DetectionResult inference(const cv::Mat& image);
    [[nodiscard]] std::vector<DetectionResult> inference(const std::vector<cv::Mat>& images);
    void inference(const std::vector<cv::Mat>& images, std::vector<DetectionResult>& det_results);
    void inference(const std::vector<ImageProxy>& images, std::vector<DetectionResult>& det_results);

    [[nodiscard]] DetRuntimeConfig get_runtime_config() const noexcept;
    void set_runtime_config(const DetRuntimeConfig& config);
    [[nodiscard]] int get_batch_size() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace myai_gpu
