#pragma once

#include "classifier.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace myai_gpu {

struct SegmentationResult {
    int height{};
    int width{};
    std::unique_ptr<unsigned char[]> mask{};

    SegmentationResult() = default;
    SegmentationResult(int mask_height, int mask_width, std::unique_ptr<unsigned char[]> mask_data) noexcept;
    SegmentationResult(const SegmentationResult& other);
    SegmentationResult& operator=(const SegmentationResult& other);
    SegmentationResult(SegmentationResult&&) noexcept = default;
    SegmentationResult& operator=(SegmentationResult&&) noexcept = default;
    ~SegmentationResult() = default;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
};

struct SegmentorConfig {
    InputColorFormat input_color_format{InputColorFormat::Bgr};
    bool keep_ratio{false};
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> std{1.0F, 1.0F, 1.0F};
    int batch_size{1};
    int height{512};
    int width{512};
    bool use_fp16{false};
    float mask_threshold{0.5F};
};

class Segmentor {
public:
    explicit Segmentor(const std::string& model_path, const std::string& config_path = {});
    Segmentor(const std::string& model_path, SegmentorConfig config);
    ~Segmentor();

    Segmentor(const Segmentor&) = delete;
    Segmentor& operator=(const Segmentor&) = delete;
    Segmentor(Segmentor&&) noexcept;
    Segmentor& operator=(Segmentor&&) noexcept;

    [[nodiscard]] SegmentationResult inference(const cv::Mat& image);
    [[nodiscard]] std::vector<SegmentationResult> inference(const std::vector<cv::Mat>& images);
    void inference(const std::vector<cv::Mat>& images, std::vector<SegmentationResult>& seg_results);
    void inference(const std::vector<ImageProxy>& images, std::vector<SegmentationResult>& seg_results);

    [[nodiscard]] int get_batch_size() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace myai_gpu
