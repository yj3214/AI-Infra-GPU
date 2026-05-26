#pragma once

#include "model.hpp"
#include "segmentor.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace myai_gpu {

class Segmentor::Impl {
public:
    explicit Impl(const std::string& model_path, const std::string& config_path = {});
    Impl(const std::string& model_path, SegmentorConfig config);

    [[nodiscard]] SegmentationResult inference(const cv::Mat& image);
    [[nodiscard]] std::vector<SegmentationResult> inference(const std::vector<cv::Mat>& images);
    void inference(const std::vector<cv::Mat>& images, std::vector<SegmentationResult>& seg_results);
    void inference(const std::vector<ImageProxy>& images, std::vector<SegmentationResult>& seg_results);

    [[nodiscard]] int get_batch_size() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;

private:
    struct PreprocessMeta {
        int original_width{};
        int original_height{};
        int resized_width{};
        int resized_height{};
        int pad_left{};
        int pad_top{};
    };

    void preprocess_to_device(const cv::Mat& image, PreprocessMeta& meta);
    void validate_request(std::size_t image_count) const;
    [[nodiscard]] SegmentationResult decode_output(const PreprocessMeta& meta) const;

private:
    SegmentorConfig config_{};
    Model model_;
};

}  // namespace myai_gpu
