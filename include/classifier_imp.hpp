#pragma once

#include "classifier.hpp"
#include "model.hpp"

#include <opencv2/core/mat.hpp>

#include <string>
#include <vector>

namespace myai_gpu {

class Classifier::Impl {
public:
    explicit Impl(const std::string& model_path, const std::string& config_path = {});
    Impl(const std::string& model_path, ClassifierConfig config);

    [[nodiscard]] ClassificationResult inference(const cv::Mat& image);
    [[nodiscard]] std::vector<ClassificationResult> inference(const std::vector<cv::Mat>& images);
    void inference(const std::vector<cv::Mat>& images, std::vector<ClassificationResult>& results);
    void inference(const std::vector<ImageProxy>& images, std::vector<ClassificationResult>& results);
    [[nodiscard]] ClassifierPerformance benchmark(const cv::Mat& image, int warmup_runs, int benchmark_runs);
    [[nodiscard]] int get_batch_size() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;

private:
    void preprocess_to_device(const cv::Mat& image_bgr);
    void validate_request(std::size_t image_count) const;
    [[nodiscard]] ClassificationResult classify_output() const;

private:
    ClassifierConfig config_{};
    Model model_;
};

}  // namespace myai_gpu
