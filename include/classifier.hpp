#pragma once

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace myai_gpu {

enum class InputColorFormat {
    Bgr,
    Rgb,
    Gray,
};

struct ImageProxy {
    int rows{};
    int cols{};
    int channels{};
    std::uint8_t* data{};
};

struct ClassificationResult {
    std::vector<float> confidence;
    std::vector<int> label_id;

    [[nodiscard]] std::size_t class_id() const noexcept;
    [[nodiscard]] float score() const noexcept;
};

struct ClassifierConfig {
    InputColorFormat input_color_format{InputColorFormat::Bgr};
    std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> std{0.229F, 0.224F, 0.225F};
    int batch_size{1};
    int height{224};
    int width{224};
    int topk{1};
    bool use_fp16{false};
};

struct ClassifierPerformance {
    int warmup_runs{};
    int benchmark_runs{};
    double total_ms{};
    double average_latency_ms{};
    double fps{};
};

class Classifier {
public:
    explicit Classifier(const std::string& model_path, const std::string& config_path = {});
    Classifier(const std::string& model_path, ClassifierConfig config);
    ~Classifier();

    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;
    Classifier(Classifier&&) noexcept;
    Classifier& operator=(Classifier&&) noexcept;

    [[nodiscard]] ClassificationResult inference(const cv::Mat& image);
    [[nodiscard]] std::vector<ClassificationResult> inference(const std::vector<cv::Mat>& images);
    void inference(const std::vector<cv::Mat>& images, std::vector<ClassificationResult>& results);
    void inference(const std::vector<ImageProxy>& images, std::vector<ClassificationResult>& results);
    [[nodiscard]] ClassifierPerformance benchmark(const cv::Mat& image, int warmup_runs = 10, int benchmark_runs = 100);
    [[nodiscard]] int get_batch_size() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace myai_gpu
