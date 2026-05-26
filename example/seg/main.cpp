#include "segmentor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultWarmupRuns = 5;
constexpr int kDefaultMeasureRuns = 50;

int parse_positive_int(const char* text, int fallback) {
    try {
        const int value = std::stoi(text);
        return value > 0 ? value : fallback;
    } catch (const std::exception&) {
        return fallback;
    }
}

cv::Mat colorize_mask(const myai_gpu::SegmentationResult& result) {
    cv::Mat mask(result.height, result.width, CV_8UC1, result.mask.get());
    cv::Mat colorized(result.height, result.width, CV_8UC3, cv::Scalar{0, 0, 0});

    for (int y = 0; y < mask.rows; ++y) {
        const auto* src = mask.ptr<unsigned char>(y);
        auto* dst = colorized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            const unsigned char label = src[x];
            if (label == 0U) {
                dst[x] = cv::Vec3b{0, 0, 0};
            } else if (label == 1U || label == 255U) {
                dst[x] = cv::Vec3b{0, 255, 0};
            } else {
                dst[x] = cv::Vec3b{
                    static_cast<unsigned char>((37U * label) % 255U),
                    static_cast<unsigned char>((17U * label + 80U) % 255U),
                    static_cast<unsigned char>((97U * label + 40U) % 255U)};
            }
        }
    }
    return colorized;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "unet.onnx";
        const std::string image_path = argc > 2 ? argv[2] : "imgs/car.jpg";
        const std::string config_path = argc > 3 ? argv[3] : "config/seg_config.json";
        const std::string output_path = argc > 4 ? argv[4] : "imgs/car_res.jpg";
        const int warmup_runs = argc > 5 ? parse_positive_int(argv[5], kDefaultWarmupRuns) : kDefaultWarmupRuns;
        const int measure_runs = argc > 6 ? parse_positive_int(argv[6], kDefaultMeasureRuns) : kDefaultMeasureRuns;

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read image: " << image_path << '\n';
            return 1;
        }

        myai_gpu::Segmentor segmentor{model_path, config_path};
        if (!segmentor.is_ready()) {
            std::cerr << "segmentor is not ready\n";
            return 1;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            [[maybe_unused]] const auto warmup_result = segmentor.inference(image);
        }

        std::vector<double> inference_times_ms;
        inference_times_ms.reserve(static_cast<std::size_t>(measure_runs));

        myai_gpu::SegmentationResult result;
        for (int i = 0; i < measure_runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            result = segmentor.inference(image);
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            inference_times_ms.push_back(elapsed);
        }

        if (result.empty()) {
            std::cerr << "segmentor returned an empty mask\n";
            return 1;
        }

        const double total_time_ms = std::accumulate(inference_times_ms.begin(), inference_times_ms.end(), 0.0);
        const double avg_time_ms = total_time_ms / static_cast<double>(inference_times_ms.size());
        const auto [min_iter, max_iter] = std::minmax_element(inference_times_ms.begin(), inference_times_ms.end());
        const double fps = 1000.0 / avg_time_ms;

        cv::Mat color_mask = colorize_mask(result);
        cv::Mat blended;
        cv::addWeighted(image, 0.65, color_mask, 0.35, 0.0, blended);
        if (!cv::imwrite(output_path, blended)) {
            std::cerr << "failed to write segmentation result image: " << output_path << '\n';
            return 1;
        }

        cv::Mat mask(result.height, result.width, CV_8UC1, result.mask.get());
        const int foreground_pixels = cv::countNonZero(mask);

        std::cout << "myai_gpu segmentor SDK sample\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "model: " << model_path << '\n';
        std::cout << "image: " << image_path << '\n';
        std::cout << "config: " << config_path << '\n';
        std::cout << "output: " << output_path << '\n';
        std::cout << "image size: " << image.cols << "x" << image.rows << '\n';
        std::cout << "mask size: " << result.width << "x" << result.height << '\n';
        std::cout << "batch size: " << segmentor.get_batch_size() << '\n';
        std::cout << "warmup runs: " << warmup_runs << '\n';
        std::cout << "measure runs: " << measure_runs << '\n';
        std::cout << "avg inference time: " << avg_time_ms << " ms\n";
        std::cout << "min inference time: " << *min_iter << " ms\n";
        std::cout << "max inference time: " << *max_iter << " ms\n";
        std::cout << "throughput: " << fps << " FPS\n";
        std::cout << "foreground pixels: " << foreground_pixels << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "seg_sdk_sample failed: " << error.what() << '\n';
        return 1;
    }
}
