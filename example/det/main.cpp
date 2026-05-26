#include "detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
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

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "yolov8s.onnx";
        const std::string image_path = argc > 2 ? argv[2] : "imgs/bus.jpg";
        const std::string config_path = argc > 3 ? argv[3] : "config/det_config.json";
        const std::string output_path = argc > 4 ? argv[4] : "imgs/bus_res.jpg";
        const int warmup_runs = argc > 5 ? parse_positive_int(argv[5], kDefaultWarmupRuns) : kDefaultWarmupRuns;
        const int measure_runs = argc > 6 ? parse_positive_int(argv[6], kDefaultMeasureRuns) : kDefaultMeasureRuns;

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read image: " << image_path << '\n';
            return 1;
        }

        myai_gpu::Detector detector{model_path, config_path};
        if (!detector.is_ready()) {
            std::cerr << "detector is not ready\n";
            return 1;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            [[maybe_unused]] const auto warmup_result = detector.inference(image);
        }

        std::vector<double> inference_times_ms;
        inference_times_ms.reserve(static_cast<std::size_t>(measure_runs));

        myai_gpu::DetectionResult result;
        for (int i = 0; i < measure_runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            result = detector.inference(image);
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            inference_times_ms.push_back(elapsed);
        }

        const double total_time_ms = std::accumulate(inference_times_ms.begin(), inference_times_ms.end(), 0.0);
        const double avg_time_ms = total_time_ms / static_cast<double>(inference_times_ms.size());
        const auto [min_iter, max_iter] = std::minmax_element(inference_times_ms.begin(), inference_times_ms.end());
        const double fps = 1000.0 / avg_time_ms;

        cv::Mat visualized = image.clone();

        for (std::size_t i = 0; i < result.size(); ++i) {
            const cv::Scalar color{0.0, 255.0, 0.0};
            cv::rectangle(visualized, result.top_lefts[i], result.bottom_rights[i], color, 2);

            std::ostringstream label;
            label << result.label_id[i] << ':' << std::fixed << std::setprecision(2) << result.confidence[i];
            const std::string label_text = label.str();

            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            const int text_x = std::max(result.top_lefts[i].x, 0);
            const int text_y = std::max(result.top_lefts[i].y - 4, text_size.height + 4);
            cv::rectangle(visualized,
                          cv::Point{text_x, text_y - text_size.height - 4},
                          cv::Point{text_x + text_size.width + 4, text_y + baseline},
                          color,
                          cv::FILLED);
            cv::putText(visualized,
                        label_text,
                        cv::Point{text_x + 2, text_y - 2},
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar{0.0, 0.0, 0.0},
                        1);
        }

        if (!cv::imwrite(output_path, visualized)) {
            std::cerr << "failed to write detection result image: " << output_path << '\n';
            return 1;
        }

        std::cout << "myai_gpu detector SDK sample\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "model: " << model_path << '\n';
        std::cout << "image: " << image_path << '\n';
        std::cout << "config: " << config_path << '\n';
        std::cout << "output: " << output_path << '\n';
        std::cout << "image size: " << image.cols << "x" << image.rows << '\n';
        std::cout << "batch size: " << detector.get_batch_size() << '\n';
        std::cout << "warmup runs: " << warmup_runs << '\n';
        std::cout << "measure runs: " << measure_runs << '\n';
        std::cout << "avg inference time: " << avg_time_ms << " ms\n";
        std::cout << "min inference time: " << *min_iter << " ms\n";
        std::cout << "max inference time: " << *max_iter << " ms\n";
        std::cout << "throughput: " << fps << " FPS\n";
        std::cout << "detections: " << result.size() << '\n';
        for (std::size_t i = 0; i < result.size(); ++i) {
            std::cout << "  #" << i
                      << " label=" << result.label_id[i]
                      << " confidence=" << result.confidence[i]
                      << " box=(" << result.top_lefts[i].x << ',' << result.top_lefts[i].y
                      << ")-(" << result.bottom_rights[i].x << ',' << result.bottom_rights[i].y << ")\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "det_sdk_sample failed: " << error.what() << '\n';
        return 1;
    }
}
