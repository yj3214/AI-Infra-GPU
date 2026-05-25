#include "classifier.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "model.plan";
        const std::string image_path = argc > 2 ? argv[2] : "imgs/cat.jpg";
        const std::string config_path = argc > 3 ? argv[3] : "config/cls_config.json";

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read image: " << image_path << '\n';
            return 1;
        }

        myai_gpu::Classifier classifier{model_path, config_path};
        if (!classifier.is_ready()) {
            std::cerr << "classifier is not ready\n";
            return 1;
        }

        const auto result = classifier.inference(image);
        const auto performance = classifier.benchmark(image, 10, 100);

        std::cout << "myai_gpu classifier SDK sample\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "model: " << model_path << '\n';
        std::cout << "image: " << image_path << '\n';
        std::cout << "config: " << config_path << '\n';
        std::cout << "image size: " << image.cols << "x" << image.rows << '\n';
        std::cout << "batch size: " << classifier.get_batch_size() << '\n';
        std::cout << "top1 labelid: " << result.class_id() << '\n';
        std::cout << "top1 confidence: " << result.score() << '\n';
        std::cout << "topk:";
        for (std::size_t i = 0; i < result.label_id.size(); ++i) {
            std::cout << " [" << result.label_id[i] << ": " << result.confidence[i] << ']';
        }
        std::cout << '\n';
        std::cout << "warmup runs: " << performance.warmup_runs << '\n';
        std::cout << "benchmark runs: " << performance.benchmark_runs << '\n';
        std::cout << "average latency after warmup: " << performance.average_latency_ms << " ms\n";
        std::cout << "FPS after warmup: " << performance.fps << " images/s\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cls_sdk_sample failed: " << error.what() << '\n';
        return 1;
    }
}
