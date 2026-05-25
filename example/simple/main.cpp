#include "model.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

void fill_zero_inputs(myai_gpu::Model& model) {
    for (const auto& input_name : model.input_names()) {
        void* input = model.host_buffer(input_name);
        const std::size_t bytes = model.buffer_size(input_name);
        if (input != nullptr && bytes > 0) {
            std::fill_n(static_cast<std::byte*>(input), bytes, std::byte{0});
        }
    }
}

}  // namespace

int main() {
    try {
        std::cout << myai_gpu::version() << '\n';
        std::cout << myai_gpu::hello() << '\n';

        myai_gpu::ModelConfig config;
        config.batch_size = 1;
        config.use_fp16 = false;

        const auto load_begin = Clock::now();
        myai_gpu::Model model{"model.plan", myai_gpu::TaskType::Unknown, config};
        const auto load_end = Clock::now();

        std::cout << "model ready: " << std::boolalpha << model.is_ready() << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "model load/init time: " << elapsed_ms(load_begin, load_end) << " ms\n";
        std::cout << "batch size: " << model.get_batch_size() << '\n';
        std::cout << "inputs: " << model.input_names().size() << '\n';
        for (const auto& name : model.input_names()) {
            std::cout << "  input: " << name << '\n';
        }
        std::cout << "outputs: " << model.output_names().size() << '\n';
        for (const auto& name : model.output_names()) {
            std::cout << "  output: " << name << '\n';
        }

        constexpr int warmup_runs = 10;
        constexpr int benchmark_runs = 100;

        fill_zero_inputs(model);

        for (int i = 0; i < warmup_runs; ++i) {
            model.inference();
        }

        const auto benchmark_begin = Clock::now();
        for (int i = 0; i < benchmark_runs; ++i) {
            model.inference();
        }
        const auto benchmark_end = Clock::now();

        const double total_ms = elapsed_ms(benchmark_begin, benchmark_end);
        const double latency_ms = total_ms / static_cast<double>(benchmark_runs);
        const double fps = latency_ms > 0.0 ? 1000.0 * static_cast<double>(model.get_batch_size()) / latency_ms : 0.0;

        std::cout << "\nperformance benchmark\n";
        std::cout << "  warmup runs: " << warmup_runs << '\n';
        std::cout << "  benchmark runs: " << benchmark_runs << '\n';
        std::cout << "  total inference time: " << total_ms << " ms\n";
        std::cout << "  average latency: " << latency_ms << " ms\n";
        std::cout << "  throughput: " << fps << " samples/s\n";
        std::cout << "  note: latency includes H2D copy, TensorRT enqueueV3, D2H copy and stream sync.\n";

        for (const auto& name : model.output_names()) {
            const auto* output = static_cast<const float*>(model.host_buffer(name));
            const std::size_t bytes = model.buffer_size(name);
            const std::size_t count = bytes / sizeof(float);
            const std::size_t show_count = std::min<std::size_t>(count, 10);

            std::cout << "output " << name << " first " << show_count << " values:";
            for (std::size_t i = 0; i < show_count; ++i) {
                std::cout << ' ' << output[i];
            }
            std::cout << '\n';
        }

        return model.is_ready() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "simple_demo failed: " << error.what() << '\n';
        return 1;
    }
}
