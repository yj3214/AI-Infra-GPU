#pragma once

#include "common/buffers.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace myai_gpu {

std::string version();
std::string hello();

class TrtLogger final : public nvinfer1::ILogger {
public:
	explicit TrtLogger(Severity severity = Severity::kWARNING) noexcept;

	void log(Severity severity, const char* msg) noexcept override;
	void set_reportable_severity(Severity severity) noexcept;
	[[nodiscard]] Severity reportable_severity() const noexcept;

private:
	Severity reportable_severity_;
};

enum class TaskType {
	Classification,
	Detection,
	Segmentation,
	Keypoints,
	Unknown,
};

enum class ModelFormat {
	Onnx,
	Plan,
};

struct ModelConfig {
	int batch_size{1};
	bool use_fp16{false};
	std::size_t workspace_size_bytes{1ULL << 30};
};

struct TensorInfo {
	std::string name;
	nvinfer1::Dims dims{};
	nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
	nvinfer1::TensorIOMode io_mode{nvinfer1::TensorIOMode::kNONE};
};

class Model {
public:
	Model(const std::string& model_path,
		  const std::string& config_path,
		  TaskType task_type);
	explicit Model(const std::string& model_path,
				   TaskType task_type = TaskType::Unknown,
				   ModelConfig config = {});
	~Model();

	Model(const Model&) = delete;
	Model& operator=(const Model&) = delete;
	Model(Model&&) noexcept;
	Model& operator=(Model&&) noexcept;

	void inference();
	void inference_device_input();
	void infer_zero_input();
	[[nodiscard]] int get_batch_size() const noexcept;
	[[nodiscard]] TaskType task_type() const noexcept;
	[[nodiscard]] bool is_ready() const noexcept;
	[[nodiscard]] cudaStream_t cuda_stream() const noexcept;
	[[nodiscard]] void* host_buffer(const std::string& tensor_name) const noexcept;
	[[nodiscard]] void* device_buffer(const std::string& tensor_name) const noexcept;
	[[nodiscard]] std::size_t buffer_size(const std::string& tensor_name) const noexcept;
	[[nodiscard]] const std::vector<TensorInfo>& tensors() const noexcept;
	[[nodiscard]] const std::vector<std::string>& input_names() const noexcept;
	[[nodiscard]] const std::vector<std::string>& output_names() const noexcept;

private:
	void parse_config(const std::string& config_path);
	void init_model(const std::string& model_path);
	void parse_model(const std::string& model_path);
	void load_model(const std::string& model_path);
	void load_infer_param();
	void create_execution_context();
	void create_cuda_stream();
	void destroy_cuda_stream() noexcept;
	void collect_tensor_info();
	void apply_dynamic_input_shapes();
	void create_buffer_manager();
	void bind_tensor_addresses();

	static ModelFormat detect_model_format(const std::string& model_path);

private:
	TaskType task_type_{TaskType::Unknown};
	ModelConfig config_{};
	TrtLogger logger_{};

	std::unique_ptr<nvinfer1::IRuntime> runtime_;
	std::shared_ptr<nvinfer1::ICudaEngine> engine_;
	std::unique_ptr<nvinfer1::IExecutionContext> context_;
	std::unique_ptr<walvisionai::gpu::BufferManager> buffers_;
	cudaStream_t cuda_stream_{nullptr};

	std::vector<TensorInfo> tensors_;
	std::vector<std::string> input_names_;
	std::vector<std::string> output_names_;
};

}  // namespace myai_gpu
