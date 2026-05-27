#include "model.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace myai_gpu {

namespace {

std::string lower_extension(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

bool has_dynamic_dim(const nvinfer1::Dims& dims) noexcept {
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open model file: " + path);
    }

    input.unsetf(std::ios::skipws);
    return {std::istream_iterator<char>{input}, std::istream_iterator<char>{}};
}

void write_binary_file(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to write plan file: " + path.string());
    }
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

std::filesystem::path plan_path_for_onnx(const std::filesystem::path& onnx_path) {
    auto plan_path = onnx_path;
    plan_path.replace_extension(".plan");
    return plan_path;
}

void check_cuda(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + " failed: " + cudaGetErrorString(status));
    }
}

}  // namespace

std::string version() {
    return "myai-gpu 0.1.0";
}

std::string hello() {
    return "hello from myai_gpu SDK";
}

TrtLogger::TrtLogger(Severity severity) noexcept : reportable_severity_(severity) {}

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity > reportable_severity_ || msg == nullptr) {
        return;
    }

    std::cerr << "[TensorRT] ";
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING";
            break;
        case Severity::kINFO:
            std::cerr << "INFO";
            break;
        case Severity::kVERBOSE:
            std::cerr << "VERBOSE";
            break;
    }
    std::cerr << ": " << msg << '\n';
}

void TrtLogger::set_reportable_severity(Severity severity) noexcept {
    reportable_severity_ = severity;
}

TrtLogger::Severity TrtLogger::reportable_severity() const noexcept {
    return reportable_severity_;
}

Model::Model(const std::string& model_path,
             const std::string& config_path,
             TaskType task_type)
    : task_type_(task_type) {
    parse_config(config_path);
    init_model(model_path);
}

Model::Model(const std::string& model_path, TaskType task_type, ModelConfig config)
    : task_type_(task_type), config_(config) {
    init_model(model_path);
}

Model::~Model() {
    destroy_cuda_stream();
}

Model::Model(Model&& other) noexcept
    : task_type_(other.task_type_),
      config_(other.config_),
      logger_(other.logger_.reportable_severity()),
      runtime_(std::move(other.runtime_)),
      engine_(std::move(other.engine_)),
      context_(std::move(other.context_)),
    buffers_(std::move(other.buffers_)),
      cuda_stream_(other.cuda_stream_),
      tensors_(std::move(other.tensors_)),
      input_names_(std::move(other.input_names_)),
      output_names_(std::move(other.output_names_)) {
    other.cuda_stream_ = nullptr;
}

Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        destroy_cuda_stream();
        task_type_ = other.task_type_;
        config_ = other.config_;
        logger_.set_reportable_severity(other.logger_.reportable_severity());
        runtime_ = std::move(other.runtime_);
        engine_ = std::move(other.engine_);
        context_ = std::move(other.context_);
        buffers_ = std::move(other.buffers_);
        cuda_stream_ = other.cuda_stream_;
        tensors_ = std::move(other.tensors_);
        input_names_ = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);
        other.cuda_stream_ = nullptr;
    }
    return *this;
}

void Model::inference() {
    if (context_ == nullptr) {
        throw std::runtime_error("TensorRT execution context is not initialized");
    }
    if (buffers_ == nullptr) {
        throw std::runtime_error("BufferManager is not initialized");
    }

    buffers_->copyInputToDeviceAsync(cuda_stream_);
    if (!context_->enqueueV3(cuda_stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    buffers_->copyOutputToHostAsync(cuda_stream_);
    check_cuda(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize");
}

void Model::inference_device_input() {
    if (context_ == nullptr) {
        throw std::runtime_error("TensorRT execution context is not initialized");
    }
    if (buffers_ == nullptr) {
        throw std::runtime_error("BufferManager is not initialized");
    }

    if (!context_->enqueueV3(cuda_stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    buffers_->copyOutputToHostAsync(cuda_stream_);
    check_cuda(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize");
}

void Model::infer_zero_input() {
    if (buffers_ == nullptr) {
        throw std::runtime_error("BufferManager is not initialized");
    }

    for (const auto& input_name : input_names_) {
        void* input = buffers_->getHostBuffer(input_name);
        const std::size_t bytes = buffers_->size(input_name);
        if (input != nullptr && bytes != walvisionai::gpu::BufferManager::kINVALID_SIZE_VALUE) {
            std::fill_n(static_cast<std::byte*>(input), bytes, std::byte{0});
        }
    }

    inference();
}

int Model::get_batch_size() const noexcept {
    return config_.batch_size;
}

TaskType Model::task_type() const noexcept {
    return task_type_;
}

bool Model::is_ready() const noexcept {
    return runtime_ != nullptr && engine_ != nullptr && context_ != nullptr && buffers_ != nullptr && cuda_stream_ != nullptr;
}

cudaStream_t Model::cuda_stream() const noexcept {
    return cuda_stream_;
}

void* Model::host_buffer(const std::string& tensor_name) const noexcept {
    return buffers_ == nullptr ? nullptr : buffers_->getHostBuffer(tensor_name);
}

void* Model::device_buffer(const std::string& tensor_name) const noexcept {
    return buffers_ == nullptr ? nullptr : buffers_->getDeviceBuffer(tensor_name);
}

std::size_t Model::buffer_size(const std::string& tensor_name) const noexcept {
    return buffers_ == nullptr ? 0 : buffers_->size(tensor_name);
}

const std::vector<TensorInfo>& Model::tensors() const noexcept {
    return tensors_;
}

const std::vector<std::string>& Model::input_names() const noexcept {
    return input_names_;
}

const std::vector<std::string>& Model::output_names() const noexcept {
    return output_names_;
}

void Model::parse_config(const std::string& config_path) {
    // Stage 2 keeps config parsing intentionally minimal. Later stages will replace
    // this with JSON parsing for batch, FP16, input size and task settings.
    if (config_path.empty()) {
        return;
    }
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("config file does not exist: " + config_path);
    }
}

void Model::init_model(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::runtime_error("model path is empty");
    }
    // 初始化TensorRT插件库，以确保所有插件都已注册
    initLibNvInferPlugins(&logger_, "");
    // 根据文件扩展名检测模型格式，并加载模型
    switch (detect_model_format(model_path)) {
        case ModelFormat::Plan:
            /*
                1. 读取模型文件
                2. 创建runtime
                3. runtime反序列化engine
            */
            load_model(model_path);
            break;
        case ModelFormat::Onnx:
            /*
                1. 创建builder、network、config、parser
                2. 解析ONNX模型文件到network
                3. 为动态输入设置优化配置文件（如果有）
                4. 构建序列化的engine
                5. 将序列化的engine写入.plan文件以供后续使用
                6. 从序列化的engine创建runtime和engine
            */
            parse_model(model_path);
            break;
    }
    // 创建执行上下文
    create_execution_context();
    // 创建CUDA流
    create_cuda_stream();
    // 应用动态输入形状
    apply_dynamic_input_shapes();
    // 收集tensor的信息（名称、形状、数据类型、IO模式等）
    collect_tensor_info();
    // 创建BufferManager，分配输入输出的host/device缓冲区
    create_buffer_manager();
    // 将缓冲区地址绑定到执行上下文，以便TensorRT在推理时能够正确地读写输入输出数据
    bind_tensor_addresses();
    // 加载推理参数（如果有），例如预处理/后处理所需的参数等
    load_infer_param();
}

void Model::parse_model(const std::string& model_path) {
    /*
        1. 创建TensorRT builder
        2. 作用：
            - 创建network定义
            - 创建builder config并设置工作空间大小和FP16选项
            - optimization profile：如果模型有动态输入维度，则需要为每个动态输入设置最小/最优/最大形状的配置文件
            - 最终构建engine并序列化为plan文件以供后续使用
    */
    auto builder = std::unique_ptr<nvinfer1::IBuilder>{nvinfer1::createInferBuilder(logger_)};
    if (builder == nullptr) {
        throw std::runtime_error("failed to create TensorRT builder");
    }
    /*
        1. 创建TensorRT network
        2. 作用：
            - ONNX解析器会把ONNX模型结构解析到 network中
            - 0U表示使用默认方式创建 network定义（不使用显式batch维度）

    */
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{builder->createNetworkV2(0U)};
    if (network == nullptr) {
        throw std::runtime_error("failed to create TensorRT network");
    }
    /*
        1. 创建TensorRT builder config
        2. 作用：
            - 设置构建engine时的配置选项，例如工作空间大小、是否启用FP16等
    */
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{builder->createBuilderConfig()};
    if (config == nullptr) {
        throw std::runtime_error("failed to create TensorRT builder config");
    }
    /*
        1. 设置workspace的工作空间大小
        2. 设置FP16选项（只有配置允许且当前平台支持快速 FP16 时才启用。）
    */
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config_.workspace_size_bytes);
    if (config_.use_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    /*
        1. 创建onnxparser并将network和logger传入
        2. 作用：
            - 解析ONNX模型文件并将其转换为TensorRT的network定义
    */
    auto parser = std::unique_ptr<nvonnxparser::IParser>{nvonnxparser::createParser(*network, logger_)};
    if (parser == nullptr) {
        throw std::runtime_error("failed to create ONNX parser");
    }
    /*
        1. 解析ONNX模型文件
        2. 作用：
            - 将ONNX模型文件解析到network定义中，以便后续构建TensorRT engine
    */
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("failed to parse ONNX model: " + model_path);
    }
    /*
        1. 为动态输入设置优化配置文件
        2. 作用：
            - 主要处理动态输入维度
    */
    auto* profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
        throw std::runtime_error("failed to create TensorRT optimization profile");
    }

    bool has_dynamic_input = false;
    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        auto* input = network->getInput(i);
        if (input == nullptr) {
            continue;
        }

        const auto dims = input->getDimensions();
        if (!has_dynamic_dim(dims)) {
            continue;
        }

        auto min_dims = dims;
        auto opt_dims = dims;
        auto max_dims = dims;
        /*
            1. 通道维是动态，则设置为3（假设是RGB图像输入）
            2. 高宽维是动态，则设置为config中指定的输入尺寸（例如224x224）
            3. 批次维是动态，则设置为config中指定的批次大小（例如1），并且最大批次大小可以设置为更大的值以支持更大的输入（例如4或8），具体取决于模型和GPU的能力
        */
        for (int32_t dim = 0; dim < dims.nbDims; ++dim) {
            if (dims.d[dim] < 0) {
                if (dims.nbDims == 4 && dim == 1) {
                    min_dims.d[dim] = 3;
                    opt_dims.d[dim] = 3;
                    max_dims.d[dim] = 3;
                } else if (dims.nbDims == 4 && dim > 1) {
                    const int input_size = dim == 2 ? config_.input_height : config_.input_width;
                    min_dims.d[dim] = input_size;
                    opt_dims.d[dim] = input_size;
                    max_dims.d[dim] = input_size;
                } else {
                    min_dims.d[dim] = dim == 0 ? 1 : 1;
                    opt_dims.d[dim] = dim == 0 ? config_.batch_size : 1;
                    max_dims.d[dim] = dim == 0 ? std::max(config_.batch_size, 1) : 1;
                }
            }
        }
        /*
            1. 将设置好的最小/最优/最大形状配置应用到优化配置文件中
            2. 作用：
                - 告诉TensorRT在构建engine时如何处理动态输入维度，以便在推理时能够接受不同形状的输入
        */
        const char* input_name = input->getName();
        if (!profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, min_dims) ||
            !profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, opt_dims) ||
            !profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, max_dims)) {
            throw std::runtime_error("failed to set optimization profile for input: " + std::string{input_name});
        }
        has_dynamic_input = true;
    }
    /*
        1. 如果模型有动态输入维度，则将优化配置文件添加到builder config中
        2. 作用：
            - 确保在构建engine时考虑动态输入的形状范围，以便生成能够处理这些输入的engine
    */
    if (has_dynamic_input) {
        if (!profile->isValid()) {
            throw std::runtime_error("TensorRT optimization profile is invalid");
        }
        config->addOptimizationProfile(profile);
    }
    /*
        1. 构建序列化的engine
        2. 作用：
            - 根据network定义和builder config构建TensorRT engine，并将其序列化为内存中的二进制数据，以便后续创建runtime和engine
    */
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>{builder->buildSerializedNetwork(*network, *config)};
    if (serialized_engine == nullptr) {
        throw std::runtime_error("failed to build TensorRT serialized engine from ONNX: " + model_path);
    }
    /*
        1. 将序列化的engine写入.plan文件以供后续使用
        2. 作用：
            - 避免每次加载模型时都从ONNX构建engine，节省时间。
    */
    write_binary_file(plan_path_for_onnx(model_path), serialized_engine->data(), serialized_engine->size());
    /*
        1. 创建runtime并从序列化的engine创建engine实例
        2. 作用：
            - 从序列化的engine数据创建TensorRT runtime和engine实例，以便后续进行推理。
    */
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(logger_)};
    if (runtime_ == nullptr) {
        throw std::runtime_error("failed to create TensorRT runtime");
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>{
        runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size())};
    if (engine_ == nullptr) {
        throw std::runtime_error("failed to deserialize built TensorRT engine");
    }
}

void Model::load_model(const std::string& model_path) {
    const auto plan = read_binary_file(model_path);
    if (plan.empty()) {
        throw std::runtime_error("plan file is empty: " + model_path);
    }

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(logger_)};
    if (runtime_ == nullptr) {
        throw std::runtime_error("failed to create TensorRT runtime");
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>{runtime_->deserializeCudaEngine(plan.data(), plan.size())};
    if (engine_ == nullptr) {
        throw std::runtime_error("failed to deserialize TensorRT plan: " + model_path);
    }
}

void Model::load_infer_param() {
    // Reserved for stage 3/4: input/output tensor addresses and task-specific
    // inference parameters will be loaded after BufferManager exists.
}

void Model::create_execution_context() {
    if (engine_ == nullptr) {
        throw std::runtime_error("TensorRT engine is not initialized");
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>{engine_->createExecutionContext()};
    if (context_ == nullptr) {
        throw std::runtime_error("failed to create TensorRT execution context");
    }
}

void Model::create_cuda_stream() {
    if (cuda_stream_ != nullptr) {
        return;
    }
    check_cuda(cudaStreamCreate(&cuda_stream_), "cudaStreamCreate");
}

void Model::destroy_cuda_stream() noexcept {
    if (cuda_stream_ != nullptr) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
}

void Model::collect_tensor_info() {
    tensors_.clear();
    input_names_.clear();
    output_names_.clear();

    if (engine_ == nullptr) {
        return;
    }

    const int32_t tensor_count = engine_->getNbIOTensors();
    // 预先分配足够的空间以避免在循环中多次重新分配
    tensors_.reserve(static_cast<std::size_t>(tensor_count));

    for (int32_t i = 0; i < tensor_count; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (tensor_name == nullptr) {
            continue;
        }

        TensorInfo info;
        info.name = tensor_name;
        info.dims = context_ != nullptr ? context_->getTensorShape(tensor_name) : engine_->getTensorShape(tensor_name);
        info.dtype = engine_->getTensorDataType(tensor_name);
        info.io_mode = engine_->getTensorIOMode(tensor_name);

        if (info.io_mode == nvinfer1::TensorIOMode::kINPUT) {
            input_names_.push_back(info.name);
        } else if (info.io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_names_.push_back(info.name);
        }

        tensors_.push_back(std::move(info));
    }
}

void Model::apply_dynamic_input_shapes() {
    if (engine_ == nullptr || context_ == nullptr) {
        return;
    }
    // 获取所有输入输出tensor的数量
    const int32_t tensor_count = engine_->getNbIOTensors();
    // 遍历每个tensor
    for (int32_t i = 0; i < tensor_count; ++i) {
        // 获取tensor名称
        const char* tensor_name = engine_->getIOTensorName(i);
        // 如果tensor名称为空或者不是输入tensor，则跳过
        if (tensor_name == nullptr || engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kINPUT) {
            continue;
        }
        // 获取tensor的形状信息，如果没有动态维度则跳过
        auto dims = engine_->getTensorShape(tensor_name);
        if (!has_dynamic_dim(dims)) {
            continue;
        }
        // 根据模型配置为动态维度设置具体的值，例如batch size
        if (dims.nbDims > 0 && dims.d[0] == -1) {
            dims.d[0] = config_.batch_size;
        }
        if (dims.nbDims == 4) {
            if (dims.d[1] == -1) {
                dims.d[1] = 3;
            }
            if (dims.d[2] == -1) {
                dims.d[2] = config_.input_height;
            }
            if (dims.d[3] == -1) {
                dims.d[3] = config_.input_width;
            }
        }
        // 将设置好的形状应用到执行上下文中，以便TensorRT能够正确地处理动态输入
        if (!context_->setInputShape(tensor_name, dims)) {
            throw std::runtime_error("failed to set dynamic input shape for tensor: " + std::string{tensor_name});
        }
    }
}

void Model::create_buffer_manager() {
    if (engine_ == nullptr || context_ == nullptr) {
        throw std::runtime_error("TensorRT engine/context is not initialized");
    }

    buffers_ = std::make_unique<walvisionai::gpu::BufferManager>(engine_, config_.batch_size, context_.get());
}

void Model::bind_tensor_addresses() {
    if (engine_ == nullptr || context_ == nullptr || buffers_ == nullptr) {
        throw std::runtime_error("TensorRT buffers are not initialized");
    }

    const int32_t tensor_count = engine_->getNbIOTensors();
    for (int32_t i = 0; i < tensor_count; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (tensor_name == nullptr) {
            continue;
        }

        void* device_buffer = buffers_->getDeviceBuffer(tensor_name);
        if (device_buffer == nullptr) {
            throw std::runtime_error("failed to get device buffer for tensor: " + std::string{tensor_name});
        }
        if (!context_->setTensorAddress(tensor_name, device_buffer)) {
            throw std::runtime_error("failed to bind TensorRT tensor address: " + std::string{tensor_name});
        }
    }
}

ModelFormat Model::detect_model_format(const std::string& model_path) {
    const auto ext = lower_extension(model_path);
    if (ext == ".plan" || ext == ".engine") {
        return ModelFormat::Plan;
    }
    if (ext == ".onnx") {
        return ModelFormat::Onnx;
    }
    throw std::runtime_error("unsupported model format: " + model_path + " (expected .plan, .engine, or .onnx)");
}

}  // namespace myai_gpu
