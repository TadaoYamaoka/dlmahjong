#include "inference.h"

#include <sstream>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger
{
    const char* error_type(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }
    void log(Severity severity, const char* msg) noexcept
    {
        if (severity == Severity::kINTERNAL_ERROR) {
            std::cerr << error_type(severity) << msg << std::endl;
        }
    }
} gLogger;

constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}


Inference::Inference(const char* filepath, const int max_batch_size) : max_batch_size(max_batch_size) {
    checkCudaErrors(cudaMalloc((void**)&x1_dev, sizeof(public_features_t) * max_batch_size));
    checkCudaErrors(cudaMalloc((void**)&x2_dev, sizeof(private_features_t) * max_batch_size));
    checkCudaErrors(cudaMalloc((void**)&y1_dev, N_ACTIONS * max_batch_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&y2_dev, max_batch_size * sizeof(float)));

    inputBindings = { x1_dev, x2_dev, y1_dev, y2_dev };

    load_model(filepath);
}

Inference::~Inference() {
    checkCudaErrors(cudaFree(x1_dev));
    checkCudaErrors(cudaFree(x2_dev));
    checkCudaErrors(cudaFree(y1_dev));
    checkCudaErrors(cudaFree(y2_dev));
}

void Inference::build(const char* filepath) {
    auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        throw std::runtime_error("createInferBuilder");
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        throw std::runtime_error("createNetworkV2");
    }

    auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        throw std::runtime_error("createBuilderConfig");
    }

    auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        throw std::runtime_error("createParser");
    }

    auto parsed = parser->parseFromFile(filepath, (int)nvinfer1::ILogger::Severity::kWARNING);
    if (!parsed)
    {
        throw std::runtime_error("parseFromFile");
    }

    builder->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(64_MiB);

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    assert(network->getNbInputs() == 2);
    nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions(), network->getInput(1)->getDimensions() };
    assert(inputDims[0].nbDims == 4);
    assert(inputDims[1].nbDims == 4);

    assert(network->getNbOutputs() == 2);

    // Optimization Profiles
    auto profile = builder->createOptimizationProfile();
    const auto dims1 = inputDims[0].d;
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
    const auto dims2 = inputDims[1].d;
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
    config->addOptimizationProfile(profile);

    auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine)
    {
        throw std::runtime_error("buildSerializedNetwork");
    }
    auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!engine)
    {
        throw std::runtime_error("deserializeCudaEngine");
    }
}

void Inference::load_model(const char* filepath) {
    build(filepath);

    context = InferUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        throw std::runtime_error("createExecutionContext");
    }

    inputDims1 = engine->getBindingDimensions(0);
    inputDims2 = engine->getBindingDimensions(1);
}

void Inference::forward(const int batch_size, float* x1, float* x2, float* y1, float* y2) {
    inputDims1.d[0] = batch_size;
    inputDims2.d[0] = batch_size;
    context->setBindingDimensions(0, inputDims1);
    context->setBindingDimensions(1, inputDims2);

    checkCudaErrors(cudaMemcpyAsync(x1_dev, x1, sizeof(public_features_t) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    checkCudaErrors(cudaMemcpyAsync(x2_dev, x2, sizeof(private_features_t) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    const bool status = context->enqueue(batch_size, inputBindings.data(), cudaStreamPerThread, nullptr);
    assert(status);
    checkCudaErrors(cudaMemcpyAsync(y1, y1_dev, sizeof(float) * N_ACTIONS * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors(cudaMemcpyAsync(y2, y2_dev, sizeof(float) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
}
