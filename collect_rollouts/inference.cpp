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
    checkCudaErrors(cudaMalloc((void**)&x_dev, sizeof(public_features_t) * max_batch_size));
    checkCudaErrors(cudaMalloc((void**)&y_dev, N_ACTIONS * max_batch_size * sizeof(float)));

    inputBindings = { x_dev, y_dev };

    load_model(filepath);
}

Inference::~Inference() {
    checkCudaErrors(cudaFree(x_dev));
    checkCudaErrors(cudaFree(y_dev));
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

    assert(network->getNbInputs() == 1);
    nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions() };
    assert(inputDims[0].nbDims == 4);

    assert(network->getNbOutputs() == 1);

    // Optimization Profiles
    auto profile = builder->createOptimizationProfile();
    const auto dims1 = inputDims[0].d;
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
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

    inputDims = engine->getBindingDimensions(0);
}

void Inference::forward(const int batch_size, float* x, float* y) {
    inputDims.d[0] = batch_size;
    context->setBindingDimensions(0, inputDims);

    checkCudaErrors(cudaMemcpyAsync(x_dev, x, sizeof(public_features_t) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    const bool status = context->enqueue(batch_size, inputBindings.data(), cudaStreamPerThread, nullptr);
    assert(status);
    checkCudaErrors(cudaMemcpyAsync(y, y_dev, sizeof(float) * N_ACTIONS * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
}
