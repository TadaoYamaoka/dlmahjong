#pragma once

#include "../../cmajiang/src_cpp/feature.h"

#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <iostream>

typedef channel_t public_features_t[N_CHANNELS_PUBLIC + 4];
typedef float policy_t[N_ACTIONS];

inline void FatalError(const std::string& s) {
    std::cerr << s << "\nAborting...\n";
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

inline void checkCudaErrors(cudaError_t status) {
    if (status != 0) {
        std::stringstream _error;
        _error << "Cuda failure\nError: " << cudaGetErrorString(status);
        FatalError(_error.str());
    }
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete obj;
        }
    }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class Inference {
public:
    Inference(const char* filepath, const int max_batch_size);
    ~Inference();

    void forward(const int batch_size, float* x, float* y);

protected:
    const int max_batch_size;
    InferUniquePtr<nvinfer1::ICudaEngine> engine;
    float* x_dev;
    float* y_dev;
    std::vector<void*> inputBindings;
    InferUniquePtr<nvinfer1::IExecutionContext> context;
    nvinfer1::Dims inputDims;

    void build(const char* filepath);
    void load_model(const char* filepath);
};
