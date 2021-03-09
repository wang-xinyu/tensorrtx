/**************************************************************************
 * Handle memory pre-alloc
 * both on host(pinned memory, allow CUDA DMA) & device
*************************************************************************/

#pragma once

#include <thread>
#include <chrono>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "holder.h"
#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
static Logger gLogger;

namespace trt {

    struct EngineConfig {
        const char* input_name;
        const char* output_name; 
        std::shared_ptr<char> trtModelStream;
        int max_batch_size; /* create engine */
        int input_h;  
        int input_w;
        int output_size;
        int stream_size;
        int device_id;
    };

    class InferenceEngine {

    public:
        InferenceEngine(const EngineConfig &enginecfg);
        InferenceEngine(InferenceEngine &&other) noexcept;
        ~InferenceEngine();

        InferenceEngine(const InferenceEngine &) = delete;
        InferenceEngine& operator=(const InferenceEngine &) = delete;
        InferenceEngine& operator=(InferenceEngine && other) = delete;

        bool doInference(const int inference_batch_size, std::function<void(float*)> preprocessing);
        float* getOutput() { return _prob; }
        std::thread::id getThreadID() { return std::this_thread::get_id(); }

    private:
        EngineConfig _engineCfg;
        float* _data{nullptr};
        float* _prob{nullptr};

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        void* _buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        int _inputIndex;
        int _outputIndex;

        int _inputSize;
        int _outputSize;

        static constexpr std::size_t _depth{sizeof(float)};

        TensorRTHolder<nvinfer1::IRuntime> _runtime{nullptr};
        TensorRTHolder<nvinfer1::ICudaEngine> _engine{nullptr};
        TensorRTHolder<nvinfer1::IExecutionContext> _context{nullptr};
        std::shared_ptr<cudaStream_t> _streamptr;
    };

}

