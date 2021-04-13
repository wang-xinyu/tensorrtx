#ifndef TRTX_INFER_ENGINE_H
#define TRTX_INFER_ENGINE_H

#include <chrono>

#include "utils.h"
#include "holder.h"
#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

static Logger gLogger;

namespace trtx {
    struct InferenceEngineConfig {
        const char* inputBlobName;
        const char* outputBlobName; 
        int maxBatchSize;
        int inputHeight;  
        int inputWidth;
        int outputSize;
    }

    class InceptionInferEngine {

    public:
        InceptionInferEngine(const EngineConfig &cfg);
        ~InceptionInferEngine();

    private:
        InceptionInferEngine engineConfig;
        float* data{nullptr};
        float* prob{nullptr};

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Also, indices are guaranteed to be less than IEngine::getNbBindings().
        int inputIndex;
        int outputIndex;

        int inputSize;
    };
}

#endif  // TRTX_INFER_ENGINE_H