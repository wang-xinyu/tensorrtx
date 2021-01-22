#include "InferenceEngine.h"

namespace trt {

   InferenceEngine::InferenceEngine(const EngineConfig &enginecfg): _engineCfg(enginecfg) { 

        assert(_engineCfg.max_batch_size > 0);

        CHECK(cudaSetDevice(_engineCfg.device_id));

        _runtime = make_holder(nvinfer1::createInferRuntime(gLogger));
        assert(_runtime);

        _engine = make_holder(_runtime->deserializeCudaEngine(_engineCfg.trtModelStream.get(), _engineCfg.stream_size)); 
        assert(_engine);

        _context = make_holder(_engine->createExecutionContext());
        assert(_context);

        _inputSize = _engineCfg.max_batch_size * 3 * _engineCfg.input_h * _engineCfg.input_w * _depth;
        _outputSize = _engineCfg.max_batch_size * _engineCfg.output_size * _depth; 

        CHECK(cudaMallocHost((void**)&_data, _inputSize));
        CHECK(cudaMallocHost((void**)&_prob, _outputSize));

        _streamptr = std::shared_ptr<cudaStream_t>( new cudaStream_t, 
            [](cudaStream_t* ptr){ 
                cudaStreamDestroy(*ptr);
                if(ptr != nullptr){ 
                    delete ptr;
                } 
            });

        CHECK(cudaStreamCreate(&*_streamptr.get()));

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(_engine->getNbBindings() == 2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        _inputIndex = _engine->getBindingIndex(_engineCfg.input_name);
        _outputIndex = _engine->getBindingIndex(_engineCfg.output_name);
        
        // Create GPU buffers on device
        CHECK(cudaMalloc(&_buffers[_inputIndex], _inputSize));
        CHECK(cudaMalloc(&_buffers[_outputIndex], _outputSize));

        _inputSize /= _engineCfg.max_batch_size;
        _outputSize /= _engineCfg.max_batch_size; 

    }

    bool InferenceEngine::doInference(const int inference_batch_size, std::function<void(float*)> preprocessing) {
        assert(inference_batch_size <= _engineCfg.max_batch_size);
        preprocessing(_data);
        CHECK(cudaSetDevice(_engineCfg.device_id));
        CHECK(cudaMemcpyAsync(_buffers[_inputIndex], _data, inference_batch_size * _inputSize, cudaMemcpyHostToDevice, *_streamptr));
        auto status = _context->enqueue(inference_batch_size, _buffers, *_streamptr, nullptr);
        CHECK(cudaMemcpyAsync(_prob, _buffers[_outputIndex], inference_batch_size * _outputSize, cudaMemcpyDeviceToHost, *_streamptr));
        CHECK(cudaStreamSynchronize(*_streamptr));
        return status;
    }

    InferenceEngine::InferenceEngine(InferenceEngine &&other) noexcept: 
        _engineCfg(other._engineCfg)
        , _data(other._data)
        , _prob(other._prob)
        , _inputIndex(other._inputIndex) 
        , _outputIndex(other._outputIndex)
        , _inputSize(other._inputSize) 
        , _outputSize(other._outputSize)
        , _runtime(std::move(other._runtime))
        , _engine(std::move(other._engine))
        , _context(std::move(other._context))
        , _streamptr(other._streamptr) { 

        _buffers[0] = other._buffers[0];
        _buffers[1] = other._buffers[1];
        other._streamptr.reset();
        other._data = nullptr;
        other._prob = nullptr;
        other._buffers[0] = nullptr; 
        other._buffers[1] = nullptr; 
    } 

    InferenceEngine::~InferenceEngine() {  
        CHECK(cudaFreeHost(_data));
        CHECK(cudaFreeHost(_prob));
        CHECK(cudaFree(_buffers[_inputIndex]));
        CHECK(cudaFree(_buffers[_outputIndex]));
    }
}