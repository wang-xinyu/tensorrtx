#include "inception_v4.h"

namespace trtx {
InceptionV4::InceptionV4(const InceptionV4Params& params) : mParams(params), mContext(nullptr), mEngine(nullptr) {}

/**
     * Builds the tensorrt engine and serializes it.
    **/
bool InceptionV4::serializeEngine() {
    // load weights
    weightMap = loadWeights(mParams.weightsFile);

    // create builder
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder);

    // create builder config
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config);

    // create engine
    bool created = buildEngine(builder, config);
    if (!created) {
        std::cerr << "Engine creation failed. Check logs." << std::endl;
        return false;
    }

    // serilaize engine
    assert(mEngine != nullptr);
    IHostMemory* modelStream{nullptr};
    modelStream = mEngine->serialize();
    assert(modelStream != nullptr);

    // write serialized engine to file
    std::ofstream trtFile(mParams.trtEngineFile, std::ios::binary);
    if (!trtFile) {
        std::cerr << "Unable to open engine file." << std::endl;
        return false;
    }

    trtFile.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    std::cout << "Engine serialized and saved." << std::endl;

    return true;
}

bool InceptionV4::buildEngine(IBuilder* builder, IBuilderConfig* config) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(mParams.inputTensorName, dt, Dims4{1, 3, mParams.inputH, mParams.inputW});
    assert(data);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    float shval[3] = {(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5};
    float scval[3] = {0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5};
    float pval[3] = {1.0, 1.0, 1.0};
    Weights shift{DataType::kFLOAT, shval, 3};
    Weights scale{DataType::kFLOAT, scval, 3};
    Weights power{DataType::kFLOAT, pval, 3};
    IScaleLayer* scale1 = network->addScale(*data, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale1);

    IActivationLayer* relu0 =
            basicConv2d(network, weightMap, *scale1->getOutput(0), 32, DimsHW{3, 3}, 2, DimsHW{0, 0}, "features.0");
    relu0 = basicConv2d(network, weightMap, *relu0->getOutput(0), 32, DimsHW{3, 3}, 1, DimsHW{0, 0}, "features.1");
    relu0 = basicConv2d(network, weightMap, *relu0->getOutput(0), 64, DimsHW{3, 3}, 1, DimsHW{1, 1}, "features.2");

    auto cat0 = mixed_3a(network, weightMap, *relu0->getOutput(0), "features.3");
    cat0 = mixed_4a(network, weightMap, *cat0->getOutput(0), "features.4");
    cat0 = mixed_5a(network, weightMap, *cat0->getOutput(0), "features.5");
    cat0 = inceptionA(network, weightMap, *cat0->getOutput(0), "features.6");
    cat0 = inceptionA(network, weightMap, *cat0->getOutput(0), "features.7");
    cat0 = inceptionA(network, weightMap, *cat0->getOutput(0), "features.8");
    cat0 = inceptionA(network, weightMap, *cat0->getOutput(0), "features.9");
    cat0 = reductionA(network, weightMap, *cat0->getOutput(0), "features.10");

    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.11");
    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.12");
    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.13");
    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.14");
    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.15");
    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.16");
    cat0 = inceptionB(network, weightMap, *cat0->getOutput(0), "features.17");
    cat0 = reductionB(network, weightMap, *cat0->getOutput(0), "features.18");

    cat0 = inceptionC(network, weightMap, *cat0->getOutput(0), "features.19");
    cat0 = inceptionC(network, weightMap, *cat0->getOutput(0), "features.20");
    cat0 = inceptionC(network, weightMap, *cat0->getOutput(0), "features.21");

    IPoolingLayer* pool2 = network->addPoolingNd(*cat0->getOutput(0), PoolingType::kAVERAGE, DimsHW{8, 8});
    assert(pool2);

    // flatten pool2 layer.
    IShuffleLayer* pool2FlattenLayer = network->addShuffle(*pool2->getOutput(0));
    pool2FlattenLayer->setReshapeDimensions(Dims2{1, 1536});

    ITensor* pool2FlattenLayerOutput = pool2FlattenLayer->getOutput(0);

    // reshape fc weight.
    Dims fcWeightDims = Dims2{1000, 1536};
    Weights fcW = weightMap["last_linear.weight"];
    IConstantLayer* fcWeightLayer = network->addConstant(fcWeightDims, fcW);
    assert(fcWeightLayer);

    // matrix multiply.
    IMatrixMultiplyLayer* fcMatrixMultiplyLayer =
            network->addMatrixMultiply(*pool2FlattenLayerOutput, MatrixOperation::kNONE, *fcWeightLayer->getOutput(0),
                                       MatrixOperation::kTRANSPOSE);
    assert(fcMatrixMultiplyLayer);

    // add fc bias
    Dims fcBiasDims = Dims2{1, 1000};
    Weights fcBias = weightMap["last_linear.bias"];
    IConstantLayer* fcBiasLayer = network->addConstant(fcBiasDims, fcBias);
    assert(fcBiasLayer);

    IElementWiseLayer* fc1 = network->addElementWise(*fcMatrixMultiplyLayer->getOutput(0), *fcBiasLayer->getOutput(0),
                                                     ElementWiseOperation::kSUM);
    assert(fc1);

    fc1->getOutput(0)->setName(mParams.outputTensorName);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    if (mParams.fp16)
        config->setFlag(BuilderFlag::kFP16);
    mEngine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    if (mEngine == nullptr)
        return false;
    return true;
}

bool InceptionV4::deserializeCudaEngine() {
    if (mContext != nullptr && mEngine != nullptr) {
        return true;
    }

    if (mEngine == nullptr) {
        char* trtModelStream{nullptr};
        size_t size{0};

        // open file
        std::ifstream f(mParams.trtEngineFile, std::ios::binary);

        if (f.good()) {
            // get size
            f.seekg(0, f.end);
            size = f.tellg();
            f.seekg(0, f.beg);

            trtModelStream = new char[size];

            // read data as a block
            f.read(trtModelStream, size);
            f.close();
        }

        if (trtModelStream == nullptr) {
            return false;
        }

        // deserialize
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime);

        mEngine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(mEngine != nullptr);

        delete[] trtModelStream;
    }

    std::cout << "deserialized engine successfully." << std::endl;

    // create execution context
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    return true;
}

void InceptionV4::doInference(float* input, float* output, int batchSize) {
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(mEngine->getNbIOTensors() == 2);

    const char* inputName = mParams.inputTensorName;
    const char* outputName = mParams.outputTensorName;

    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};

    // Create GPU buffers on device
    cudaMalloc(&deviceInput, batchSize * 3 * mParams.inputH * mParams.inputW * sizeof(float));
    cudaMalloc(&deviceOutput, batchSize * mParams.outputSize * sizeof(float));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    cudaMemcpyAsync(deviceInput, input, batchSize * 3 * mParams.inputH * mParams.inputW * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    mContext->setTensorAddress(inputName, deviceInput);
    mContext->setTensorAddress(outputName, deviceOutput);

    mContext->enqueueV3(stream);

    cudaMemcpyAsync(output, deviceOutput, batchSize * mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

/**
     * Cleans up any state created in the InceptionV4Trt class
    **/
bool InceptionV4::cleanUp() {
    if (mContext != nullptr)
        ;

    if (mEngine != nullptr)
        ;

    return true;
}
}  // namespace trtx
