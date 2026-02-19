#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "utils.h"

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// stuff we know about the network and the input/output blobs
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

class MyStreamReaderV2 : public nvinfer1::IStreamReaderV2 {
   public:
    MyStreamReaderV2(const std::string& filepath) : mFile(filepath, std::ios::binary) {
        if (!mFile) {
            std::cerr << "Error opening engine file: " << filepath << std::endl;
        }
    }

    ~MyStreamReaderV2() override { close(); }

    bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override {
        switch (where) {
            case nvinfer1::SeekPosition::kSET:
                mFile.seekg(offset, std::ios::beg);
                break;
            case nvinfer1::SeekPosition::kCUR:
                mFile.seekg(offset, std::ios_base::cur);
                break;
            case nvinfer1::SeekPosition::kEND:
                mFile.seekg(offset, std::ios::end);
                break;
        }
        return mFile.good();
    }

    int64_t read(void* destination, int64_t nbBytes, cudaStream_t stream) noexcept override {
        if (!mFile.good()) {
            return -1;
        }

        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, destination);
        if (err != cudaSuccess || attributes.type == cudaMemoryTypeHost ||
            attributes.type == cudaMemoryTypeUnregistered) {
            mFile.read(static_cast<char*>(destination), nbBytes);
            return mFile.gcount();
        } else if (attributes.type == cudaMemoryTypeDevice) {
            std::unique_ptr<char[]> tmpBuf(new char[nbBytes]);
            mFile.read(tmpBuf.get(), nbBytes);
            int64_t bytesRead = mFile.gcount();
            cudaMemcpyAsync(destination, tmpBuf.get(), bytesRead, cudaMemcpyHostToDevice, stream);
            return bytesRead;
        }
        return -1;
    }

    void close() {
        if (mFile.is_open()) {
            mFile.close();
        }
    }

    void reset() {
        mFile.clear();
        mFile.seekg(0);
    }

    bool isOpen() const { return mFile.is_open(); }

   private:
    std::ifstream mFile;
};

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                          std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* hSwish(INetworkDefinition* network, ITensor& input, std::string name) {
    auto hsig = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    ILayer* hsw = network->addElementWise(input, *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(hsw);
    return hsw;
}

ILayer* convBnHswish(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch,
                     int ksize, int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (ksize - 1) / 2;
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);

    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-3);
    ILayer* hsw = hSwish(network, *bn1->getOutput(0), lname + "2");
    assert(hsw);
    return hsw;
}

ILayer* seLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c, int w,
                std::string lname) {
    IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW(w, w));
    pool->setStrideNd(DimsHW{w, w});

    int seChannels = weightMap[lname + "fc1.bias"].count;

    IConvolutionLayer* fc1 = network->addConvolutionNd(*pool->getOutput(0), seChannels, DimsHW{1, 1},
                                                       weightMap[lname + "fc1.weight"], weightMap[lname + "fc1.bias"]);

    auto relu1 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* fc2 = network->addConvolutionNd(*relu1->getOutput(0), c, DimsHW{1, 1},
                                                       weightMap[lname + "fc2.weight"], weightMap[lname + "fc2.bias"]);

    auto hsig = network->addActivation(*fc2->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);

    ILayer* se = network->addElementWise(input, *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(se);
    return se;
}

ILayer* convSeq1(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int output,
                 int hdim, int k, int s, bool use_se, bool use_hs, int w, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (k - 1) / 2;
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, hdim, DimsHW{k, k}, weightMap[lname + "0.0.weight"], emptywts);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(hdim);
    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "0.1", 1e-3);
    ITensor *tensor3, *tensor4;
    tensor3 = nullptr;
    tensor4 = nullptr;
    if (use_hs) {
        ILayer* hsw = hSwish(network, *bn1->getOutput(0), lname + "2");
        tensor3 = hsw->getOutput(0);
    } else {
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        tensor3 = relu1->getOutput(0);
    }
    if (use_se) {
        ILayer* se1 = seLayer(network, weightMap, *tensor3, hdim, w, lname + "1.");
        tensor4 = se1->getOutput(0);
    } else {
        tensor4 = tensor3;
    }

    IConvolutionLayer* conv2 =
            network->addConvolutionNd(*tensor4, output, DimsHW{1, 1}, weightMap[lname + "2.0.weight"], emptywts);
    IScaleLayer* bn2 = addBatchNorm(network, weightMap, *conv2->getOutput(0), lname + "2.1", 1e-3);
    assert(bn2);

    return bn2;
}

// For blocks with no expansion AND no SE (like MobileNetV3-Large features.1)
ILayer* convSeq0(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int output,
                 int hdim, int k, int s, bool use_hs, int w, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (k - 1) / 2;

    // Depthwise conv + BN + activation
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, hdim, DimsHW{k, k}, weightMap[lname + "0.0.weight"], emptywts);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(hdim);
    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "0.1", 1e-3);

    ITensor* tensor3;
    if (use_hs) {
        ILayer* hsw = hSwish(network, *bn1->getOutput(0), lname + "depthwise_hswish");
        tensor3 = hsw->getOutput(0);
    } else {
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        tensor3 = relu1->getOutput(0);
    }

    IConvolutionLayer* conv2 =
            network->addConvolutionNd(*tensor3, output, DimsHW{1, 1}, weightMap[lname + "1.0.weight"], emptywts);
    IScaleLayer* bn2 = addBatchNorm(network, weightMap, *conv2->getOutput(0), lname + "1.1", 1e-3);
    assert(bn2);
    return bn2;
}

ILayer* convSeq2(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int output,
                 int hdim, int k, int s, bool use_se, bool use_hs, int w, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (k - 1) / 2;
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, hdim, DimsHW{1, 1}, weightMap[lname + "0.0.weight"], emptywts);
    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "0.1", 1e-3);
    ITensor* tensor3;
    if (use_hs) {
        ILayer* hsw1 = hSwish(network, *bn1->getOutput(0), lname + "expansion_hswish");
        tensor3 = hsw1->getOutput(0);
    } else {
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        tensor3 = relu1->getOutput(0);
    }
    IConvolutionLayer* conv2 =
            network->addConvolutionNd(*tensor3, hdim, DimsHW{k, k}, weightMap[lname + "1.0.weight"], emptywts);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{p, p});
    conv2->setNbGroups(hdim);
    IScaleLayer* bn2 = addBatchNorm(network, weightMap, *conv2->getOutput(0), lname + "1.1", 1e-3);

    ITensor* tensor6;
    if (use_hs) {
        ILayer* hsw2 = hSwish(network, *bn2->getOutput(0), lname + "depthwise_hswish");
        tensor6 = hsw2->getOutput(0);
    } else {
        IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        tensor6 = relu2->getOutput(0);
    }

    ITensor* tensor7;
    if (use_se) {
        ILayer* se1 = seLayer(network, weightMap, *tensor6, hdim, w, lname + "2.");
        tensor7 = se1->getOutput(0);
    } else {
        tensor7 = tensor6;
    }
    const char* projIdx = use_se ? "3." : "2.";
    IConvolutionLayer* conv3 = network->addConvolutionNd(*tensor7, output, DimsHW{1, 1},
                                                         weightMap[lname + projIdx + "0.weight"], emptywts);
    IScaleLayer* bn3 = addBatchNorm(network, weightMap, *conv3->getOutput(0), lname + projIdx + "1", 1e-3f);
    assert(bn3);
    return bn3;
}

ILayer* invertedRes(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                    std::string lname, int inch, int outch, int s, int hidden, int k, bool use_se, bool use_hs, int w) {
    bool use_res_connect = (s == 1 && inch == outch);
    ILayer* conv = nullptr;
    if (inch == hidden) {
        if (use_se) {
            conv = convSeq1(network, weightMap, input, outch, hidden, k, s, use_se, use_hs, w, lname + "block.");
        } else {
            conv = convSeq0(network, weightMap, input, outch, hidden, k, s, use_hs, w, lname + "block.");
        }
    } else {
        conv = convSeq2(network, weightMap, input, outch, hidden, k, s, use_se, use_hs, w, lname + "block.");
    }

    ILayer* finalLayer;
    if (!use_res_connect) {
        finalLayer = conv;
    } else {
        IElementWiseLayer* ew3 = network->addElementWise(input, *conv->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew3);
        finalLayer = ew3;
    }

    return finalLayer;
}

// Creat the engine using only the API and not any parser.
IHostMemory* createEngineSmall(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{static_cast<int>(maxBatchSize), 3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../mbv3_small.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto ew1 = convBnHswish(network, weightMap, *data, 16, 3, 2, 1, "features.0.");
    auto ir1 = invertedRes(network, weightMap, *ew1->getOutput(0), "features.1.", 16, 16, 2, 16, 3, 1, 0, 56);
    auto ir2 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.2.", 16, 24, 2, 72, 3, 0, 0, 28);
    auto ir3 = invertedRes(network, weightMap, *ir2->getOutput(0), "features.3.", 24, 24, 1, 88, 3, 0, 0, 28);
    auto ir4 = invertedRes(network, weightMap, *ir3->getOutput(0), "features.4.", 24, 40, 2, 96, 5, 1, 1, 14);
    auto ir5 = invertedRes(network, weightMap, *ir4->getOutput(0), "features.5.", 40, 40, 1, 240, 5, 1, 1, 14);
    auto ir6 = invertedRes(network, weightMap, *ir5->getOutput(0), "features.6.", 40, 40, 1, 240, 5, 1, 1, 14);
    auto ir7 = invertedRes(network, weightMap, *ir6->getOutput(0), "features.7.", 40, 48, 1, 120, 5, 1, 1, 14);
    auto ir8 = invertedRes(network, weightMap, *ir7->getOutput(0), "features.8.", 48, 48, 1, 144, 5, 1, 1, 14);
    auto ir9 = invertedRes(network, weightMap, *ir8->getOutput(0), "features.9.", 48, 96, 2, 288, 5, 1, 1, 7);
    auto ir10 = invertedRes(network, weightMap, *ir9->getOutput(0), "features.10.", 96, 96, 1, 576, 5, 1, 1, 7);
    auto ir11 = invertedRes(network, weightMap, *ir10->getOutput(0), "features.11.", 96, 96, 1, 576, 5, 1, 1, 7);
    ILayer* ew2 = convBnHswish(network, weightMap, *ir11->getOutput(0), 576, 1, 1, 1, "features.12.");

    IPoolingLayer* pool1 = network->addPoolingNd(*ew2->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool1);
    pool1->setStrideNd(DimsHW{7, 7});

    auto reshape1 = network->addShuffle(*pool1->getOutput(0));
    reshape1->setReshapeDimensions(Dims2{maxBatchSize, 576});

    Weights fc1_weight = weightMap["classifier.0.weight"];
    IConstantLayer* fc1_const = network->addConstant(Dims2{1024, 576}, fc1_weight);
    auto mm1 = network->addMatrixMultiply(*reshape1->getOutput(0), MatrixOperation::kNONE, *fc1_const->getOutput(0),
                                          MatrixOperation::kTRANSPOSE);
    assert(mm1);

    Weights fc1_bias = weightMap["classifier.0.bias"];
    IConstantLayer* bias1_const = network->addConstant(Dims2{1, 1024}, fc1_bias);
    auto add1 = network->addElementWise(*mm1->getOutput(0), *bias1_const->getOutput(0), ElementWiseOperation::kSUM);
    assert(add1);

    ILayer* sw2 = hSwish(network, *add1->getOutput(0), "hSwish.0");

    Weights fc2_weight = weightMap["classifier.3.weight"];
    IConstantLayer* fc2_const = network->addConstant(Dims2{1000, 1024}, fc2_weight);
    auto mm2 = network->addMatrixMultiply(*sw2->getOutput(0), MatrixOperation::kNONE, *fc2_const->getOutput(0),
                                          MatrixOperation::kTRANSPOSE);
    assert(mm2);

    Weights fc2_bias = weightMap["classifier.3.bias"];
    IConstantLayer* bias2_const = network->addConstant(Dims2{1, 1000}, fc2_bias);
    auto add2 = network->addElementWise(*mm2->getOutput(0), *bias2_const->getOutput(0), ElementWiseOperation::kSUM);
    assert(add2);

    add2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*add2->getOutput(0));

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    std::cout << "build out" << std::endl;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    delete network;
    return plan;
}

IHostMemory* createEngineLarge(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{static_cast<int>(maxBatchSize), 3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../mbv3_large.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    //auto test1 = network->addActivation(*data, ActivationType::kRELU);
    auto ew1 = convBnHswish(network, weightMap, *data, 16, 3, 2, 1, "features.0.");
    auto ir1 = invertedRes(network, weightMap, *ew1->getOutput(0), "features.1.", 16, 16, 1, 16, 3, 0, 0, 112);
    auto ir2 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.2.", 16, 24, 2, 64, 3, 0, 0, 56);
    auto ir3 = invertedRes(network, weightMap, *ir2->getOutput(0), "features.3.", 24, 24, 1, 72, 3, 0, 0, 56);
    auto ir4 = invertedRes(network, weightMap, *ir3->getOutput(0), "features.4.", 24, 40, 2, 72, 5, 1, 0, 28);
    auto ir5 = invertedRes(network, weightMap, *ir4->getOutput(0), "features.5.", 40, 40, 1, 120, 5, 1, 0, 28);
    auto ir6 = invertedRes(network, weightMap, *ir5->getOutput(0), "features.6.", 40, 40, 1, 120, 5, 1, 0, 28);
    auto ir7 = invertedRes(network, weightMap, *ir6->getOutput(0), "features.7.", 40, 80, 2, 240, 3, 0, 1, 14);
    auto ir8 = invertedRes(network, weightMap, *ir7->getOutput(0), "features.8.", 80, 80, 1, 200, 3, 0, 1, 14);
    auto ir9 = invertedRes(network, weightMap, *ir8->getOutput(0), "features.9.", 80, 80, 1, 184, 3, 0, 1, 14);
    auto ir10 = invertedRes(network, weightMap, *ir9->getOutput(0), "features.10.", 80, 80, 1, 184, 3, 0, 1, 14);
    auto ir11 = invertedRes(network, weightMap, *ir10->getOutput(0), "features.11.", 80, 112, 1, 480, 3, 1, 1, 14);
    auto ir12 = invertedRes(network, weightMap, *ir11->getOutput(0), "features.12.", 112, 112, 1, 672, 3, 1, 1, 14);
    auto ir13 = invertedRes(network, weightMap, *ir12->getOutput(0), "features.13.", 112, 160, 2, 672, 5, 1, 1, 7);
    auto ir14 = invertedRes(network, weightMap, *ir13->getOutput(0), "features.14.", 160, 160, 1, 960, 5, 1, 1, 7);
    auto ir15 = invertedRes(network, weightMap, *ir14->getOutput(0), "features.15.", 160, 160, 1, 960, 5, 1, 1, 7);
    ILayer* ew2 = convBnHswish(network, weightMap, *ir15->getOutput(0), 960, 1, 1, 1, "features.16.");

    IPoolingLayer* pool1 = network->addPoolingNd(*ew2->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool1);
    pool1->setStrideNd(DimsHW{7, 7});

    auto reshape1 = network->addShuffle(*pool1->getOutput(0));
    reshape1->setReshapeDimensions(Dims2{maxBatchSize, 960});
    Weights fc1_weight = weightMap["classifier.0.weight"];
    IConstantLayer* fc1_const = network->addConstant(Dims2{1280, 960}, fc1_weight);
    auto mm1 = network->addMatrixMultiply(*reshape1->getOutput(0), MatrixOperation::kNONE, *fc1_const->getOutput(0),
                                          MatrixOperation::kTRANSPOSE);
    assert(mm1);

    Weights fc1_bias = weightMap["classifier.0.bias"];
    IConstantLayer* bias1_const = network->addConstant(Dims2{1, 1280}, fc1_bias);
    auto add1 = network->addElementWise(*mm1->getOutput(0), *bias1_const->getOutput(0), ElementWiseOperation::kSUM);
    assert(add1);

    ILayer* sw2 = hSwish(network, *add1->getOutput(0), "hSwish.1");
    Weights fc2_weight = weightMap["classifier.3.weight"];
    IConstantLayer* fc2_const = network->addConstant(Dims2{1000, 1280}, fc2_weight);
    auto mm2 = network->addMatrixMultiply(*sw2->getOutput(0), MatrixOperation::kNONE, *fc2_const->getOutput(0),
                                          MatrixOperation::kTRANSPOSE);
    assert(mm2);

    Weights fc2_bias = weightMap["classifier.3.bias"];
    IConstantLayer* bias2_const = network->addConstant(Dims2{1, 1000}, fc2_bias);
    auto add2 = network->addElementWise(*mm2->getOutput(0), *bias2_const->getOutput(0), ElementWiseOperation::kSUM);

    add2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*add2->getOutput(0));

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    std::cout << "build out" << std::endl;
    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    delete network;
    return plan;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string mode) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    IHostMemory* plan = nullptr;

    if (mode == "small") {
        std::cout << "create engine small" << std::endl;
        plan = createEngineSmall(maxBatchSize, builder, config, DataType::kFLOAT);
    } else if (mode == "large") {
        std::cout << "create engine large" << std::endl;
        plan = createEngineLarge(maxBatchSize, builder, config, DataType::kFLOAT);
    }
    assert(plan != nullptr);

    (*modelStream) = plan;

    delete config;
    delete builder;
}

void doInference(nvinfer1::ICudaEngine& engine, float* input, float* output, int batchSize) {
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine.createExecutionContext()};
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return;
    }

    // Define input tensor dimensions, assuming a 4D tensor [batchSize, channels, height, width].
    nvinfer1::Dims4 inputDims(batchSize, 3, INPUT_H, INPUT_W);
    size_t inputSize = batchSize * 3 * INPUT_H * INPUT_W * sizeof(float);
    size_t outputSize = batchSize * OUTPUT_SIZE * sizeof(float);

    // Allocate device memory for input and output tensors.
    void* dInput = nullptr;
    void* dOutput = nullptr;
    CHECK(cudaMalloc(&dInput, inputSize));
    CHECK(cudaMalloc(&dOutput, outputSize));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Copy input data from host to device asynchronously.
    CHECK(cudaMemcpyAsync(dInput, input, inputSize, cudaMemcpyHostToDevice, stream));

    // Bind the device memory buffers to the corresponding tensor names.
    context->setTensorAddress(INPUT_BLOB_NAME, dInput);
    context->setTensorAddress(OUTPUT_BLOB_NAME, dOutput);
    context->setInputShape(INPUT_BLOB_NAME, inputDims);
    context->enqueueV3(stream);

    // Copy the inference output from device back to host asynchronously.
    CHECK(cudaMemcpyAsync(output, dOutput, outputSize, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Release stream and buffers
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(dInput));
    CHECK(cudaFree(dOutput));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mobilenetv3 -s small/large  // serialize small model to plan file" << std::endl;
        std::cerr << "./mobilenetv3 -d small/large  // deserialize and run inference" << std::endl;
        std::cerr << "  image_path           : path to input image (optional, defaults to all-ones)" << std::endl;
        std::cerr << "" << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  ./mobilenetv3 -s small/large          # Build MobileNetV3-Large engine" << std::endl;
        std::cerr << "  ./mobilenetv3 -d small/large dog.jpg  # Run inference on image" << std::endl;
        std::cerr << "  ./mobilenetv3 -d small/large          # Run inference with all-ones test" << std::endl;
        return -1;
    }

    std::string mode = std::string(argv[2]);

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream, mode);
        assert(modelStream != nullptr);

        std::string filename = "mobilenetv3_" + mode + ".plan";
        std::ofstream p(filename, std::ios::binary);
        if (!p) {
            std::cerr << "Could not open plan output file: " << filename << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        p.close();
        delete modelStream;

        std::cout << "Model serialized successfully to " << filename << std::endl;
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::string imagePath;
        if (argc >= 4) {
            imagePath = std::string(argv[3]);
        }

        std::cout << "Using MobileNetV3-" << mode << " model with image: " << imagePath << std::endl;

        std::string engineFile = "mobilenetv3_" + mode + ".plan";
        MyStreamReaderV2 reader(engineFile);
        if (!reader.isOpen()) {
            std::cerr << "Failed to open engine file: " << engineFile << std::endl;
            return -1;
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "Failed to create InferRuntime." << std::endl;
            return -1;
        }

        ICudaEngine* engine = runtime->deserializeCudaEngine(reader);
        if (!engine) {
            std::cerr << "Failed to deserialize engine." << std::endl;
            delete runtime;
            return -1;
        }

        std::vector<float> imageData;
        static float prob[OUTPUT_SIZE];

        // Check if image file exists, otherwise use all-ones
        std::ifstream imageFile(imagePath);
        if (imageFile.good()) {
            imageData = preprocessImage(imagePath);
            std::cout << "Using image: " << imagePath << std::endl;
            std::cout << "Image preprocessed to shape: [1, 3, " << INPUT_H << ", " << INPUT_W << "]" << std::endl;
        } else {
            imageData.assign(3 * INPUT_H * INPUT_W, 1.0f);
            std::cout << "Image not found, using all-ones test data" << std::endl;
            std::cout << "Test data shape: [1, 3, " << INPUT_H << ", " << INPUT_W << "]" << std::endl;
        }

        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        auto start = std::chrono::system_clock::now();
        doInference(*engine, imageData.data(), prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // Print predictions
        if (imageFile.good()) {
            printTopPredictions(prob, OUTPUT_SIZE, 5);
        } else {
            std::cout << "\nOutput:\n\n";
            for (unsigned int i = 0; i < OUTPUT_SIZE; i++) {
                std::cout << prob[i] << ", ";
                if (i % 10 == 0)
                    std::cout << i / 10 << std::endl;
            }
        }
        std::cout << "\n";
        delete engine;
        delete runtime;
    } else {
        std::cerr << "Invalid argument: " << argv[1] << std::endl;
        return -1;
    }

    return 0;
}
