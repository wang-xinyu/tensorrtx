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

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
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

IElementWiseLayer* convBnRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (ksize - 1) / 2;
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shval[0] = -6.0;
    scval[0] = 1.0;
    pval[0] = 1.0;
    Weights shift{DataType::kFLOAT, shval, 1};
    Weights scale{DataType::kFLOAT, scval, 1};
    Weights power{DataType::kFLOAT, pval, 1};
    weightMap[lname + "cbr.scale"] = scale;
    weightMap[lname + "cbr.shift"] = shift;
    weightMap[lname + "cbr.power"] = power;
    IScaleLayer* scale1 = network->addScale(*bn1->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power);
    assert(scale1);

    IActivationLayer* relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IElementWiseLayer* ew1 =
            network->addElementWise(*relu1->getOutput(0), *relu2->getOutput(0), ElementWiseOperation::kSUB);
    assert(ew1);
    return ew1;
}

ILayer* invertedRes(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                    std::string lname, int inch, int outch, int s, int exp) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int hidden = inch * exp;
    bool use_res_connect = (s == 1 && inch == outch);

    IScaleLayer* bn1 = nullptr;
    if (exp != 1) {
        IElementWiseLayer* ew1 = convBnRelu(network, weightMap, input, hidden, 1, 1, 1, lname + "conv.0.");
        IElementWiseLayer* ew2 =
                convBnRelu(network, weightMap, *ew1->getOutput(0), hidden, 3, s, hidden, lname + "conv.1.");
        IConvolutionLayer* conv1 = network->addConvolutionNd(*ew2->getOutput(0), outch, DimsHW{1, 1},
                                                             weightMap[lname + "conv.2.weight"], emptywts);
        assert(conv1);
        bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv.3", 1e-5);
    } else {
        IElementWiseLayer* ew1 = convBnRelu(network, weightMap, input, hidden, 3, s, hidden, lname + "conv.0.");
        IConvolutionLayer* conv1 = network->addConvolutionNd(*ew1->getOutput(0), outch, DimsHW{1, 1},
                                                             weightMap[lname + "conv.1.weight"], emptywts);
        assert(conv1);
        bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv.2", 1e-5);
    }
    if (!use_res_connect)
        return bn1;
    IElementWiseLayer* ew3 = network->addElementWise(input, *bn1->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew3);
    return ew3;
}

// Creat the engine using only the API and not any parser.
IHostMemory* createPlan(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{static_cast<int>(maxBatchSize), 3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../mobilenet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto ew1 = convBnRelu(network, weightMap, *data, 32, 3, 2, 1, "features.0.");
    ILayer* ir1 = invertedRes(network, weightMap, *ew1->getOutput(0), "features.1.", 32, 16, 1, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.2.", 16, 24, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.3.", 24, 24, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.4.", 24, 32, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.5.", 32, 32, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.6.", 32, 32, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.7.", 32, 64, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.8.", 64, 64, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.9.", 64, 64, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.10.", 64, 64, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.11.", 64, 96, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.12.", 96, 96, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.13.", 96, 96, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.14.", 96, 160, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.15.", 160, 160, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.16.", 160, 160, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.17.", 160, 320, 1, 6);
    IElementWiseLayer* ew2 = convBnRelu(network, weightMap, *ir1->getOutput(0), 1280, 1, 1, 1, "features.18.");

    IPoolingLayer* pool1 = network->addPoolingNd(*ew2->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool1);

    auto inputReshape = network->addShuffle(*pool1->getOutput(0));
    inputReshape->setReshapeDimensions(nvinfer1::Dims2{1, 1280});
    IConstantLayer* filterConst = network->addConstant(nvinfer1::Dims2{1000, 1280}, weightMap["classifier.1.weight"]);
    auto mm = network->addMatrixMultiply(*inputReshape->getOutput(0), MatrixOperation::kNONE,
                                         *filterConst->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(mm);

    IConstantLayer* biasConst = network->addConstant(Dims{2, {1, 1000}}, weightMap["classifier.1.bias"]);
    auto add1 = network->addElementWise(*mm->getOutput(0), *biasConst->getOutput(0), ElementWiseOperation::kSUM);
    add1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*add1->getOutput(0));

    // Build engine
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

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    IHostMemory* plan = createPlan(maxBatchSize, builder, config, DataType::kFLOAT);
    (*modelStream) = plan;
    delete config;
    delete builder;
}

void doInference(nvinfer1::ICudaEngine& engine, float* input, float* output, int batchSize) {
    // Create an execution context from the engine.
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine.createExecutionContext()};
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return;
    }

    // Define input tensor dimensions, assuming a 4D tensor [batchSize, channels, height, width].
    // For MobileNet, channels are typically 3.
    nvinfer1::Dims4 inputDims(batchSize, 3, INPUT_H, INPUT_W);
    size_t inputSize = batchSize * 3 * INPUT_H * INPUT_W * sizeof(float);
    size_t outputSize = batchSize * OUTPUT_SIZE * sizeof(float);

    void* dInput = nullptr;
    void* dOutput = nullptr;
    CHECK(cudaMalloc(&dInput, inputSize));
    CHECK(cudaMalloc(&dOutput, outputSize));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Copy input data from host to device asynchronously
    CHECK(cudaMemcpyAsync(dInput, input, inputSize, cudaMemcpyHostToDevice, stream));

    // Bind the device memory buffers to the corresponding tensor names.
    // This uses the new interface to directly set addresses.
    context->setTensorAddress(INPUT_BLOB_NAME, dInput);
    context->setTensorAddress(OUTPUT_BLOB_NAME, dOutput);
    context->setInputShape(INPUT_BLOB_NAME, inputDims);
    context->enqueueV3(stream);

    // Copy the inference output from device back to host asynchronously.
    CHECK(cudaMemcpyAsync(output, dOutput, outputSize, cudaMemcpyDeviceToHost, stream));

    CHECK(cudaStreamSynchronize(stream));

    // Clean up: destroy the CUDA stream and free device buffers.
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(dInput));
    CHECK(cudaFree(dOutput));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mobilenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mobilenet -d   // deserialize plan file and run inference" << std::endl;
        std::cerr << "  image_path     // path to input image (optional, defaults to all-ones)" << std::endl;
        std::cerr << "" << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  ./mobilenet_v2 -s          # Build MobileNetV2 engine" << std::endl;
        std::cerr << "  ./mobilenet_v2 -d dog.jpg  # Run inference on image" << std::endl;
        std::cerr << "  ./mobilenet_v2 -d          # Run inference with all-ones test" << std::endl;
        return -1;
    }

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("mobilenet.plan", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        p.close();
        delete modelStream;
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::string imagePath;
        if (argc >= 3) {
            imagePath = std::string(argv[2]);
        }
        std::cout << "Using image: " << imagePath << std::endl;

        // Deserialize and run inference.
        std::string engineFile = "mobilenet.plan";
        MyStreamReaderV2 reader(engineFile);  // Construct with file path.
        if (!reader.isOpen()) {
            std::cerr << "Failed to open engine file: " << engineFile << std::endl;
            return -1;
        }

        // Create runtime and deserialize engine.
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(reader);
        if (!engine) {
            std::cerr << "Failed to deserialize engine." << std::endl;
            delete runtime;
            return -1;
        }

        // Run inference
        std::vector<float> imageData;
        static float prob[OUTPUT_SIZE];

        std::ifstream imageFile(imagePath);
        if (imageFile.good()) {
            imageData = preprocessImage(imagePath);
            std::cout << "Image preprocessed to shape: [1, 3, " << INPUT_H << ", " << INPUT_W << "]" << std::endl;
        } else {
            imageData.assign(3 * INPUT_H * INPUT_W, 1.0f);
            std::cout << "Image not found, using all-ones test data" << std::endl;
            std::cout << "Test data shape: [1, 3, " << INPUT_H << ", " << INPUT_W << "]" << std::endl;
        }

        // Run inference with real image
        std::cout << "\nRunning inference..." << std::endl;
        auto start = std::chrono::system_clock::now();
        doInference(*engine, imageData.data(), prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

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

        delete engine;
        delete runtime;
    } else {
        return -1;
    }

    return 0;
}
