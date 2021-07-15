#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <sstream>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 24;
static const int INPUT_W = 94;
static const int OUTPUT_SIZE = 18 * 68;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
using namespace nvinfer1;
const std::string alphabet[] = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                                "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                                "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                                "新",
                                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                                "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                                "W", "X", "Y", "Z", "I", "O", "-"
};


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

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
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input,
                            std::string lname, float eps) {
    float *gamma = (float *) weightMap[lname + ".weight"].values;
    float *beta = (float *) weightMap[lname + ".bias"].values;
    float *mean = (float *) weightMap[lname + ".running_mean"].values;
    float *var = (float *) weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IConvolutionLayer *
small_basic_block(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input,
                  int nbOutputMaps, std::string lname) {
    IConvolutionLayer *conv = network->addConvolutionNd(input, nbOutputMaps / 4, DimsHW{1, 1},
                                                        weightMap[lname + ".block.0.weight"],
                                                        weightMap[lname + ".block.0.bias"]);

    auto relu = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu->getOutput(0), nbOutputMaps / 4, DimsHW{3, 1},
                                                         weightMap[lname + ".block.2.weight"],
                                                         weightMap[lname + ".block.2.bias"]);
    conv2->setPaddingNd(DimsHW{1, 0});
    auto relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer *conv3 = network->addConvolutionNd(*relu2->getOutput(0), nbOutputMaps / 4, DimsHW{1, 3},
                                                         weightMap[lname + ".block.4.weight"],
                                                         weightMap[lname + ".block.4.bias"]);
    conv3->setPaddingNd(DimsHW{0, 1});
    auto relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer *conv4 = network->addConvolutionNd(*relu3->getOutput(0), nbOutputMaps, DimsHW{1, 1},
                                                         weightMap[lname + ".block.6.weight"],
                                                         weightMap[lname + ".block.6.bias"]);

    return conv4;
}



ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {C, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights("../LPRNet.wts");
    //LPRnet
    IConvolutionLayer *conv = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["backbone.0.weight"],weightMap["backbone.0.bias"]);
    assert(conv);
    ILayer *tmp = addBatchNorm2d(network, weightMap, *conv->getOutput(0), "backbone.1", 1e-5);
    auto relu = network->addActivation(*tmp->getOutput(0), ActivationType::kRELU);


    //f0
    auto f0 = network->addPoolingNd(*relu->getOutput(0), PoolingType::kAVERAGE, DimsHW{5, 5});
    f0->setStrideNd(DimsHW{5, 5});

    auto p = network->addPoolingNd(*relu->getOutput(0), PoolingType::kMAX, Dims3{1, 3, 3});
    p->setStrideNd(Dims3{1, 1, 1});

    auto small = small_basic_block(network, weightMap, *p->getOutput(0), 128, "backbone.4");

    ILayer *tmp2 = addBatchNorm2d(network, weightMap, *small->getOutput(0), "backbone.5", 1e-5);
    auto relu2 = network->addActivation(*tmp2->getOutput(0), ActivationType::kRELU);


    auto f1 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{5, 5});
    f1->setStrideNd(DimsHW{5, 5});


    auto p2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, Dims3{1, 3, 3});
    p2->setStrideNd(Dims3{2, 1, 2});

    auto small2 = small_basic_block(network, weightMap, *p2->getOutput(0), 256, "backbone.8");

    ILayer *tmp3 = addBatchNorm2d(network, weightMap, *small2->getOutput(0), "backbone.9", 1e-5);
    auto relu3 = network->addActivation(*tmp3->getOutput(0), ActivationType::kRELU);

    auto small3 = small_basic_block(network, weightMap, *relu3->getOutput(0), 256, "backbone.11");
    ILayer *tmp4 = addBatchNorm2d(network, weightMap, *small3->getOutput(0), "backbone.12", 1e-5);
    auto relu4 = network->addActivation(*tmp4->getOutput(0), ActivationType::kRELU);

    auto f2 = network->addPoolingNd(*relu4->getOutput(0), PoolingType::kAVERAGE, DimsHW{4, 10});
    f2->setStrideNd(DimsHW{4, 2});

    auto p3 = network->addPoolingNd(*relu4->getOutput(0), PoolingType::kMAX, Dims3{1, 3, 3});
    p3->setStrideNd(Dims3{4, 1, 2});
    Dims pf3 = p3->getOutput(0)->getDimensions();
    IConvolutionLayer *conv2 = network->addConvolutionNd(*p3->getOutput(0), 256, DimsHW{1, 4},
                                                         weightMap["backbone.16.weight"],
                                                         weightMap["backbone.16.bias"]);
    ILayer *tmp5 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "backbone.17", 1e-5);
    auto relu5 = network->addActivation(*tmp5->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer *conv3 = network->addConvolutionNd(*relu5->getOutput(0), 68, DimsHW{13, 1},
                                                         weightMap["backbone.20.weight"],
                                                         weightMap["backbone.20.bias"]);

    ILayer *tmp6 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "backbone.21", 1e-5);
    auto backbone = network->addActivation(*tmp6->getOutput(0), ActivationType::kRELU);

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 64 * 4 * 18));
    for (int i = 0; i < 64 * 4 * 18; i++) {
        deval[i] = 2.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 64 * 4 * 18};
    IConstantLayer *d = network->addConstant(Dims4{1, 64, 4, 18}, deconvwts11);
    IElementWiseLayer *f_pow = network->addElementWise(*f0->getOutput(0), *d->getOutput(0), ElementWiseOperation::kPOW);
    Dims pf0 = f0->getOutput(0)->getDimensions();
    Dims pD = d->getOutput(0)->getDimensions();
    Dims pf_pow = f_pow->getOutput(0)->getDimensions();

    auto f_mean = network->addReduce(*f_pow->getOutput(0), ReduceOperation::kAVG, 0XF, true);
    Dims pf_mean = f_mean->getOutput(0)->getDimensions();
    IElementWiseLayer *f_div = network->addElementWise(*f0->getOutput(0), *f_mean->getOutput(0),
                                                       ElementWiseOperation::kDIV);

    Dims pf_div = f_div->getOutput(0)->getDimensions();
    float *deval2 = reinterpret_cast<float *>(malloc(sizeof(float) * 1 * 128 * 4 * 18));
    for (int i = 0; i < 128 * 4 * 18 * 1; i++) {
        deval2[i] = 2.0;
    }
    Weights deconvwts22{DataType::kFLOAT, deval2, 128 * 4 * 18 * 1};
    IConstantLayer *d2 = network->addConstant(Dims4{1, 128, 4, 18}, deconvwts22);
    IElementWiseLayer *f_pow2 = network->addElementWise(*f1->getOutput(0), *d2->getOutput(0),
                                                        ElementWiseOperation::kPOW);
    auto f_mean2 = network->addReduce(*f_pow2->getOutput(0), ReduceOperation::kAVG, 0XF, true);
    IElementWiseLayer *f_div2 = network->addElementWise(*f1->getOutput(0), *f_mean2->getOutput(0),
                                                        ElementWiseOperation::kDIV);


    float *deval3 = reinterpret_cast<float *>(malloc(sizeof(float) * 256 * 4 * 18 * 1));
    for (int i = 0; i < 256 * 4 * 18 * 1; i++) {
        deval3[i] = 2.0;
    }
    Weights deconvwts33{DataType::kFLOAT, deval3, 256 * 4 * 18 * 1};
    IConstantLayer *d3 = network->addConstant(Dims4{1, 256, 4, 18}, deconvwts33);
    IElementWiseLayer *f_pow3 = network->addElementWise(*f2->getOutput(0), *d3->getOutput(0),
                                                        ElementWiseOperation::kPOW);
    auto f_mean3 = network->addReduce(*f_pow3->getOutput(0), ReduceOperation::kAVG, 0XF, true);
    IElementWiseLayer *f_div3 = network->addElementWise(*f2->getOutput(0), *f_mean3->getOutput(0),
                                                        ElementWiseOperation::kDIV);


    float *deval4 = reinterpret_cast<float *>(malloc(sizeof(float) * 68 * 4 * 18 * 1));
    for (int i = 0; i < 68 * 4 * 18 * 1; i++) {
        deval4[i] = 2.0;
    }
    Weights deconvwts44{DataType::kFLOAT, deval4, 68 * 4 * 18 * 1};
    IConstantLayer *d4 = network->addConstant(Dims4{1, 68, 4, 18}, deconvwts44);
    IElementWiseLayer *f_pow4 = network->addElementWise(*backbone->getOutput(0), *d4->getOutput(0),
                                                        ElementWiseOperation::kPOW);
    auto f_mean4 = network->addReduce(*f_pow4->getOutput(0), ReduceOperation::kAVG, 0XF, true);
    IElementWiseLayer *f_div4 = network->addElementWise(*backbone->getOutput(0), *f_mean4->getOutput(0),
                                                        ElementWiseOperation::kDIV);


    ITensor *inputTensors[] = {f_div->getOutput(0), f_div2->getOutput(0), f_div3->getOutput(0), f_div4->getOutput(0)};

    auto f_divdims = f_div->getOutput(0)->getDimensions();
    auto f_div2dims = f_div2->getOutput(0)->getDimensions();
    auto f_div3dims = f_div3->getOutput(0)->getDimensions();
    auto backbonedims = backbone->getOutput(0)->getDimensions();
    auto cat = network->addConcatenation(inputTensors, 4);
    Dims pcat = cat->getOutput(0)->getDimensions();
    IConvolutionLayer *container = network->addConvolutionNd(*cat->getOutput(0), 68, DimsHW{1, 1},
                                                             weightMap["container.0.weight"],
                                                             weightMap["container.0.bias"]);

    auto logits = network->addReduce(*container->getOutput(0), ReduceOperation::kAVG, 0X04, false);

    Dims dims = logits->getOutput(0)->getDimensions();
    std::cout << "logits shape " << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << std::endl;

    logits->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*logits->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;
}


void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}


void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char **argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("LPRnet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("LPRnet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./LPRnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./LPRnet -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    cv::Mat img = cv::imread("../1.jpg");
    cv::Mat pr_img;
    cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_CUBIC);
    // For multi-batch, I feed the same image multiple times.
    // If you want to process different images in a batch, you need adapt it.
   //cv::Mat blob = cv::dnn::blobFromImage(pr_img, 0.0078125, pr_img.size(), cv::Scalar(127.5, 127.5, 127.5), true,
                                          //false);
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[2] - 127.5)*0.0078125;
            data[i + INPUT_H * INPUT_W] = ((float)uc_pixel[1]-127.5)*0.0078125;
            data[i] = ((float)uc_pixel[0]-127.5)*0.0078125;
            uc_pixel += 3;
            ++i;
        }
    }

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    std::vector<int> preds;
    std::cout << std::endl;
    for (int i = 0; i < 18; i++) {
        int maxj = 0;
        for (int j = 0; j < 68; j++) {
            if (prob[i + 18 * j] > prob[i + 18 * maxj]) maxj = j;
        }
        preds.push_back(maxj);
    }
    int pre_c = preds[0];
    std::vector<int> no_repeat_blank_label;
    for (auto c: preds) {
        if (c == pre_c || c == 68 - 1) {
            if (c == 68 - 1) pre_c = c;
            continue;
        }
        no_repeat_blank_label.push_back(c);
        pre_c = c;
    }
    std::string str;
    for (auto v: no_repeat_blank_label) {
        str += alphabet[v];
    }
    std::cout<<"result:"<<str<<std::endl;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    return 0;
}
