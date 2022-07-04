#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "utils.h"
#include "cuda_runtime_api.h"
#include "logging.h"

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0     // GPU id
#define BATCH_SIZE 1 // currently, only support BATCH=1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 120;
static const int INPUT_W = 160;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME_1 = "semi";
const char *OUTPUT_BLOB_NAME_2 = "desc";

static Logger gLogger;

// create the engine using only the API and not any parser.
ICudaEngine *createEngine(IBuilder *builder, IBuilderConfig *config, std::string path, DataType dt)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(path);

    IConvolutionLayer *conv1a = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["conv1a.weight"], weightMap["conv1a.bias"]);
    assert(conv1a);
    conv1a->setStrideNd(DimsHW{1, 1});
    conv1a->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu1 = network->addActivation(*conv1a->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer *conv1b = network->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv1b.weight"], weightMap["conv1b.bias"]);
    assert(conv1b);
    conv1b->setStrideNd(DimsHW{1, 1});
    conv1b->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu2 = network->addActivation(*conv1b->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IPoolingLayer *pool1 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer *conv2a = network->addConvolutionNd(*pool1->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2a.weight"], weightMap["conv2a.bias"]);
    assert(conv2a);
    conv2a->setStrideNd(DimsHW{1, 1});
    conv2a->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu3 = network->addActivation(*conv2a->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    IConvolutionLayer *conv2b = network->addConvolutionNd(*relu3->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2b.weight"], weightMap["conv2b.bias"]);
    assert(conv2b);
    conv2b->setStrideNd(DimsHW{1, 1});
    conv2b->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu4 = network->addActivation(*conv2b->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    IPoolingLayer *pool2 = network->addPoolingNd(*relu4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer *conv3a = network->addConvolutionNd(*pool2->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv3a.weight"], weightMap["conv3a.bias"]);
    assert(conv3a);
    conv3a->setStrideNd(DimsHW{1, 1});
    conv3a->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu44 = network->addActivation(*conv3a->getOutput(0), ActivationType::kRELU);
    assert(relu44);

    IConvolutionLayer *conv3b = network->addConvolutionNd(*relu44->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv3b.weight"], weightMap["conv3b.bias"]);
    assert(conv3b);
    conv3b->setStrideNd(DimsHW{1, 1});
    conv3b->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu5 = network->addActivation(*conv3b->getOutput(0), ActivationType::kRELU);
    assert(relu5);

    IPoolingLayer *pool3 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool3);
    pool3->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer *conv4a = network->addConvolutionNd(*pool3->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv4a.weight"], weightMap["conv4a.bias"]);
    assert(conv4a);
    conv4a->setStrideNd(DimsHW{1, 1});
    conv4a->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu6 = network->addActivation(*conv4a->getOutput(0), ActivationType::kRELU);
    assert(relu6);

    IConvolutionLayer *conv4b = network->addConvolutionNd(*relu6->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv4b.weight"], weightMap["conv4b.bias"]);
    assert(conv4b);
    conv4b->setStrideNd(DimsHW{1, 1});
    conv4b->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu7 = network->addActivation(*conv4b->getOutput(0), ActivationType::kRELU);
    assert(relu7);

    IConvolutionLayer *convPa = network->addConvolutionNd(*relu7->getOutput(0), 256, DimsHW{3, 3}, weightMap["convPa.weight"], weightMap["convPa.bias"]);
    assert(convPa);
    convPa->setStrideNd(DimsHW{1, 1});
    convPa->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu8 = network->addActivation(*convPa->getOutput(0), ActivationType::kRELU);
    assert(relu8);

    IConvolutionLayer *convPb = network->addConvolutionNd(*relu8->getOutput(0), 65, DimsHW{1, 1}, weightMap["convPb.weight"], weightMap["convPb.bias"]);
    assert(convPb);
    convPb->setStrideNd(DimsHW{1, 1});

    IConvolutionLayer *convDa = network->addConvolutionNd(*relu7->getOutput(0), 256, DimsHW{3, 3}, weightMap["convDa.weight"], weightMap["convDa.bias"]);
    assert(convDa);
    convDa->setStrideNd(DimsHW{1, 1});
    convDa->setPaddingNd(DimsHW{1, 1});
    IActivationLayer *relu9 = network->addActivation(*convDa->getOutput(0), ActivationType::kRELU);
    assert(relu9);

    IConvolutionLayer *convDb = network->addConvolutionNd(*relu9->getOutput(0), 256, DimsHW{1, 1}, weightMap["convDb.weight"], weightMap["convDb.bias"]);
    assert(convDb);
    convDb->setStrideNd(DimsHW{1, 1});

    convPb->getOutput(0)->setName(OUTPUT_BLOB_NAME_1);
    std::cout << "set name out1" << std::endl;
    network->markOutput(*convPb->getOutput(0));

    convDb->getOutput(0)->setName(OUTPUT_BLOB_NAME_2);
    std::cout << "set name out2" << std::endl;
    network->markOutput(*convDb->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1 << 20);

#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif

    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

// Creat the engine using only the API and not any parser.

void APIToModel(std::string path, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(builder, config, path, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

int main(int argc, char **argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 3 && std::string(argv[1]) == "-s")
    {
        IHostMemory *modelStream{nullptr};
        APIToModel(std::string(argv[2]), &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("supernet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./supernet -s <path_to_.wts_file>  // serialize model to plan file" << std::endl;
        return -1;
    }

    return 0;
}
