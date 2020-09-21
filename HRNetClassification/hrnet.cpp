#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace nvinfer1;
static Logger gLogger;

#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

const char* INPUT_BLOB_NAME = "image";
const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
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

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;

}
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    //std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, std::string convname, std::string bnname, bool bias=false) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1;
    //Dims dim;
    if (!bias)
    {
        conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[convname + ".weight"], emptywts);
        
    }
    else
    {
        conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[convname + ".weight"], weightMap[convname + ".bias"]);
    }
   
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), bnname, 1e-4);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    return lr;
}

IActivationLayer* ResBlock2Conv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolution(input, inch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ stride, stride });
    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    ///
    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), inch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ stride, stride });
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    //////
    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
    assert(conv3);
    conv1->setStride(DimsHW{ stride, stride });
    conv3->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv4 = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + ".downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{ stride, stride });
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

IActivationLayer* ResBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // in 256 out 64
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ stride, stride });
    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    ///
    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ stride, stride });
    conv2->setPadding(DimsHW{ 1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    //////
    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), inch, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
    assert(conv3);
    conv1->setStride(DimsHW{ stride, stride });
    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);

    IElementWiseLayer* ew1;
    ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

IActivationLayer* liteResBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // in 256 out 64
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 1, 1 });
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    ///
    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ 1, 1 });
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IElementWiseLayer* ew1;
    ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

ILayer* netAddUpsample(INetworkDefinition* network, ITensor* input, int inputChannels, int stride)
{
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3); // chw
    assert(inpDims.d[1] == inpDims.d[2]);
    int h = inpDims.d[1];
    int w = inpDims.d[2];
    // add pre multiply matrix as a constant
    /*
    kSPATIA Elements correspond to different spatial data.

    kCHANNEL Elements correspond to different channels.
    */
    nvinfer1::Dims preDims{ 3,
                           {1, stride * h, w},
                           {nvinfer1::DimensionType::kCHANNEL,
                            nvinfer1::DimensionType::kSPATIAL,
                            nvinfer1::DimensionType::kSPATIAL} };
    int size = stride * h * w;
    nvinfer1::Weights preMul{ nvinfer1::DataType::kFLOAT, nullptr, size };
    float* preWt = new float[size];
    /* (2*h * w)
    [ [1, 0, ..., 0],
      [1, 0, ..., 0],
      [0, 1, ..., 0],
      [0, 1, ..., 0],
      ...,
      ...,
      [0, 0, ..., 1],
      [0, 0, ..., 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int s = 0; s < stride; ++s)
        {
            for (int j = 0; j < w; ++j, ++idx)
            {
                preWt[idx] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    preMul.values = preWt;
    nvinfer1::IConstantLayer* preM = network->addConstant(preDims, preMul);
    assert(preM != nullptr);
    //std::string preLayerName = "preMul_" + std::to_string(layerIdx);
    //preM->setName(preLayerName.c_str());
    // add post multiply matrix as a constant
    nvinfer1::Dims postDims{ 3,
                            {1, h, stride * w},
                            {nvinfer1::DimensionType::kCHANNEL,
                             nvinfer1::DimensionType::kSPATIAL,
                             nvinfer1::DimensionType::kSPATIAL} };
    size = stride * h * w;
    nvinfer1::Weights postMul{ nvinfer1::DataType::kFLOAT, nullptr, size };
    float* postWt = new float[size];
    /* (h * 2*w)
    [ [1, 1, 0, 0, ..., 0, 0],
      [0, 0, 1, 1, ..., 0, 0],
      ...,
      ...,
      [0, 0, 0, 0, ..., 1, 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int j = 0; j < stride * w; ++j, ++idx)
        {
            postWt[idx] = (j / stride == i) ? 1.0 : 0.0;
        }
    }
    postMul.values = postWt;
    nvinfer1::IConstantLayer* post_m = network->addConstant(postDims, postMul);
    assert(post_m != nullptr);
    // add matrix multiply layers for upsampling
    nvinfer1::IMatrixMultiplyLayer* mm1
        = network->addMatrixMultiply(*preM->getOutput(0),
            nvinfer1::MatrixOperation::kNONE, *input,
            nvinfer1::MatrixOperation::kNONE);
    assert(mm1 != nullptr);
    nvinfer1::IMatrixMultiplyLayer* mm2
        = network->addMatrixMultiply(*mm1->getOutput(0),
            nvinfer1::MatrixOperation::kNONE,
            *post_m->getOutput(0),
            nvinfer1::MatrixOperation::kNONE);
    assert(mm2 != nullptr);
    return mm2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("E:\\LearningCodes\\GithubRepo\\HRNet-Image-Classification\\tools\\HRNetClassify.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

   //ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx
    auto id_993 = convBnLeaky(network, weightMap, *data, 64, 3, 2, 1, "conv1", "bn1");  //conv1.weight 
    auto id_996 = convBnLeaky(network, weightMap, *id_993->getOutput(0), 64, 3, 2, 1, "conv2", "bn2");  //conv1.weight                                                                                 //Res
    // IActivationLayer* ResBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    auto id_1008 = ResBlock2Conv(network, weightMap, *id_996->getOutput(0), 64, 256, 1, "layer1.0"); 
    auto id_1018 = ResBlock(network, weightMap, *id_1008->getOutput(0), 256, 64, 1, "layer1.1");
    
    // transition1-1
    auto id_1021 = convBnLeaky(network, weightMap, *id_1018->getOutput(0), 18, 3, 1, 1, "transition1.0.0", "transition1.0.1");
    auto id_1031 = liteResBlock(network, weightMap, *id_1021->getOutput(0), 18, "stage2.0.branches.0.0");
    auto id_1038 = liteResBlock(network, weightMap, *id_1031->getOutput(0), 18, "stage2.0.branches.0.1");
    //右侧分支
    auto id_1024 = convBnLeaky(network, weightMap, *id_1018->getOutput(0), 36, 3, 2, 1, "transition1.1.0.0", "transition1.1.0.1");
    auto id_1045 = liteResBlock(network, weightMap, *id_1024->getOutput(0), 36, "stage2.0.branches.1.0");
    auto id_1052 = liteResBlock(network, weightMap, *id_1045->getOutput(0), 36, "stage2.0.branches.1.1");
    //dim = id_1052->getOutput(0)->getDimensions();
    // conv+bn+upsample
    IConvolutionLayer* id_1053 = network->addConvolution(*id_1052->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage2.0.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1053);
    id_1053->setStride(DimsHW{ 1, 1 });
    id_1053->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* id_1054 = addBatchNorm2d(network, weightMap, *id_1053->getOutput(0), "stage2.0.fuse_layers.0.1.1", 1e-5);
    //dim = id_1053->getOutput(0)->getDimensions();
   // dim = id_1054->getOutput(0)->getDimensions();

    ILayer* id_1083 = netAddUpsample(network, id_1054->getOutput(0), 18, 2);
    //dim = id_1083->getOutput(0)->getDimensions();
    IElementWiseLayer* id_1084 = network->addElementWise(*id_1083->getOutput(0), *id_1038->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1085 = network->addActivation(*id_1084->getOutput(0), ActivationType::kRELU);
    //dim = id_1085->getOutput(0)->getDimensions();
    // transition1-2
    IConvolutionLayer* id_1086 = network->addConvolution(*id_1038->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage2.0.fuse_layers.1.0.0.0.weight"], emptywts);
    assert(id_1086);
    id_1086->setStride(DimsHW{ 2, 2 });
    id_1086->setPadding(DimsHW{ 1, 1 });
    //dim = id_1086->getOutput(0)->getDimensions();

    IScaleLayer* id_1087 = addBatchNorm2d(network, weightMap, *id_1086->getOutput(0), "stage2.0.fuse_layers.1.0.0.1", 1e-5);
    IElementWiseLayer* id_1088 = network->addElementWise(*id_1087->getOutput(0), *id_1052->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1089 = network->addActivation(*id_1088->getOutput(0), ActivationType::kRELU);
    //dim = id_1087->getOutput(0)->getDimensions();
    //dim = id_1088->getOutput(0)->getDimensions();
    //dim = id_1089->getOutput(0)->getDimensions();

    ///////////////////////////////////
    // transition2-1  stage_3
    auto id_1099 = liteResBlock(network, weightMap, *id_1085->getOutput(0), 18, "stage3.0.branches.0.0");
    auto id_1106 = liteResBlock(network, weightMap, *id_1099->getOutput(0), 18, "stage3.0.branches.0.1");
    // transition2-2  stage_3
    auto id_1113 = liteResBlock(network, weightMap, *id_1089->getOutput(0), 36, "stage3.0.branches.1.0");
    auto id_1120 = liteResBlock(network, weightMap, *id_1113->getOutput(0), 36, "stage3.0.branches.1.1");
    // transition2-3  stage_3
    auto id_1092 = convBnLeaky(network, weightMap, *id_1089->getOutput(0), 72, 3, 2, 1, "transition2.2.0.0", "transition2.2.0.1");
    //dim = id_1092->getOutput(0)->getDimensions();  // 14
    auto id_1127 = liteResBlock(network, weightMap, *id_1092->getOutput(0), 72, "stage3.0.branches.2.0");
    auto id_1134 = liteResBlock(network, weightMap, *id_1127->getOutput(0), 72, "stage3.0.branches.2.1");

    /////// 多分辨率模块 密集连接
    //conv bn up
    //dim = id_1120->getOutput(0)->getDimensions();
    IConvolutionLayer* id_1135 = network->addConvolution(*id_1120->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage3.0.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1135);
    id_1135->setStride(DimsHW{ 1, 1 });
    id_1135->setPadding(DimsHW{ 0, 0 });
    //dim = id_1135->getOutput(0)->getDimensions();
    IScaleLayer* id_1136 = addBatchNorm2d(network, weightMap, *id_1135->getOutput(0), "stage3.0.fuse_layers.0.1.1", 1e-5);
    //dim = id_1136->getOutput(0)->getDimensions();
    ILayer* id_1165 = netAddUpsample(network, id_1136->getOutput(0), 18, 2);
    IElementWiseLayer* id_1166 = network->addElementWise(*id_1165->getOutput(0), *id_1106->getOutput(0), ElementWiseOperation::kSUM);
    
    IConvolutionLayer* id_1167 = network->addConvolution(*id_1134->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage3.0.fuse_layers.0.2.0.weight"], emptywts);
    assert(id_1167);
    id_1167->setStride(DimsHW{ 1, 1 });
    id_1167->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1168 = addBatchNorm2d(network, weightMap, *id_1167->getOutput(0), "stage3.0.fuse_layers.0.2.1", 1e-5);
    //dim = id_1168->getOutput(0)->getDimensions();
    ILayer* id_1197 = netAddUpsample(network, id_1168->getOutput(0), 18, 4);
    //dim = id_1197->getOutput(0)->getDimensions();
    //dim = id_1166->getOutput(0)->getDimensions();
    IElementWiseLayer* id_1198 = network->addElementWise(*id_1166->getOutput(0), *id_1197->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1199 = network->addActivation(*id_1198->getOutput(0), ActivationType::kRELU);

    //2
    //dim = id_1106->getOutput(0)->getDimensions();
    IConvolutionLayer* id_1200 = network->addConvolution(*id_1106->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage3.0.fuse_layers.1.0.0.0.weight"], emptywts);
    assert(id_1200);
    id_1200->setStride(DimsHW{ 2, 2 });
    id_1200->setPadding(DimsHW{ 1, 1 });
    //dim = id_1200->getOutput(0)->getDimensions();

    IScaleLayer* id_1201 = addBatchNorm2d(network, weightMap, *id_1200->getOutput(0), "stage3.0.fuse_layers.1.0.0.1", 1e-5);
    IElementWiseLayer* id_1202 = network->addElementWise(*id_1201->getOutput(0), *id_1120->getOutput(0), ElementWiseOperation::kSUM);
    //dim = id_1202->getOutput(0)->getDimensions();

    //dim = id_1134->getOutput(0)->getDimensions();
    IConvolutionLayer* id_1203 = network->addConvolution(*id_1134->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage3.0.fuse_layers.1.2.0.weight"], emptywts);
    assert(id_1203);
    id_1203->setStride(DimsHW{ 1, 1 });
    id_1203->setPadding(DimsHW{ 0, 0 });
    //dim = id_1203->getOutput(0)->getDimensions();
    IScaleLayer* id_1204 = addBatchNorm2d(network, weightMap, *id_1203->getOutput(0), "stage3.0.fuse_layers.1.2.1", 1e-5);
    //dim = id_1204->getOutput(0)->getDimensions();
    ILayer* id_1233 = netAddUpsample(network, id_1204->getOutput(0), 36, 2);
    //dim = id_1233->getOutput(0)->getDimensions();
    IElementWiseLayer* id_1234 = network->addElementWise(*id_1202->getOutput(0), *id_1233->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1235 = network->addActivation(*id_1234->getOutput(0), ActivationType::kRELU);

    // 3
    IConvolutionLayer* id_1236 = network->addConvolution(*id_1106->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage3.0.fuse_layers.2.0.0.0.weight"], emptywts);
    assert(id_1236);
    id_1236->setStride(DimsHW{ 2, 2 });
    id_1236->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1237 = addBatchNorm2d(network, weightMap, *id_1236->getOutput(0), "stage3.0.fuse_layers.2.0.0.1", 1e-5);
    IActivationLayer* id_1238 = network->addActivation(*id_1237->getOutput(0), ActivationType::kRELU);
    
    IConvolutionLayer* id_1239 = network->addConvolution(*id_1238->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage3.0.fuse_layers.2.0.1.0.weight"], emptywts);
    assert(id_1239);
    id_1239->setStride(DimsHW{ 2, 2 });
    id_1239->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1240 = addBatchNorm2d(network, weightMap, *id_1239->getOutput(0), "stage3.0.fuse_layers.2.0.1.1", 1e-5);

    IConvolutionLayer* id_1241 = network->addConvolution(*id_1120->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage3.0.fuse_layers.2.1.0.0.weight"], emptywts);
    assert(id_1241);
    id_1241->setStride(DimsHW{ 2, 2 });
    id_1241->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1242 = addBatchNorm2d(network, weightMap, *id_1241->getOutput(0), "stage3.0.fuse_layers.2.1.0.1", 1e-5);

    IElementWiseLayer* id_1243 = network->addElementWise(*id_1240->getOutput(0), *id_1242->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_1244 = network->addElementWise(*id_1243->getOutput(0), *id_1134->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1245 = network->addActivation(*id_1244->getOutput(0), ActivationType::kRELU);

    //
    auto id_1252 = liteResBlock(network, weightMap, *id_1199->getOutput(0), 18, "stage3.1.branches.0.0");
    auto id_1259 = liteResBlock(network, weightMap, *id_1252->getOutput(0), 18, "stage3.1.branches.0.1");
    auto id_1266 = liteResBlock(network, weightMap, *id_1235->getOutput(0), 36, "stage3.1.branches.1.0");
    auto id_1273 = liteResBlock(network, weightMap, *id_1266->getOutput(0), 36, "stage3.1.branches.1.1");
    auto id_1280 = liteResBlock(network, weightMap, *id_1245->getOutput(0), 72, "stage3.1.branches.2.0");
    auto id_1287 = liteResBlock(network, weightMap, *id_1280->getOutput(0), 72, "stage3.1.branches.2.1");

    /////// 多分辨率模块 密集连接 
    //1: （1259+up(1273)）+up(1287)
    //1-1  1259+up(1273)
    //dim = id_1273->getOutput(0)->getDimensions();
    IConvolutionLayer* id_1288 = network->addConvolution(*id_1273->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage3.1.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1288);
    id_1288->setStride(DimsHW{ 1, 1 });
    id_1288->setPadding(DimsHW{ 0, 0 });
    //dim = id_1288->getOutput(0)->getDimensions();
    IScaleLayer* id_1289 = addBatchNorm2d(network, weightMap, *id_1288->getOutput(0), "stage3.1.fuse_layers.0.1.1", 1e-5);
    ILayer* id_1318 = netAddUpsample(network, id_1289->getOutput(0), 18, 2);
    IElementWiseLayer* id_1319 = network->addElementWise(*id_1259->getOutput(0), *id_1318->getOutput(0), ElementWiseOperation::kSUM);
    //dim = id_1319->getOutput(0)->getDimensions();
    //1-2 up(1287)  conv bn up
    IConvolutionLayer* id_1320 = network->addConvolution(*id_1134->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage3.1.fuse_layers.0.2.0.weight"], emptywts);
    assert(id_1320);
    id_1320->setStride(DimsHW{ 1, 1 });
    id_1320->setPadding(DimsHW{ 0, 0 });
    //dim = id_1320->getOutput(0)->getDimensions();

    IScaleLayer* id_1321 = addBatchNorm2d(network, weightMap, *id_1320->getOutput(0), "stage3.1.fuse_layers.0.2.1", 1e-5);
    ILayer* id_1350 = netAddUpsample(network, id_1321->getOutput(0), 18, 4);
    //1-3: + / relu
    IElementWiseLayer* id_1351 = network->addElementWise(*id_1319->getOutput(0), *id_1350->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1352 = network->addActivation(*id_1351->getOutput(0), ActivationType::kRELU);
    

    //2: conv(1259)+1273 + up(1287)
    IConvolutionLayer* id_1353 = network->addConvolution(*id_1259->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage3.1.fuse_layers.1.0.0.0.weight"], emptywts);
    assert(id_1353);
    id_1353->setStride(DimsHW{ 2, 2 });
    id_1353->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1354 = addBatchNorm2d(network, weightMap, *id_1353->getOutput(0), "stage3.1.fuse_layers.1.0.0.1", 1e-5);
    IElementWiseLayer* id_1355 = network->addElementWise(*id_1354->getOutput(0), *id_1273->getOutput(0), ElementWiseOperation::kSUM);
   
    
    IConvolutionLayer* id_1356 = network->addConvolution(*id_1287->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage3.1.fuse_layers.1.2.0.weight"], emptywts);
    assert(id_1356);
    id_1356->setStride(DimsHW{ 1, 1 });
    id_1356->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1357 = addBatchNorm2d(network, weightMap, *id_1356->getOutput(0), "stage3.1.fuse_layers.1.2.1", 1e-5);
    //dim = id_1357->getOutput(0)->getDimensions();
    ILayer* id_1386 = netAddUpsample(network, id_1357->getOutput(0), 36, 2);
    IElementWiseLayer* id_1387 = network->addElementWise(*id_1355->getOutput(0), *id_1386->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1388 = network->addActivation(*id_1387->getOutput(0), ActivationType::kRELU);

    //3 conv(1259)+conv(1273)+1287
    IConvolutionLayer* id_1389 = network->addConvolution(*id_1259->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage3.1.fuse_layers.2.0.0.0.weight"], emptywts);
    assert(id_1389);
    id_1389->setStride(DimsHW{ 2, 2 });
    id_1389->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1390 = addBatchNorm2d(network, weightMap, *id_1389->getOutput(0), "stage3.1.fuse_layers.2.0.0.1", 1e-5);
    IActivationLayer* id_1391 = network->addActivation(*id_1390->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* id_1392 = network->addConvolution(*id_1391->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage3.1.fuse_layers.2.0.1.0.weight"], emptywts);
    assert(id_1392);
    id_1392->setStride(DimsHW{ 2, 2 });
    id_1392->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1393 = addBatchNorm2d(network, weightMap, *id_1392->getOutput(0), "stage3.1.fuse_layers.2.0.1.1", 1e-5);

    IConvolutionLayer* id_1394 = network->addConvolution(*id_1273->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage3.1.fuse_layers.2.1.0.0.weight"], emptywts);
    assert(id_1394);
    id_1394->setStride(DimsHW{ 2, 2 });
    id_1394->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1395 = addBatchNorm2d(network, weightMap, *id_1394->getOutput(0), "stage3.1.fuse_layers.2.1.0.1", 1e-5);

    IElementWiseLayer* id_1396 = network->addElementWise(*id_1393->getOutput(0), *id_1395->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_1397 = network->addElementWise(*id_1396->getOutput(0), *id_1287->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1398 = network->addActivation(*id_1397->getOutput(0), ActivationType::kRELU);

    //
    auto id_1405 = liteResBlock(network, weightMap, *id_1352->getOutput(0), 18, "stage3.2.branches.0.0");
    auto id_1412= liteResBlock(network, weightMap, *id_1405->getOutput(0), 18, "stage3.2.branches.0.1");
    auto id_1419 = liteResBlock(network, weightMap, *id_1388->getOutput(0), 36, "stage3.2.branches.1.0");
    auto id_1426 = liteResBlock(network, weightMap, *id_1419->getOutput(0), 36, "stage3.2.branches.1.1");
    auto id_1433 = liteResBlock(network, weightMap, *id_1398->getOutput(0), 72, "stage3.2.branches.2.0");
    auto id_1440 = liteResBlock(network, weightMap, *id_1433->getOutput(0), 72, "stage3.2.branches.2.1");


    // 1412 + up(1426)+up(1440) 
    IConvolutionLayer* id_1441 = network->addConvolution(*id_1426->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage3.2.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1441);
    id_1441->setStride(DimsHW{ 1, 1 });
    id_1441->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1442 = addBatchNorm2d(network, weightMap, *id_1441->getOutput(0), "stage3.2.fuse_layers.0.1.1", 1e-5);
    ILayer* id_1471 = netAddUpsample(network, id_1442->getOutput(0), 18, 2);
    IElementWiseLayer* id_1472 = network->addElementWise(*id_1412->getOutput(0), *id_1471->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1473= network->addConvolution(*id_1440->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage3.2.fuse_layers.0.2.0.weight"], emptywts);
    assert(id_1473);
    id_1473->setStride(DimsHW{ 1, 1 });
    id_1473->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1474 = addBatchNorm2d(network, weightMap, *id_1473->getOutput(0), "stage3.2.fuse_layers.0.2.1", 1e-5);
    ILayer* id_1503 = netAddUpsample(network, id_1474->getOutput(0), 18, 4);
   
    IElementWiseLayer* id_1504 = network->addElementWise(*id_1472->getOutput(0), *id_1503->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1505 = network->addActivation(*id_1504->getOutput(0), ActivationType::kRELU);

    // conv(1412)+1426+up(1440)
    IConvolutionLayer* id_1506 = network->addConvolution(*id_1412->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage3.2.fuse_layers.1.0.0.0.weight"], emptywts);
    assert(id_1506);
    id_1506->setStride(DimsHW{ 2, 2 });
    id_1506->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1507 = addBatchNorm2d(network, weightMap, *id_1506->getOutput(0), "stage3.2.fuse_layers.1.0.0.1", 1e-5);
    IElementWiseLayer* id_1508= network->addElementWise(*id_1507->getOutput(0), *id_1426->getOutput(0), ElementWiseOperation::kSUM);
                                                                                                                
    IConvolutionLayer* id_1509 = network->addConvolution(*id_1440->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage3.2.fuse_layers.1.2.0.weight"], emptywts);
    assert(id_1509);
    id_1509->setStride(DimsHW{ 1, 1 });
    id_1509->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1510 = addBatchNorm2d(network, weightMap, *id_1509->getOutput(0), "stage3.2.fuse_layers.1.2.1", 1e-5);
    ILayer* id_1539 = netAddUpsample(network, id_1510->getOutput(0), 36, 2);
    IElementWiseLayer* id_1540 = network->addElementWise(*id_1508->getOutput(0), *id_1539->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1541 = network->addActivation(*id_1540->getOutput(0), ActivationType::kRELU);

    // conv(1412)+conv(1426)+1440
    IConvolutionLayer* id_1542 = network->addConvolution(*id_1412->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage3.2.fuse_layers.2.0.0.0.weight"], emptywts);
    assert(id_1542);
    id_1542->setStride(DimsHW{ 2, 2 });
    id_1542->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1543 = addBatchNorm2d(network, weightMap, *id_1542->getOutput(0), "stage3.2.fuse_layers.2.0.0.1", 1e-5);
    IActivationLayer* id_1544 = network->addActivation(*id_1543->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* id_1545 = network->addConvolution(*id_1544->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage3.2.fuse_layers.2.0.1.0.weight"], emptywts);
    assert(id_1545);
    id_1545->setStride(DimsHW{ 2, 2 });
    id_1545->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1546 = addBatchNorm2d(network, weightMap, *id_1545->getOutput(0), "stage3.2.fuse_layers.2.0.1.1", 1e-5);

    IConvolutionLayer* id_1547 = network->addConvolution(*id_1426->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage3.2.fuse_layers.2.1.0.0.weight"], emptywts);
    assert(id_1547);
    id_1547->setStride(DimsHW{ 2, 2 });
    id_1547->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1548 = addBatchNorm2d(network, weightMap, *id_1547->getOutput(0), "stage3.2.fuse_layers.2.1.0.1", 1e-5);

    IElementWiseLayer* id_1549 = network->addElementWise(*id_1546->getOutput(0), *id_1548->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_1550 = network->addElementWise(*id_1549->getOutput(0), *id_1440->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1551 = network->addActivation(*id_1550->getOutput(0), ActivationType::kRELU);

    //
    auto id_1561 = liteResBlock(network, weightMap, *id_1505->getOutput(0), 18, "stage4.0.branches.0.0");
    auto id_1568 = liteResBlock(network, weightMap, *id_1561->getOutput(0), 18, "stage4.0.branches.0.1");
    auto id_1575 = liteResBlock(network, weightMap, *id_1541->getOutput(0), 36, "stage4.0.branches.1.0");
    auto id_1582 = liteResBlock(network, weightMap, *id_1575->getOutput(0), 36, "stage4.0.branches.1.1");
    auto id_1589 = liteResBlock(network, weightMap, *id_1551->getOutput(0), 72, "stage4.0.branches.2.0");
    auto id_1596 = liteResBlock(network, weightMap, *id_1589->getOutput(0), 72, "stage4.0.branches.2.1");

    // transition
    auto id_1554 = convBnLeaky(network, weightMap, *id_1551->getOutput(0), 144, 3, 2, 1, "transition3.3.0.0", "transition3.3.0.1");
    auto id_1603 = liteResBlock(network, weightMap, *id_1554->getOutput(0), 144, "stage4.0.branches.3.0");
    auto id_1610 = liteResBlock(network, weightMap, *id_1603->getOutput(0), 144, "stage4.0.branches.3.1");

    // 下面的就是4个分支了
    // 1568+up(1582)+up(1596)+up(1610)
    IConvolutionLayer* id_1611 = network->addConvolution(*id_1582->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage4.0.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1611);
    id_1611->setStride(DimsHW{ 1, 1 });
    id_1611->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1612 = addBatchNorm2d(network, weightMap, *id_1611->getOutput(0), "stage4.0.fuse_layers.0.1.1", 1e-5);
    ILayer* id_1641 = netAddUpsample(network, id_1612->getOutput(0), 18, 2);
    IElementWiseLayer* id_1642 = network->addElementWise(*id_1641->getOutput(0), *id_1568->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1643 = network->addConvolution(*id_1596->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage4.0.fuse_layers.0.2.0.weight"], emptywts);
    assert(id_1643);
    id_1643->setStride(DimsHW{ 1, 1 });
    id_1643->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1644 = addBatchNorm2d(network, weightMap, *id_1643->getOutput(0), "stage4.0.fuse_layers.0.2.1", 1e-5);
    ILayer* id_1673 = netAddUpsample(network, id_1644->getOutput(0), 18, 4);
    IElementWiseLayer* id_1674 = network->addElementWise(*id_1642->getOutput(0), *id_1673->getOutput(0), ElementWiseOperation::kSUM);

    //3
    IConvolutionLayer* id_1675 = network->addConvolution(*id_1610->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage4.0.fuse_layers.0.3.0.weight"], emptywts);
    assert(id_1675);
    id_1675->setStride(DimsHW{ 1, 1 });
    id_1675->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1676 = addBatchNorm2d(network, weightMap, *id_1675->getOutput(0), "stage4.0.fuse_layers.0.3.1", 1e-5);
    ILayer* id_1705 = netAddUpsample(network, id_1676->getOutput(0), 18, 8);
    IElementWiseLayer* id_1706 = network->addElementWise(*id_1705->getOutput(0), *id_1674->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1707 = network->addActivation(*id_1706->getOutput(0), ActivationType::kRELU);

    // conv(1568)+1582+up(1596)+up(1610)
    IConvolutionLayer* id_1708 = network->addConvolution(*id_1568->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.1.0.0.0.weight"], emptywts);
    assert(id_1708);
    id_1708->setStride(DimsHW{ 2, 2 });
    id_1708->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1709 = addBatchNorm2d(network, weightMap, *id_1708->getOutput(0), "stage4.0.fuse_layers.1.0.0.1", 1e-5);
    IElementWiseLayer* id_1710 = network->addElementWise(*id_1709->getOutput(0), *id_1582->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1711 = network->addConvolution(*id_1596->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage4.0.fuse_layers.1.2.0.weight"], emptywts);
    assert(id_1711);
    id_1711->setStride(DimsHW{ 1, 1 });
    id_1711->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1712 = addBatchNorm2d(network, weightMap, *id_1711->getOutput(0), "stage4.0.fuse_layers.1.2.1", 1e-5);
    ILayer* id_1741 = netAddUpsample(network, id_1712->getOutput(0), 36, 2);
    IElementWiseLayer* id_1742 = network->addElementWise(*id_1741->getOutput(0), *id_1710->getOutput(0), ElementWiseOperation::kSUM);
    
    IConvolutionLayer* id_1743 = network->addConvolution(*id_1610->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage4.0.fuse_layers.1.3.0.weight"], emptywts);
    assert(id_1743);
    id_1743->setStride(DimsHW{ 1, 1 });
    id_1743->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1744 = addBatchNorm2d(network, weightMap, *id_1743->getOutput(0), "stage4.0.fuse_layers.1.3.1", 1e-5);
    ILayer* id_1773 = netAddUpsample(network, id_1744->getOutput(0), 36, 4);
    IElementWiseLayer* id_1774 = network->addElementWise(*id_1773->getOutput(0), *id_1742->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1775 = network->addActivation(*id_1774->getOutput(0), ActivationType::kRELU);

    // conv(1568)+conv(1582)+1596+up(1610)
    IConvolutionLayer* id_1776 = network->addConvolution(*id_1568->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.2.0.0.0.weight"], emptywts);
    assert(id_1776);
    id_1776->setStride(DimsHW{ 2, 2 });
    id_1776->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1777 = addBatchNorm2d(network, weightMap, *id_1776->getOutput(0), "stage4.0.fuse_layers.2.0.0.1", 1e-5);
    IActivationLayer* id_1778 = network->addActivation(*id_1777->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* id_1779 = network->addConvolution(*id_1778->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.2.0.1.0.weight"], emptywts);
    assert(id_1779);
    id_1779->setStride(DimsHW{ 2, 2 });
    id_1779->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1780 = addBatchNorm2d(network, weightMap, *id_1779->getOutput(0), "stage4.0.fuse_layers.2.0.1.1", 1e-5);

    IConvolutionLayer* id_1781 = network->addConvolution(*id_1582->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.2.1.0.0.weight"], emptywts);
    assert(id_1781);
    id_1781->setStride(DimsHW{ 2, 2 });
    id_1781->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1782 = addBatchNorm2d(network, weightMap, *id_1781->getOutput(0), "stage4.0.fuse_layers.2.1.0.1", 1e-5);

    IElementWiseLayer* id_1783 = network->addElementWise(*id_1780->getOutput(0), *id_1782->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_1784 = network->addElementWise(*id_1783->getOutput(0), *id_1596->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1785 = network->addConvolution(*id_1610->getOutput(0), 72, DimsHW{ 1, 1 }, weightMap["stage4.0.fuse_layers.2.3.0.weight"], emptywts);
    assert(id_1785);
    id_1785->setStride(DimsHW{ 1, 1 });
    id_1785->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1786 = addBatchNorm2d(network, weightMap, *id_1785->getOutput(0), "stage4.0.fuse_layers.2.3.1", 1e-5);
    ILayer* id_1815 = netAddUpsample(network, id_1786->getOutput(0), 72, 2);

    IElementWiseLayer* id_1816 = network->addElementWise(*id_1784->getOutput(0), *id_1815->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1817 = network->addActivation(*id_1816->getOutput(0), ActivationType::kRELU);

    // conv(1568)+conv(1582)+conv(1596)+(1610)
    // 1568(cbr)1820(cbr)1823(cb)1825
    IConvolutionLayer* id_1818 = network->addConvolution(*id_1568->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.3.0.0.0.weight"], emptywts);
    assert(id_1818);
    id_1818->setStride(DimsHW{ 2, 2 });
    id_1818->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1819 = addBatchNorm2d(network, weightMap, *id_1818->getOutput(0), "stage4.0.fuse_layers.3.0.0.1", 1e-5);
    IActivationLayer* id_1820 = network->addActivation(*id_1819->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* id_1821 = network->addConvolution(*id_1820->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.3.0.1.0.weight"], emptywts);
    assert(id_1821);
    id_1821->setStride(DimsHW{ 2, 2 });
    id_1821->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1822 = addBatchNorm2d(network, weightMap, *id_1821->getOutput(0), "stage4.0.fuse_layers.3.0.1.1", 1e-5);
    IActivationLayer* id_1823 = network->addActivation(*id_1822->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* id_1824 = network->addConvolution(*id_1823->getOutput(0), 144, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.3.0.2.0.weight"], emptywts);
    assert(id_1824);
    id_1824->setStride(DimsHW{ 2, 2 });
    id_1824->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1825 = addBatchNorm2d(network, weightMap, *id_1824->getOutput(0), "stage4.0.fuse_layers.3.0.2.1", 1e-5);

    // 1582(cbr)1828(cb)1830
    IConvolutionLayer* id_1826 = network->addConvolution(*id_1582->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.3.1.0.0.weight"], emptywts);
    assert(id_1826);
    id_1826->setStride(DimsHW{ 2, 2 });
    id_1826->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1827 = addBatchNorm2d(network, weightMap, *id_1826->getOutput(0), "stage4.0.fuse_layers.3.1.0.1", 1e-5);
    IActivationLayer* id_1828 = network->addActivation(*id_1827->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* id_1829 = network->addConvolution(*id_1828->getOutput(0), 144, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.3.1.1.0.weight"], emptywts);
    assert(id_1829);
    id_1829->setStride(DimsHW{ 2, 2 });
    id_1829->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1830 = addBatchNorm2d(network, weightMap, *id_1829->getOutput(0), "stage4.0.fuse_layers.3.1.1.1", 1e-5);

    IElementWiseLayer* id_1831 = network->addElementWise(*id_1830->getOutput(0), *id_1825->getOutput(0), ElementWiseOperation::kSUM);

    // 1596(cb)1832
    IConvolutionLayer* id_1832 = network->addConvolution(*id_1596->getOutput(0), 144, DimsHW{ 3, 3 }, weightMap["stage4.0.fuse_layers.3.2.0.0.weight"], emptywts);
    assert(id_1832);
    id_1832->setStride(DimsHW{ 2, 2 });
    id_1832->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1833 = addBatchNorm2d(network, weightMap, *id_1832->getOutput(0), "stage4.0.fuse_layers.3.2.0.1", 1e-5);

    IElementWiseLayer* id_1834 = network->addElementWise(*id_1833->getOutput(0), *id_1831->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_1835 = network->addElementWise(*id_1834->getOutput(0), *id_1610->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1836 = network->addActivation(*id_1835->getOutput(0), ActivationType::kRELU);
    //
    auto id_1843 = liteResBlock(network, weightMap, *id_1707->getOutput(0), 18, "stage4.1.branches.0.0");
    auto id_1850 = liteResBlock(network, weightMap, *id_1843->getOutput(0), 18, "stage4.1.branches.0.1");
    auto id_1857 = liteResBlock(network, weightMap, *id_1775->getOutput(0), 36, "stage4.1.branches.1.0");
    auto id_1864 = liteResBlock(network, weightMap, *id_1857->getOutput(0), 36, "stage4.1.branches.1.1");
    auto id_1871 = liteResBlock(network, weightMap, *id_1817->getOutput(0), 72, "stage4.1.branches.2.0");
    auto id_1878 = liteResBlock(network, weightMap, *id_1871->getOutput(0), 72, "stage4.1.branches.2.1");
    auto id_1885 = liteResBlock(network, weightMap, *id_1836->getOutput(0), 144, "stage4.1.branches.3.0");
    auto id_1892 = liteResBlock(network, weightMap, *id_1885->getOutput(0), 144, "stage4.1.branches.3.1");
    
    // 四个分支的密集连接
    // 1850+up1864+up1878+up1892
    IConvolutionLayer* id_1893 = network->addConvolution(*id_1864->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage4.1.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1893);
    id_1893->setStride(DimsHW{ 1, 1 });
    id_1893->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1894 = addBatchNorm2d(network, weightMap, *id_1893->getOutput(0), "stage4.1.fuse_layers.0.1.1", 1e-5);
    ILayer* id_1923 = netAddUpsample(network, id_1894->getOutput(0), 18, 2);
    IElementWiseLayer* id_1924 = network->addElementWise(*id_1850->getOutput(0), *id_1923->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1925 = network->addConvolution(*id_1878->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage4.1.fuse_layers.0.2.0.weight"], emptywts);
    assert(id_1925);
    id_1925->setStride(DimsHW{ 1, 1 });
    id_1925->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1926 = addBatchNorm2d(network, weightMap, *id_1925->getOutput(0), "stage4.1.fuse_layers.0.2.1", 1e-5);
    ILayer* id_1955 = netAddUpsample(network, id_1926->getOutput(0), 18, 4);
    IElementWiseLayer* id_1956 = network->addElementWise(*id_1924->getOutput(0), *id_1955->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1957 = network->addConvolution(*id_1892->getOutput(0), 18, DimsHW{ 1, 1 }, weightMap["stage4.1.fuse_layers.0.3.0.weight"], emptywts);
    assert(id_1957);
    id_1957->setStride(DimsHW{ 1, 1 });
    id_1957->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1958 = addBatchNorm2d(network, weightMap, *id_1957->getOutput(0), "stage4.1.fuse_layers.0.3.1", 1e-5);
    ILayer* id_1987 = netAddUpsample(network, id_1958->getOutput(0), 18, 8);
    IElementWiseLayer* id_1988 = network->addElementWise(*id_1956->getOutput(0), *id_1987->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_1989 = network->addActivation(*id_1988->getOutput(0), ActivationType::kRELU);

    // conv1850+1864+up1878+up1892
    IConvolutionLayer* id_1990 = network->addConvolution(*id_1850->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.1.0.0.0.weight"], emptywts);
    assert(id_1990);
    id_1990->setStride(DimsHW{ 2, 2 });
    id_1990->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_1991 = addBatchNorm2d(network, weightMap, *id_1990->getOutput(0), "stage4.1.fuse_layers.1.0.0.1", 1e-5);
    IElementWiseLayer* id_1992 = network->addElementWise(*id_1991->getOutput(0), *id_1864->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_1993 = network->addConvolution(*id_1878->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage4.1.fuse_layers.1.2.0.weight"], emptywts);
    assert(id_1993);
    id_1993->setStride(DimsHW{ 1, 1 });
    id_1993->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_1994 = addBatchNorm2d(network, weightMap, *id_1993->getOutput(0), "stage4.1.fuse_layers.1.2.1", 1e-5);
    ILayer* id_2023 = netAddUpsample(network, id_1994->getOutput(0), 36, 2);
    IElementWiseLayer* id_2024 = network->addElementWise(*id_1992->getOutput(0), *id_2023->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_2025 = network->addConvolution(*id_1892->getOutput(0), 36, DimsHW{ 1, 1 }, weightMap["stage4.1.fuse_layers.1.3.0.weight"], emptywts);
    assert(id_2025);
    id_2025->setStride(DimsHW{ 1, 1 });
    id_2025->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_2026 = addBatchNorm2d(network, weightMap, *id_2025->getOutput(0), "stage4.1.fuse_layers.1.3.1", 1e-5);
    ILayer* id_2055 = netAddUpsample(network, id_2026->getOutput(0), 36, 4);
    IElementWiseLayer* id_2056 = network->addElementWise(*id_2024->getOutput(0), *id_2055->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_2057 = network->addActivation(*id_2056->getOutput(0), ActivationType::kRELU);

    //conv1850 + conv 1864 + 1878 + up1892
    IConvolutionLayer* id_2058 = network->addConvolution(*id_1850->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.2.0.0.0.weight"], emptywts);
    assert(id_2058);
    id_2058->setStride(DimsHW{ 2, 2 });
    id_2058->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2059 = addBatchNorm2d(network, weightMap, *id_2058->getOutput(0), "stage4.1.fuse_layers.2.0.0.1", 1e-5);
    IActivationLayer* id_2060 = network->addActivation(*id_2059->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* id_2061 = network->addConvolution(*id_2060->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.2.0.1.0.weight"], emptywts);
    assert(id_2061);
    id_2061->setStride(DimsHW{ 2, 2 });
    id_2061->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2062 = addBatchNorm2d(network, weightMap, *id_2061->getOutput(0), "stage4.1.fuse_layers.2.0.1.1", 1e-5);

    IConvolutionLayer* id_2063 = network->addConvolution(*id_1864->getOutput(0), 72, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.2.1.0.0.weight"], emptywts);
    assert(id_2063);
    id_2063->setStride(DimsHW{ 2, 2 });
    id_2063->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2064 = addBatchNorm2d(network, weightMap, *id_2063->getOutput(0), "stage4.1.fuse_layers.2.1.0.1", 1e-5);

    IElementWiseLayer* id_2065 = network->addElementWise(*id_2062->getOutput(0), *id_2064->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_2066 = network->addElementWise(*id_1878->getOutput(0), *id_2065->getOutput(0), ElementWiseOperation::kSUM);

    IConvolutionLayer* id_2067 = network->addConvolution(*id_1892->getOutput(0), 72, DimsHW{ 1, 1 }, weightMap["stage4.1.fuse_layers.2.3.0.weight"], emptywts);
    assert(id_2067);
    id_2067->setStride(DimsHW{ 1, 1 });
    id_2067->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* id_2068 = addBatchNorm2d(network, weightMap, *id_2067->getOutput(0), "stage4.1.fuse_layers.2.3.1", 1e-5);
    ILayer* id_2097 = netAddUpsample(network, id_2068->getOutput(0), 72, 2);

    IElementWiseLayer* id_2098 = network->addElementWise(*id_2097->getOutput(0), *id_2066->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_2099 = network->addActivation(*id_2098->getOutput(0), ActivationType::kRELU);

    // conv1850+conv1864+conv1878+1892
    IConvolutionLayer* id_2100 = network->addConvolution(*id_1850->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.3.0.0.0.weight"], emptywts);
    assert(id_2100);
    id_2100->setStride(DimsHW{ 2, 2 });
    id_2100->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2101 = addBatchNorm2d(network, weightMap, *id_2100->getOutput(0), "stage4.1.fuse_layers.3.0.0.1", 1e-5);
    IActivationLayer* id_2102 = network->addActivation(*id_2101->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* id_2103 = network->addConvolution(*id_2102->getOutput(0), 18, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.3.0.1.0.weight"], emptywts);
    assert(id_2103);
    id_2103->setStride(DimsHW{ 2, 2 });
    id_2103->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2104 = addBatchNorm2d(network, weightMap, *id_2103->getOutput(0), "stage4.1.fuse_layers.3.0.1.1", 1e-5);
    IActivationLayer* id_2105 = network->addActivation(*id_2104->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* id_2106 = network->addConvolution(*id_2105->getOutput(0), 144, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.3.0.2.0.weight"], emptywts);
    assert(id_2106);
    id_2106->setStride(DimsHW{ 2, 2 });
    id_2106->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2107 = addBatchNorm2d(network, weightMap, *id_2106->getOutput(0), "stage4.1.fuse_layers.3.0.2.1", 1e-5);

    // 
    IConvolutionLayer* id_2108 = network->addConvolution(*id_1864->getOutput(0), 36, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.3.1.0.0.weight"], emptywts);
    assert(id_2108);
    id_2108->setStride(DimsHW{ 2, 2 });
    id_2108->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2109 = addBatchNorm2d(network, weightMap, *id_2108->getOutput(0), "stage4.1.fuse_layers.3.1.0.1", 1e-5);
    IActivationLayer* id_2110 = network->addActivation(*id_2109->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* id_2111 = network->addConvolution(*id_2110->getOutput(0), 144, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.3.1.1.0.weight"], emptywts);
    assert(id_2111);
    id_2111->setStride(DimsHW{ 2, 2 });
    id_2111->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2112 = addBatchNorm2d(network, weightMap, *id_2111->getOutput(0), "stage4.1.fuse_layers.3.1.1.1", 1e-5);

    IElementWiseLayer* id_2113 = network->addElementWise(*id_2107->getOutput(0), *id_2112->getOutput(0), ElementWiseOperation::kSUM);

    // 
    IConvolutionLayer* id_2114 = network->addConvolution(*id_1878->getOutput(0), 144, DimsHW{ 3, 3 }, weightMap["stage4.1.fuse_layers.3.2.0.0.weight"], emptywts);
    assert(id_2114);
    id_2114->setStride(DimsHW{ 2, 2 });
    id_2114->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* id_2115 = addBatchNorm2d(network, weightMap, *id_2114->getOutput(0), "stage4.1.fuse_layers.3.2.0.1", 1e-5);

    IElementWiseLayer* id_2116 = network->addElementWise(*id_2113->getOutput(0), *id_2115->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* id_2117 = network->addElementWise(*id_2116->getOutput(0), *id_1892->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* id_2118 = network->addActivation(*id_2117->getOutput(0), ActivationType::kRELU);

    //res
    auto id_2174 = ResBlock2Conv(network, weightMap, *id_2118->getOutput(0), 256, 1024, 1, "incre_modules.3.0");
    auto id_2158 = ResBlock2Conv(network, weightMap, *id_2099->getOutput(0), 128, 512, 1, "incre_modules.2.0");
    auto id_2142 = ResBlock2Conv(network, weightMap, *id_2057->getOutput(0), 64, 256, 1, "incre_modules.1.0");
    auto id_2130 = ResBlock2Conv(network, weightMap, *id_1989->getOutput(0), 32, 128, 1, "incre_modules.0.0");

    auto id_2145 = convBnLeaky(network, weightMap, *id_2130->getOutput(0), 256, 3, 2, 1, "downsamp_modules.0.0", "downsamp_modules.0.1", true);
    IElementWiseLayer* id_2146 = network->addElementWise(*id_2145->getOutput(0), *id_2142->getOutput(0), ElementWiseOperation::kSUM);
    auto id_2161= convBnLeaky(network, weightMap, *id_2146->getOutput(0), 512, 3, 2, 1, "downsamp_modules.1.0", "downsamp_modules.1.1", true);
    IElementWiseLayer* id_2162 = network->addElementWise(*id_2161->getOutput(0), *id_2158->getOutput(0), ElementWiseOperation::kSUM);
    auto id_2177 = convBnLeaky(network, weightMap, *id_2162->getOutput(0), 1024, 3, 2, 1, "downsamp_modules.2.0", "downsamp_modules.2.1", true);
    IElementWiseLayer* id_2178 = network->addElementWise(*id_2177->getOutput(0), *id_2174->getOutput(0), ElementWiseOperation::kSUM);

    auto id_2181 = convBnLeaky(network, weightMap, *id_2178->getOutput(0), 2048, 1, 1, 0, "final_layer.0", "final_layer.1", true);
    //   y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
    auto pool = network->addPoolingNd(*id_2181->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7, 7 });
    pool->setPaddingNd(DimsHW{ 0, 0 });
    pool->setStrideNd(DimsHW{ 1, 1 });
    // self.classifier = nn.Linear(2048, 1000)
    IFullyConnectedLayer* out = network->addFullyConnected(*pool->getOutput(0), 1000, weightMap["classifier.weight"], weightMap["classifier.bias"]);
    assert(out);
    out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*out->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize((1 << 30));  // 1G
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

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
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = "hrnet.engine";
    //engine_name = "E:\\LearningCodes\\GithubRepo\\tensorrtx\\yolov5\\build\\yolov5s.wts";
    argv[1] = "-d";
    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (std::string(argv[1]) == "-d")
    {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }

    std::vector<std::string> file_names;
    file_names.push_back("E:\\Datasets\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\val_41.JPEG");
    //if (read_files_in_dir(argv[2], file_names) < 0) {
    //    std::cout << "read_files_in_dir failed." << std::endl;
    //    return -1;
    //}

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    /*
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp_image = ((resized_img/255. - mean) / std).astype(np.float32)
    */
    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(file_names[f - fcount + 1 + b]); // BGR
            if (img.empty()) continue;
            // cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
            cv::Mat pr_img;
            cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[b * 3 * INPUT_H * INPUT_W + i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229; // R-0.485
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
        // Run inference  
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        float maxp = 0;
        int index = 0;
        for (int b = 0; b < fcount; b++) {
            for (int j = 0; j<1000; ++j)
            {
                float p = prob[b * OUTPUT_SIZE + j];
                if (p > maxp)
                {
                    maxp = p;
                    index = j;
                }
            }
        }
        std::cout << "out index: " << index << std::endl;
    }
}