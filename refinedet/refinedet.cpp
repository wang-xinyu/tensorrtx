#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "logging.h"
#include "calibrator.h"
#include "configure.h"

#include <torch/script.h> // One-stop header.
#include "torch/torch.h"
#include "torch/jit.h"

using namespace nvinfer1;
static Logger gLogger;

//对矩形区域进行修正，防止图片越界
void RoiCorrect(const cv::Mat &m, cv::Rect &r)
{
    if (r.x < 0) r.x = 0;
    if (r.y < 0) r.y = 0;

    if(r.x >= m.cols-1) r.x=0;
    if(r.y >= m.rows-1) r.y=0;

    if(r.width <= 0) r.width = 1;
    if(r.height <= 0) r.height = 1;

    if(r.x + r.width > m.cols - 1) r.width = m.cols - 1 - r.x;
    if(r.y + r.height > m.rows - 1) r.height = m.rows - 1 - r.y;
}

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
        Weights wt{DataType::kFLOAT, nullptr, 0};
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

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
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

//convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
ILayer* convRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p,\
        int linx, const std::string pre_name = "vgg.", bool b_dilate = false) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    if (weightMap.count(pre_name + std::to_string(linx) + ".weight") == 0)
        std::cout << "no key: " <<pre_name + std::to_string(linx) + ".weight" << std::endl;

    if (weightMap.count(pre_name + std::to_string(linx) + ".bias") == 0)
        std::cout << "no key: " <<pre_name + std::to_string(linx) + ".bias" << std::endl;

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[pre_name + std::to_string(linx) + ".weight"], weightMap[pre_name + std::to_string(linx) + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    if(true == b_dilate)
    {
       conv1->setDilation(DimsHW{3, 3});
    }

    auto lr = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

    return lr;
}

//convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
ILayer* convRelu_extras(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, const std::string weight_name, const std::string bias_name){

    if (weightMap.count(weight_name) == 0)
        std::cout << "no key: " <<weight_name << std::endl;

    if (weightMap.count(bias_name) == 0)
        std::cout << "no key: " <<bias_name << std::endl;

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[weight_name], weightMap[bias_name]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    auto lr = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

    return lr;
}

//convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
IConvolutionLayer* convReluconv_tcb0(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int indx_0, int indx_1){

    std::string name_w0 = "tcb0." + (std::string)std::to_string(indx_0) + ".weight";
    std::string name_b0 = "tcb0." + (std::string)std::to_string(indx_0) + ".bias";

    std::string name_w1 = "tcb0." + (std::string)std::to_string(indx_1) + ".weight";
    std::string name_b1 = "tcb0." + (std::string)std::to_string(indx_1) + ".bias";

    if (weightMap.count(name_w0) == 0)
        std::cout << "no key: " <<name_w0 << std::endl;
    if (weightMap.count(name_b0) == 0)
        std::cout << "no key: " <<name_b0 << std::endl;
    if (weightMap.count(name_w1) == 0)
        std::cout << "no key: " <<name_w1 << std::endl;
    if (weightMap.count(name_b1) == 0)
        std::cout << "no key: " <<name_b1 << std::endl;

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[name_w0], weightMap[name_b0]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    auto lr = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*lr->getOutput(0), 256, DimsHW{3, 3}, weightMap[name_w1], weightMap[name_b1]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});

    return conv2;
}

ILayer* ReluconvRelu_tcb2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int indx_0){
    auto lr = network->addActivation(input, ActivationType::kRELU);

    std::string name_w0 = "tcb2." + (std::string)std::to_string(indx_0) + ".weight";
    std::string name_b0 = "tcb2." + (std::string)std::to_string(indx_0) + ".bias";

    if (weightMap.count(name_w0) == 0)
        std::cout << "no key: " <<name_w0 << std::endl;

    if (weightMap.count(name_b0) == 0)
        std::cout << "no key: " <<name_b0 << std::endl;

    IConvolutionLayer* conv1 = network->addConvolutionNd(*lr->getOutput(0), outch, DimsHW{ksize, ksize}, weightMap[name_w0], weightMap[name_b0]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    auto lr1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    return lr1;
}

ILayer* conv_permutation(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, const std::string weight_name, const std::string bias_name)
{
    if (weightMap.count(weight_name) == 0)
        std::cout << "no key: " <<weight_name << std::endl;
    if (weightMap.count(bias_name) == 0)
        std::cout << "no key: " <<bias_name << std::endl;
    IConvolutionLayer* a0 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[weight_name], weightMap[bias_name]);
    assert(a0);
    a0->setStrideNd(DimsHW{s, s});
    a0->setPaddingNd(DimsHW{p, p});

    auto sfl = network->addShuffle(*a0->getOutput(0));
    sfl->setFirstTranspose(Permutation{1, 2, 0});

    return sfl;
}

ILayer* cat_4_tensor(INetworkDefinition *network, ILayer*tensor_0, ILayer*tensor_1, ILayer*tensor_2, ILayer*tensor_3)
{
    Dims dim_;
    dim_.nbDims=1;
    dim_.d[0]=-1;
    //40 40 12 --->>40*40*12
    auto arm_loc_00 = network->addShuffle(*tensor_0->getOutput(0));
    assert(arm_loc_00);
    arm_loc_00->setReshapeDimensions(dim_);

    //20 20 12 --->>20*20*12
    auto arm_loc_11 = network->addShuffle(*tensor_1->getOutput(0));
    assert(arm_loc_11);
    arm_loc_11->setReshapeDimensions(dim_);  //Dims2(-1, 1)

    //10 10 12 --->>10*10*12
    auto arm_loc_22 = network->addShuffle(*tensor_2->getOutput(0));
    assert(arm_loc_22);
    arm_loc_22->setReshapeDimensions(dim_);

    //5 5 12 --->>5*5*12
    auto arm_loc_33 = network->addShuffle(*tensor_3->getOutput(0));
    assert(arm_loc_33);
    arm_loc_33->setReshapeDimensions(dim_);

//
//    Dims dim0 = arm_loc_00->getOutput(0)->getDimensions();
//    std::cout <<"debug  arm_loc_0 dim==" << dim0.d[0] << " " << dim0.d[1] << " " << dim0.d[2] << " " << dim0.d[3] << std::endl;
//    Dims dim1 = arm_loc_11->getOutput(0)->getDimensions();
//    std::cout <<"debug  arm_loc_1 dim==" << dim1.d[0] << " " << dim1.d[1] << " " << dim1.d[2] << " " << dim1.d[3] << std::endl;
//    Dims dim2 = arm_loc_22->getOutput(0)->getDimensions();
//    std::cout <<"debug  arm_loc_2 dim==" << dim2.d[0] << " " << dim2.d[1] << " " << dim2.d[2] << " " << dim2.d[3] << std::endl;
//    Dims dim3 = arm_loc_33->getOutput(0)->getDimensions();
//    std::cout <<"debug  arm_loc_3 dim==" << dim3.d[0] << " " << dim3.d[1] << " " << dim3.d[2] << " " << dim3.d[3] << std::endl;

    ITensor* arm_loc_t[] = {arm_loc_00->getOutput(0), arm_loc_11->getOutput(0), arm_loc_22->getOutput(0), arm_loc_33->getOutput(0)};
    auto arm_loc = network->addConcatenation(arm_loc_t, 4);
    //[25500]
    return arm_loc;
}


ILayer* reshapeSoftmax(INetworkDefinition *network, ITensor& input, int ch) {
    //输入进来是一维的[12750]
    //先变成[XX,ch]
    auto re1 = network->addShuffle(input);
    assert(re1);
    re1->setReshapeDimensions(Dims3(1, -1, ch)); //[1,6375,2];
//     re1->setReshapeDimensions(Dims2(-1, ch)); //[6375,2];

    Dims dim0 = re1->getOutput(0)->getDimensions();
    std::cout <<"debug  re1 dim==" << dim0.d[0] << " " << dim0.d[1] << " " << dim0.d[2] << " " << dim0.d[3] << std::endl;

//    return re1;/////////////////////////////////////////

    auto sm = network->addSoftMax(*re1->getOutput(0));
    sm->setAxes(1<<2);
    assert(sm);
    //再变成一维的,保持和传进来的形状一样
    Dims dim_;
    dim_.nbDims=1;
    dim_.d[0]=-1;
    auto re2 = network->addShuffle(*sm->getOutput(0));
    assert(re2);
    re2->setReshapeDimensions(dim_);

    return re2;
}

IScaleLayer* L2norm(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string pre_name = "conv4_3_L2Norm.weight")
{
    //aa = x.pow(2)  ## [1,512,40,40]
    const static float pval1[3]{0.0, 1.0, 2.0};
    Weights wshift1{DataType::kFLOAT, pval1, 1};
    Weights wscale1{DataType::kFLOAT, pval1+1, 1};
    Weights wpower1{DataType::kFLOAT, pval1+2, 1};
    IScaleLayer* scale1 = network->addScale(
            input,
            ScaleMode::kUNIFORM,
            wshift1,
            wscale1,
            wpower1);
    assert(scale1);

   //bb =  x.pow(2).sum(dim=1, keepdim=True)  ## [1,1,40,40]
    IReduceLayer* reduce1 = network->addReduce(*scale1->getOutput(0),
                                               ReduceOperation::kSUM,
                                               1,
                                               true);
    assert(reduce1);

    //norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps  # [1,1,40,40]
    const static float pval2[3]{0.0, 1.0, 0.5};
    Weights wshift2{DataType::kFLOAT, pval2, 1};
    Weights wscale2{DataType::kFLOAT, pval2+1, 1};
    Weights wpower2{DataType::kFLOAT, pval2+2, 1};
    IScaleLayer* scale2 = network->addScale(
            *reduce1->getOutput(0),
            ScaleMode::kUNIFORM,
            wshift2,
            wscale2,
            wpower2);
    assert(scale2);

    // x = torch.div(x,norm)
    IElementWiseLayer* ew2 = network->addElementWise(input,
                                                     *scale2->getOutput(0),
                                                     ElementWiseOperation::kDIV);
    assert(ew2);

    //out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
    int len = weightMap[pre_name].count;
    float* pval3 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    std::fill_n(pval3, len, 1.0);
    Weights wpower3{DataType::kFLOAT, pval3, len};
    weightMap[pre_name + ".power3"] = wpower3;

    float* pval4 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    std::fill_n(pval4, len, 0.0);
    Weights wpower4{DataType::kFLOAT, pval4, len};
    weightMap[pre_name + ".power4"] = wpower4;

    IScaleLayer* scale3 = network->addScale(
            *ew2->getOutput(0),
            ScaleMode::kCHANNEL,
            wpower4,
            weightMap[pre_name],
            wpower3);
    assert(scale3);
    return scale3;
}


//convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(path_wts);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    DimsHW maxpool_hw = DimsHW(2,2);

    auto lr0 = convRelu(network, weightMap, *data, 64, 3, 1, 1, 0);
    auto lr1 = convRelu(network, weightMap, *lr0->getOutput(0), 64, 3, 1, 1, 2);
    IPoolingLayer* pool1 = network->addPoolingNd(*lr1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    auto lr2 = convRelu(network, weightMap, *pool1->getOutput(0), 128, 3, 1, 1, 5);
    auto lr3 = convRelu(network, weightMap, *lr2->getOutput(0), 128, 3, 1, 1, 7);
    IPoolingLayer* pool2 = network->addPoolingNd(*lr3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});

    auto lr4 = convRelu(network, weightMap, *pool2->getOutput(0), 256, 3, 1, 1, 10);
    auto lr5 = convRelu(network, weightMap, *lr4->getOutput(0), 256, 3, 1, 1, 12);
    auto lr6 = convRelu(network, weightMap, *lr5->getOutput(0), 256, 3, 1, 1, 14);
    IPoolingLayer* pool3 = network->addPoolingNd(*lr6->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool3);
    pool3->setStrideNd(DimsHW{2, 2});

    auto lr7 = convRelu(network, weightMap, *pool3->getOutput(0), 512, 3, 1, 1, 17);
    auto lr8 = convRelu(network, weightMap, *lr7->getOutput(0), 512, 3, 1, 1, 19);
    auto lr9 = convRelu(network, weightMap, *lr8->getOutput(0), 512, 3, 1, 1, 21);
    IPoolingLayer* pool4 = network->addPoolingNd(*lr9->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool4);
    pool4->setStrideNd(DimsHW{2, 2});

    auto lr24 = convRelu(network, weightMap, *pool4->getOutput(0), 512, 3, 1, 1, 24);
    auto lr26 = convRelu(network, weightMap, *lr24->getOutput(0), 512, 3, 1, 1, 26);
    auto lr28 = convRelu(network, weightMap, *lr26->getOutput(0), 512, 3, 1, 1, 28);
    IPoolingLayer* pool5 = network->addPoolingNd(*lr28->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool5);
    pool5->setStrideNd(DimsHW{2, 2});

    auto lr31 = convRelu(network, weightMap, *pool5->getOutput(0), 1024, 3, 1, 3, 31,"vgg.",true);

    //s_0
    auto out_conv4_3_L2Norm = L2norm(network, weightMap, *lr9->getOutput(0),"conv4_3_L2Norm.weight");
    //s_1
    auto out_conv5_3_L2Norm = L2norm(network, weightMap, *lr28->getOutput(0),"conv5_3_L2Norm.weight");

    //s_2
    auto lr33 = convRelu(network, weightMap, *lr31->getOutput(0), 1024, 1, 1, 0, 33);

    auto extras0 = convRelu_extras(network, weightMap, *lr33->getOutput(0), 256, 1, 1, 0, "extras.0.weight", "extras.0.bias");
    //s_3
    auto extras1 = convRelu_extras(network, weightMap, *extras0->getOutput(0), 512, 3, 2, 1, "extras.1.weight", "extras.1.bias");

    auto arm_loc_0 = conv_permutation(network, weightMap, *out_conv4_3_L2Norm->getOutput(0), 12, 3, 1, 1, "arm_loc.0.weight", "arm_loc.0.bias");
    auto arm_loc_1 = conv_permutation(network, weightMap, *out_conv5_3_L2Norm->getOutput(0), 12, 3, 1, 1, "arm_loc.1.weight", "arm_loc.1.bias");
    auto arm_loc_2 = conv_permutation(network, weightMap, *lr33->getOutput(0), 12, 3, 1, 1, "arm_loc.2.weight", "arm_loc.2.bias");
    auto arm_loc_3 = conv_permutation(network, weightMap, *extras1->getOutput(0), 12, 3, 1, 1, "arm_loc.3.weight", "arm_loc.3.bias");

    auto arm_conf_0 = conv_permutation(network, weightMap, *out_conv4_3_L2Norm->getOutput(0), 6, 3, 1, 1, "arm_conf.0.weight", "arm_conf.0.bias");
    auto arm_conf_1 = conv_permutation(network, weightMap, *out_conv5_3_L2Norm->getOutput(0), 6, 3, 1, 1, "arm_conf.1.weight", "arm_conf.1.bias");
    auto arm_conf_2 = conv_permutation(network, weightMap, *lr33->getOutput(0), 6, 3, 1, 1, "arm_conf.2.weight", "arm_conf.2.bias");
    auto arm_conf_3 = conv_permutation(network, weightMap, *extras1->getOutput(0), 6, 3, 1, 1, "arm_conf.3.weight", "arm_conf.3.bias");

    auto arm_loc = cat_4_tensor(network, arm_loc_0, arm_loc_1, arm_loc_2, arm_loc_3);
    auto arm_conf = cat_4_tensor(network, arm_conf_0, arm_conf_1, arm_conf_2, arm_conf_3);

    auto ss_0 = convReluconv_tcb0(network, weightMap, *extras1->getOutput(0),  256, 3, 1, 1, 9, 11);
    auto ss_00 = ReluconvRelu_tcb2(network, weightMap, *ss_0->getOutput(0),  256, 3, 1, 1, 10);
    auto ss_1 = convReluconv_tcb0(network, weightMap, *lr33->getOutput(0),  256, 3, 1, 1, 6, 8);

    IDeconvolutionLayer* tcb1_2 = network->addDeconvolutionNd(*ss_00->getOutput(0), 256, DimsHW{2, 2}, weightMap["tcb1.2.weight"], weightMap["tcb1.2.bias"]);  //nn.ConvTranspose2d(256, 256, 2, 2)
    tcb1_2->setStrideNd(DimsHW{2, 2});
    assert(tcb1_2);
    auto ss_1_add = network->addElementWise(*ss_1->getOutput(0), *tcb1_2->getOutput(0), ElementWiseOperation::kSUM);
    auto ss_11 = ReluconvRelu_tcb2(network, weightMap, *ss_1_add->getOutput(0),  256, 3, 1, 1, 7);

    auto ss_2 = convReluconv_tcb0(network, weightMap, *out_conv5_3_L2Norm->getOutput(0),  256, 3, 1, 1, 3, 5);
    IDeconvolutionLayer* tcb1_1 = network->addDeconvolutionNd(*ss_11->getOutput(0), 256, DimsHW{2, 2}, weightMap["tcb1.1.weight"], weightMap["tcb1.1.bias"]);  //nn.ConvTranspose2d(256, 256, 2, 2)
    tcb1_1->setStrideNd(DimsHW{2, 2});
    assert(tcb1_1);
    auto ss_2_add = network->addElementWise(*ss_2->getOutput(0), *tcb1_1->getOutput(0), ElementWiseOperation::kSUM);
    auto ss_22 = ReluconvRelu_tcb2(network, weightMap, *ss_2_add->getOutput(0),  256, 3, 1, 1, 4);

    auto ss_3 = convReluconv_tcb0(network, weightMap, *out_conv4_3_L2Norm->getOutput(0),  256, 3, 1, 1, 0, 2);
    IDeconvolutionLayer* tcb1_0 = network->addDeconvolutionNd(*ss_22->getOutput(0), 256, DimsHW{2, 2}, weightMap["tcb1.0.weight"], weightMap["tcb1.0.bias"]);  //nn.ConvTranspose2d(256, 256, 2, 2)
    tcb1_0->setStrideNd(DimsHW{2, 2});
    assert(tcb1_0);
    auto ss_3_add = network->addElementWise(*ss_3->getOutput(0), *tcb1_0->getOutput(0), ElementWiseOperation::kSUM);
    auto ss_33 = ReluconvRelu_tcb2(network, weightMap, *ss_3_add->getOutput(0),  256, 3, 1, 1, 1);

    auto odm_loc_0 = conv_permutation(network, weightMap, *ss_33->getOutput(0), 12, 3, 1, 1, "odm_loc.0.weight", "odm_loc.0.bias");
    auto odm_loc_1 = conv_permutation(network, weightMap, *ss_22->getOutput(0), 12, 3, 1, 1, "odm_loc.1.weight", "odm_loc.1.bias");
    auto odm_loc_2 = conv_permutation(network, weightMap, *ss_11->getOutput(0), 12, 3, 1, 1, "odm_loc.2.weight", "odm_loc.2.bias");
    auto odm_loc_3 = conv_permutation(network, weightMap, *ss_00->getOutput(0), 12, 3, 1, 1, "odm_loc.3.weight", "odm_loc.3.bias");

    auto odm_conf_0 = conv_permutation(network, weightMap, *ss_33->getOutput(0), 3 * num_class, 3, 1, 1, "odm_conf.0.weight", "odm_conf.0.bias");
    auto odm_conf_1 = conv_permutation(network, weightMap, *ss_22->getOutput(0), 3 * num_class, 3, 1, 1, "odm_conf.1.weight", "odm_conf.1.bias");
    auto odm_conf_2 = conv_permutation(network, weightMap, *ss_11->getOutput(0), 3 * num_class, 3, 1, 1, "odm_conf.2.weight", "odm_conf.2.bias");
    auto odm_conf_3 = conv_permutation(network, weightMap, *ss_00->getOutput(0), 3 * num_class, 3, 1, 1, "odm_conf.3.weight", "odm_conf.3.bias");

    auto odm_loc = cat_4_tensor(network, odm_loc_0, odm_loc_1, odm_loc_2, odm_loc_3);
    auto odm_conf = cat_4_tensor(network, odm_conf_0, odm_conf_1, odm_conf_2, odm_conf_3);

    //25500
    Dims dim = arm_loc->getOutput(0)->getDimensions();
    std::cout <<"debug  arm_loc dim==" << dim.d[0] << " " << dim.d[1] << " " << dim.d[2] << " " << dim.d[3] << std::endl;
    arm_loc->getOutput(0)->setName(OUTPUT_BLOB_NAME_arm_loc);
    network->markOutput(*arm_loc->getOutput(0));

    auto arm_conf_111 = reshapeSoftmax(network, *arm_conf->getOutput(0), 2);
    //12750
    Dims dim2 = arm_conf_111->getOutput(0)->getDimensions();
    std::cout <<"debug  arm_conf dim==" << dim2.d[0] << " " << dim2.d[1] << " " << dim2.d[2] << " " << dim2.d[3] << std::endl;
    arm_conf_111->getOutput(0)->setName(OUTPUT_BLOB_NAME_arm_conf);
    network->markOutput(*arm_conf_111->getOutput(0));

    //25500
    Dims dim3 = odm_loc->getOutput(0)->getDimensions();
    std::cout <<"debug  odm_loc dim==" << dim3.d[0] << " " << dim3.d[1] << " " << dim3.d[2] << " " << dim3.d[3] << std::endl;
    odm_loc->getOutput(0)->setName(OUTPUT_BLOB_NAME_odm_loc);
    network->markOutput(*odm_loc->getOutput(0));

    //159375
    Dims dim4 = odm_conf->getOutput(0)->getDimensions();
    odm_conf = reshapeSoftmax(network, *odm_conf->getOutput(0), 25);
    std::cout <<"debug  odm_conf dim==" << dim4.d[0] << " " << dim4.d[1] << " " << dim4.d[2] << " " << dim4.d[3] << std::endl;
    odm_conf->getOutput(0)->setName(OUTPUT_BLOB_NAME_odm_conf);
    network->markOutput(*odm_conf->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;
    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

torch::Tensor PriorBox()
{
    std::vector<float> mean;
    std::vector<int> feature_maps = {40,20,10,5};
    int image_size = 320;
    std::vector<int> steps = {8,16,32,64};
    std::vector<int> min_sizes = {32,64,128,256};
    std::vector<int> aspect_ratios = {2,2,2,2};
    for(int k=0;k<feature_maps.size();k++)
    {
        int f = feature_maps[k];
        for(int i=0;i<f;i++)
        {
            for(int j=0;j<f;j++)
            {
                float f_k = image_size * 1.0 / steps[k];
                float cx = (j + 0.5) / f_k;
                float cy = (i + 0.5) / f_k;
                float s_k = min_sizes[k] * 1.0 / image_size;
                mean.push_back(cx);
                mean.push_back(cy);
                mean.push_back(s_k);
                mean.push_back(s_k);

                float ar = aspect_ratios[k];
                mean.push_back(cx);
                mean.push_back(cy);
                mean.push_back(s_k * 1.0 * sqrt(ar));
                mean.push_back(s_k * 1.0 / sqrt(ar));

                mean.push_back(cx);
                mean.push_back(cy);
                mean.push_back(s_k * 1.0 / sqrt(ar));
                mean.push_back(s_k * 1.0 * sqrt(ar));
            }
        }
    }

    torch::Tensor m_prior;
    int m_prior_size = 6375;
    m_prior = torch::from_blob(mean.data(),{m_prior_size,4}).cuda();
    m_prior = m_prior.clamp(0,1);
    //    std::cout<<m_prior<<std::endl;
    return m_prior.toType(torch::kFloat64);
}


torch::Tensor decode(const torch::Tensor _loc,torch::Tensor _prior,bool b_form_pt = false)
{
    std::vector<float> variance({0.1,0.2});
    torch::Tensor top_2 = torch::tensor({0,1}).cuda().to(torch::kLong);
    torch::Tensor bottom_2 = torch::tensor({2,3}).cuda().to(torch::kLong);

    auto c1 = _prior.index_select(1,top_2)+_loc.index_select(1,top_2).mul(variance[0])*_prior.index_select(1,bottom_2);
    auto c2 = _prior.index_select(1,bottom_2)*torch::exp(_loc.index_select(1,bottom_2)*variance[1]);
    auto _retv = torch::cat({c1,c2},1);
    if(b_form_pt)
    {
        auto c3 = _retv.index_select(1,top_2)-_retv.index_select(1,bottom_2).div(2);
        auto c4 = c3 + _retv.index_select(1,bottom_2);
        return torch::cat({c3,c4},1);
    } else
    {
        return _retv;
    }

}

torch::Tensor center(torch::Tensor retv)
{
    auto c1 = retv.select(1,0).unsqueeze(1);
    auto c2 = retv.select(1,1).unsqueeze(1);
    auto c3 = retv.select(1,2).unsqueeze(1);
    auto c4 = retv.select(1,3).unsqueeze(1);

    auto _retv = torch::cat({(c1+c3).div(2),(c2+c4).div(2),c3-c1,c4-c2},1);
    return _retv;
}

bool nms(const torch::Tensor& boxes, const torch::Tensor& scores, torch::Tensor &keep, int &count,float overlap, int top_k)
{
    count =0;
    keep = torch::zeros({scores.size(0)}).to(torch::kLong).to(scores.device());
    if(0 == boxes.numel())
    {
        return false;
    }

    torch::Tensor x1 = boxes.select(1,0).clone();
    torch::Tensor y1 = boxes.select(1,1).clone();
    torch::Tensor x2 = boxes.select(1,2).clone();
    torch::Tensor y2 = boxes.select(1,3).clone();
    torch::Tensor area = (x2-x1)*(y2-y1);
    //    std::cout<<area<<std::endl;

    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(scores.unsqueeze(1), 0, 0);
    torch::Tensor v = std::get<0>(sort_ret).squeeze(1).to(scores.device());
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1).to(scores.device());

    int num_ = idx.size(0);
    if(num_ > top_k) //python:idx = idx[-top_k:]
    {
        idx = idx.slice(0,num_-top_k,num_).clone();
    }
    torch::Tensor xx1,yy1,xx2,yy2,w,h;
    while(idx.numel() > 0)
    {
        auto i = idx[-1];
        keep[count] = i;
        count += 1;
        if(1 == idx.size(0))
        {
            break;
        }
        idx = idx.slice(0,0,idx.size(0)-1).clone();

        xx1 = x1.index_select(0,idx);
        yy1 = y1.index_select(0,idx);
        xx2 = x2.index_select(0,idx);
        yy2 = y2.index_select(0,idx);

        xx1 = xx1.clamp(x1[i].item().toFloat(),INT_MAX*1.0);
        yy1 = yy1.clamp(y1[i].item().toFloat(),INT_MAX*1.0);
        xx2 = xx2.clamp(INT_MIN*1.0,x2[i].item().toFloat());
        yy2 = yy2.clamp(INT_MIN*1.0,y2[i].item().toFloat());

        w = xx2 - xx1;
        h = yy2 - yy1;

        w = w.clamp(0,INT_MAX);
        h = h.clamp(0,INT_MAX);

        torch::Tensor inter = w * h;
        torch::Tensor rem_areas = area.index_select(0,idx);

        torch::Tensor union_ = (rem_areas - inter) + area[i];
        torch::Tensor Iou = inter * 1.0 / union_;
        torch::Tensor index_small = Iou < overlap;
        auto mask_idx = torch::nonzero(index_small).squeeze();
        idx = idx.index_select(0,mask_idx);//pthon: idx = idx[IoU.le(overlap)]
    }
    return true;
}

void doInference(IExecutionContext& context, void* buffers[], cudaStream_t &stream, float* input, std::vector<std::vector<float>> &detections) {
    auto start_infer = std::chrono::system_clock::now();
    detections.clear();
    int batchSize = 1;
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
//    std::cout<<"engine.getNbBindings()==="<<engine.getNbBindings()<<std::endl;
    assert(engine.getNbBindings() == 5);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex_arm_loc = engine.getBindingIndex(OUTPUT_BLOB_NAME_arm_loc);
    const int outputIndex_arm_conf = engine.getBindingIndex(OUTPUT_BLOB_NAME_arm_conf);
    const int outputIndex_odm_loc = engine.getBindingIndex(OUTPUT_BLOB_NAME_odm_loc);
    const int outputIndex_odm_conf = engine.getBindingIndex(OUTPUT_BLOB_NAME_odm_conf);
//    const int outputIndex2 = engine.getBindingIndex("prob2");
//    printf("inputIndex=%d\n",inputIndex);
//    printf("outputIndex_arm_loc=%d\n",outputIndex_arm_loc);
//    printf("outputIndex_arm_conf=%d\n",outputIndex_arm_conf);
//    printf("outputIndex_odm_loc=%d\n",outputIndex_odm_loc);
//    printf("outputIndex_odm_conf=%d\n",outputIndex_odm_conf);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaDeviceSynchronize();
    auto end_infer = std::chrono::system_clock::now();
    double during_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count();
    std::cout <<"time consume context.enqueue===" <<  during_time << "ms" << std::endl;

    auto start_houchuli = std::chrono::system_clock::now();
    int m_prior_size = 6375;
    torch::Tensor m_prior = PriorBox();
    torch::Tensor arm_loc = torch::from_blob(buffers[outputIndex_arm_loc],{m_prior_size,4}).cuda().toType(torch::kFloat64).unsqueeze(0);
    torch::Tensor arm_conf = torch::from_blob(buffers[outputIndex_arm_conf],{m_prior_size,2}).cuda().toType(torch::kFloat64).unsqueeze(0);
    torch::Tensor odm_loc = torch::from_blob(buffers[outputIndex_odm_loc],{m_prior_size,4}).cuda().toType(torch::kFloat64).unsqueeze(0);
    torch::Tensor odm_conf = torch::from_blob(buffers[outputIndex_odm_conf],{m_prior_size,25}).cuda().toType(torch::kFloat64).unsqueeze(0);

    float obj_threshed = 0.01;
    torch::Tensor arm_object_conf = arm_conf.squeeze(0).select(1,1);
    torch::Tensor object_index = arm_object_conf > obj_threshed;
    object_index=object_index.unsqueeze(1);

    torch::Tensor object_index_1 = object_index.expand_as(odm_conf.squeeze(0)).toType(torch::kFloat64);
    auto filter_odm_conf = odm_conf.squeeze(0).toType(torch::kFloat64) * object_index_1;
    torch::Tensor conf_preds_ = filter_odm_conf.clone().toType(torch::kFloat64);
    torch::Tensor conf_preds = conf_preds_.transpose(1,0).toType(torch::kFloat64);
    torch::Tensor default_m = decode(arm_loc[0],m_prior);
//    default_m = center(default_m);
    bool b_form_pt = true;
    torch::Tensor decode_boxes_m = decode(odm_loc[0],default_m,b_form_pt);//6375,4

    float conf_thresh = 0.01;
    float mask_thresh = 0.01;

    torch::Tensor result_out;
    for(int i=1;i<25;i++)
    {
        torch::Tensor c_mask_m = conf_preds[i] > mask_thresh;
        torch::Tensor nonzero_index = torch::nonzero(c_mask_m);
        torch::Tensor  score_m = torch::index_select(conf_preds[i],0,nonzero_index.squeeze(1));
        torch::Tensor  boxes_m = torch::index_select(decode_boxes_m,0,nonzero_index.squeeze(1));

        torch::Tensor keep;
        int count = 0;
        float overlap = 0.45;
        int top_k=1000;
        nms(boxes_m, score_m, keep, count, overlap, top_k);
        if(0 == count) { continue; }

        keep = keep.slice(0,0,count).clone();
        torch::Tensor score_my = score_m.index_select(0,keep);
        torch::Tensor boxes_my = boxes_m.index_select(0,keep);

        if(score_my[0].item().toFloat() < conf_thresh)
        {
            continue;
        }
//        boxes_my.select(1,0).mul_(width);
//        boxes_my.select(1,1).mul_(height);
//        boxes_my.select(1,2).mul_(width);
//        boxes_my.select(1,3).mul_(height);
        torch::Tensor label_tensor = torch::full_like(score_my.unsqueeze(1),i);
        torch::Tensor result_ = torch::cat({boxes_my.toType(torch::kFloat64),score_my.unsqueeze(1).toType(torch::kFloat64),label_tensor.toType(torch::kFloat64)},1);
        if(0 == result_out.numel())
        {
            result_out = result_.clone();
        }else
        {
            result_out = torch::cat({result_out,result_},0);//按行拼接
        }
    }
    if(0 == result_out.numel()) { std::cout<<"libtorch refinedet obj_small: nothing detect!"<<std::endl; return ;}
    result_out =result_out.cpu();

    // x1,y1,x2,y2,score,id
    auto result_data = result_out.accessor<double, 2>();
    for(int i=0;i<result_data.size(0);i++)
    {
        float score = result_data[i][4];
        float x1 = result_data[i][0];
        float y1 = result_data[i][1];
        float x2 = result_data[i][2];
        float y2 = result_data[i][3];
        int id_label = result_data[i][5];

        std::vector<float> v_detections;
        v_detections.push_back(0); //image_id
        v_detections.push_back(id_label); //label
        v_detections.push_back(score); //score
        v_detections.push_back(x1); //xmin
        v_detections.push_back(y1); //ymin
        v_detections.push_back(x2); //xmax
        v_detections.push_back(y2); //ymax
        detections.push_back(v_detections);
    }
    cudaDeviceSynchronize();
    auto end_houchuli = std::chrono::system_clock::now();
    double during_time_houchuli = std::chrono::duration_cast<std::chrono::milliseconds>(end_houchuli - start_houchuli).count();
    std::cout <<"time consume houchuli===" <<  during_time_houchuli << "ms" << std::endl;
}

void base_transform(const cv::Mat &m_src,float *data)
{
    cv::Mat image;
    cv::resize(m_src,image,cv::Size(INPUT_W,INPUT_H));
    if(1 == image.channels()) { cv::cvtColor(image,image,CV_GRAY2BGR); }

    for(int i=0;i<INPUT_H;i++)
    {
        uchar* img_data = image.ptr<uchar>(i); //得到行指针首地址
        for(int j=0;j<INPUT_W;j++)
        {
            int offset = i * INPUT_H + j;
            data[offset] = (float)(img_data[j*3 + 2] * 1.0 - 123.0);
            data[offset + INPUT_H * INPUT_W] = (float)(img_data[j*3 + 1] * 1.0 - 117.0);
            data[offset + 2 * INPUT_H * INPUT_W] = (float)(img_data[j*3 + 0] * 1.0 - 104.0);
        }
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

#ifdef SERIALIZE
    IHostMemory* modelStream{nullptr};
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p(path_save_engine, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;

#elif defined  INFER
    std::ifstream file(path_engine, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

#else
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "configure.h should difine SERIALIZE INFER" << std::endl;
    std::cerr << "please check!" << std::endl;
    return -1;
#endif

    std::vector<std::string> file_names;
    if (read_files_in_dir(p_dir_name, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    float data[3 * INPUT_H * INPUT_W];

    IRuntime* runtime = createInferRuntime(gLogger);     //400M
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size); //777M
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();  //971M
    assert(context != nullptr);
    delete[] trtModelStream;

    const int batchSize = 1;
    const int inputIndex=0;
    const int outputIndex_arm_loc=1;
    const int outputIndex_arm_conf=3;
    const int outputIndex_odm_loc=2;
    const int outputIndex_odm_conf=4;

    //初始化cuda显存 输入和4个输出显存
    void* buffers[5];
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[0], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));

    const int OUTPUT_SIZE_arm_loc = 25500; //这里需要根据自己的大小来
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_arm_loc], batchSize * OUTPUT_SIZE_arm_loc * sizeof(float)));

    const int OUTPUT_SIZE_arm_conf = 12750; //这里需要根据自己的大小来
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_arm_conf], batchSize * OUTPUT_SIZE_arm_conf * sizeof(float)));

    const int OUTPUT_SIZE_odm_loc = 25500; //这里需要根据自己的大小来
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_odm_loc], batchSize * OUTPUT_SIZE_odm_loc * sizeof(float)));

    const int OUTPUT_SIZE_odm_conf = 159375; //这里需要根据自己的大小来
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_odm_conf], batchSize * OUTPUT_SIZE_odm_conf * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


    int fcount = 0;
    auto t_0 = std::chrono::steady_clock::now();
    for (auto f: file_names) {
        fcount++;
        std::cout << "\n" << fcount << "  " << f << std::endl;
        std::cout << std::string(p_dir_name) + "/" + f << std::endl;

        auto start_read = std::chrono::system_clock::now();
        cv::Mat img = cv::imread(std::string(p_dir_name) + "/" + f);
        cudaDeviceSynchronize();
        auto end_read = std::chrono::system_clock::now();
        double during_time_read = std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count();
        std::cout <<"time consume during_time_read===" <<  during_time_read << "ms" << std::endl;

        if (img.empty()) continue;

        auto start_yuchuli = std::chrono::system_clock::now();
        base_transform(img,data);
        cudaDeviceSynchronize();
        auto end_yuchuli = std::chrono::system_clock::now();
        double during_time_yuchuli = std::chrono::duration_cast<std::chrono::milliseconds>(end_yuchuli - start_yuchuli).count();
        std::cout <<"time consume base_transform===" <<  during_time_yuchuli << "ms" << std::endl;

        auto start_doInfer = std::chrono::system_clock::now();
        std::vector<std::vector<float>> detections;
        doInference(*context, buffers, stream, data, detections);
        cudaDeviceSynchronize();
        auto end_doInfer = std::chrono::system_clock::now();
        double during_doinfer = std::chrono::duration_cast<std::chrono::milliseconds>(end_doInfer - start_doInfer).count();
        std::cout <<"time consume doInference===" <<  during_doinfer << "ms" << std::endl;

        /* Print the detection results. */
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const std::vector<float> &d = detections[i];

            CHECK_EQ(d.size(), 7);
            const float score = d[2];

            int label = int(d[1]);
            if (label >= num_class || label < 0)
            {
                std::cout << "label_Error!" << std::endl;
                continue;
            }
            if(score < TH)
            {
                continue;
            }
            cv::Rect r;
            r.x = d[3] * img.cols;
            r.y = d[4] * img.rows;
            r.width = d[5] * img.cols - r.x;
            r.height = d[6] * img.rows - r.y;

            RoiCorrect(img, r);
            if(T_show)
            {
                cv::rectangle(img,r,cv::Scalar(255,0,0),2);
            }
            if (T_show == 0)
            {
                std::string name_1 = f.substr(0,f.size()-4);
                std::string path_txt = save_path_txt + name_1 + ".txt";
                std::ofstream fout(path_txt);
                fout << label_map[label] << " " << score << " " << r.x << " " << r.y << " " << r.x + r.width
                     << " " << r.y + r.height << std::endl; //使用自己的label
            }
        }
        if(T_show)
        {
            cv::namedWindow("show",0);
            cv::imshow("show",img);
            cv::waitKey(0);
        }
    }
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex_arm_loc]));
    CUDA_CHECK(cudaFree(buffers[outputIndex_arm_conf]));

    CUDA_CHECK(cudaFree(buffers[outputIndex_odm_loc]));
    CUDA_CHECK(cudaFree(buffers[outputIndex_odm_conf]));

    cudaDeviceSynchronize();
    auto ttt = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - t_0).count();
    std::cout << "all consume time="<<ttt <<"ms"<<std::endl;
    std::cout << "-----------end-----------------------"<<std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
