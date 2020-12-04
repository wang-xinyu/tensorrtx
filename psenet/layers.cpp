#include "layers.h"

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
    float *gamma = (float *)weightMap[lname + "gamma"].values; // scale
    float *beta = (float *)weightMap[lname + "beta"].values;   // offset
    float *mean = (float *)weightMap[lname + "moving_mean"].values;
    float *var = (float *)weightMap[lname + "moving_variance"].values;
    int len = weightMap[lname + "moving_variance"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer *bottleneck(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int ch, int stride, std::string lname, int branch_type)
{

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1 = network->addConvolution(input, ch, DimsHW{1, 1}, weightMap[lname + "conv1/weights"], emptywts);
    assert(conv1);

    Dims conv1_shape = conv1->getOutput(0)->getDimensions();

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv1/BatchNorm/", 1e-5);
    assert(bn1);

    Dims bn1_shape = bn1->getOutput(0)->getDimensions();

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    Dims relu1_shape = relu1->getOutput(0)->getDimensions();

    IConvolutionLayer *conv2 = network->addConvolution(*relu1->getOutput(0), ch, DimsHW{3, 3}, weightMap[lname + "conv2/weights"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});

    Dims conv2_shape = conv2->getOutput(0)->getDimensions();

    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "conv2/BatchNorm/", 1e-5);
    assert(bn2);

    Dims bn2_shape = bn2->getOutput(0)->getDimensions();

    IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    Dims relu2_shape = relu2->getOutput(0)->getDimensions();

    IConvolutionLayer *conv3 = network->addConvolution(*relu2->getOutput(0), ch * 4, DimsHW{1, 1}, weightMap[lname + "conv3/weights"], emptywts);
    assert(conv3);

    Dims conv3_shape = conv3->getOutput(0)->getDimensions();

    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "conv3/BatchNorm/", 1e-5);
    assert(bn3);
    IElementWiseLayer *ew1;
    Dims ew1_shape;

    // branch_type 0:shortcut,1:conv+bn+shortcut,2:maxpool+shortcut
    if (branch_type == 0)
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
        ew1_shape = ew1->getOutput(0)->getDimensions();
        assert(ew1);
    }
    else if (branch_type == 1)
    {
        IConvolutionLayer *conv4 = network->addConvolution(input, ch * 4, DimsHW{1, 1}, weightMap[lname + "shortcut/weights"], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});
        IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "shortcut/BatchNorm/", 1e-5);
        assert(bn4);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
        ew1_shape = ew1->getOutput(0)->getDimensions();
        assert(ew1);
    }
    else
    {
        IPoolingLayer *pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{1, 1});
        assert(pool);
        pool->setStrideNd(DimsHW{2, 2});
        ew1 = network->addElementWise(*pool->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
        ew1_shape = ew1->getOutput(0)->getDimensions();
        assert(ew1);
    }

    IActivationLayer *relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);

    Dims relu3_shape = relu3->getOutput(0)->getDimensions();

    assert(relu3);
    return relu3;
}

IActivationLayer *ConvRelu(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int kernel, int stride, std::string lname)
{
    IConvolutionLayer *conv = network->addConvolution(input, 256, DimsHW{kernel, kernel}, weightMap[lname + "weights"], weightMap[lname + "biases"]);
    assert(conv);
    conv->setStride(DimsHW{stride, stride});
    if (kernel == 3 || stride == 2)
    {
        conv->setPadding(DimsHW{1, 1});
    }

    IActivationLayer *ac = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    assert(ac);
    return ac;
}