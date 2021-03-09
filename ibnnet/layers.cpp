#include "layers.h"

namespace trtxapi {

    ITensor* MeanStd(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* input, const std::string lname, const float* mean, const float* std, const bool div255) {
        if(div255) {
            Weights Div_225{ DataType::kFLOAT, nullptr, 3 };
            float *wgt = reinterpret_cast<float*>(malloc(sizeof(float) * 3));
            std::fill_n(wgt, 3, 255.0f); 
            Div_225.values = wgt;
            weightMap[lname + ".div"] = Div_225;
            IConstantLayer* d = network->addConstant(Dims3{ 3, 1, 1 }, Div_225);
            input = network->addElementWise(*input, *d->getOutput(0), ElementWiseOperation::kDIV)->getOutput(0);
        }
        Weights Mean{ DataType::kFLOAT, nullptr, 3 };
        Mean.values = mean;
        IConstantLayer* m = network->addConstant(Dims3{ 3, 1, 1 }, Mean);
        IElementWiseLayer* sub_mean = network->addElementWise(*input, *m->getOutput(0), ElementWiseOperation::kSUB);
        if (std != nullptr) {
            Weights Std{ DataType::kFLOAT, nullptr, 3 };
            Std.values = std;
            IConstantLayer* s = network->addConstant(Dims3{ 3, 1, 1 }, Std);
            IElementWiseLayer* std_mean = network->addElementWise(*sub_mean->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
            return std_mean->getOutput(0);
        } else {
            return sub_mean->getOutput(0);
        }
    }

    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname, const float eps) {
        float *gamma = (float*)weightMap[lname + ".weight"].values;
        float *beta = (float*)weightMap[lname + ".bias"].values;
        float *mean = (float*)weightMap[lname + ".running_mean"].values;
        float *var = (float*)weightMap[lname + ".running_var"].values;
        int len = weightMap[lname + ".running_var"].count;

        float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            scval[i] = gamma[i] / sqrt(var[i] + eps);
        }
        Weights wscale{DataType::kFLOAT, scval, len};

        float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        }
        Weights wshift{DataType::kFLOAT, shval, len};

        float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            pval[i] = 1.0;
        }
        Weights wpower{DataType::kFLOAT, pval, len};

        weightMap[lname + ".scale"] = wscale;
        weightMap[lname + ".shift"] = wshift;
        weightMap[lname + ".power"] = wpower;
        IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, wshift, wscale, wpower);
        assert(scale_1);
        return scale_1;
    }

    IScaleLayer* addInstanceNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname, const float eps) {

        int len = weightMap[lname + ".weight"].count;

        IReduceLayer* reduce1 = network->addReduce(input, 
            ReduceOperation::kAVG,
            6, 
            true);
        assert(reduce1);

        IElementWiseLayer* ew1 = network->addElementWise(input, 
            *reduce1->getOutput(0),
            ElementWiseOperation::kSUB);  
        assert(ew1);

        const static float pval1[3]{0.0, 1.0, 2.0};   
        Weights wshift1{DataType::kFLOAT, pval1, 1};
        Weights wscale1{DataType::kFLOAT, pval1+1, 1};
        Weights wpower1{DataType::kFLOAT, pval1+2, 1};

        IScaleLayer* scale1 = network->addScale(
            *ew1->getOutput(0), 
            ScaleMode::kUNIFORM,
            wshift1,  
            wscale1,  
            wpower1); 
        assert(scale1);

        IReduceLayer* reduce2 = network->addReduce(
            *scale1->getOutput(0), 
            ReduceOperation::kAVG,
            6, 
            true);
        assert(reduce2);

        const static float pval2[3]{eps, 1.0, 0.5}; 
        Weights wshift2{DataType::kFLOAT, pval2, 1};
        Weights wscale2{DataType::kFLOAT, pval2+1, 1};
        Weights wpower2{DataType::kFLOAT, pval2+2, 1};
        
        IScaleLayer* scale2 = network->addScale(
            *reduce2->getOutput(0), 
            ScaleMode::kUNIFORM,
            wshift2,  
            wscale2,  
            wpower2);
        assert(scale2);

        IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), 
            *scale2->getOutput(0),
            ElementWiseOperation::kDIV); 
        assert(ew2);

        float* pval3 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        std::fill_n(pval3, len, 1.0); 
        Weights wpower3{DataType::kFLOAT, pval3, len};
        weightMap[lname + ".power3"] = wpower3;

        IScaleLayer* scale3 = network->addScale(
            *ew2->getOutput(0), 
            ScaleMode::kCHANNEL,
            weightMap[lname + ".bias"], 
            weightMap[lname + ".weight"],  
            wpower3); 
        assert(scale3);
        return scale3;
    }

    IConcatenationLayer* addIBN(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname) {
        Dims spliteDims = input.getDimensions();
        ISliceLayer *split1 = network->addSlice(input, 
            Dims3{0, 0, 0}, 
            Dims3{spliteDims.d[0]/2, spliteDims.d[1], spliteDims.d[2]}, 
            Dims3{1, 1, 1});
        assert(split1);

        ISliceLayer *split2 = network->addSlice(input, 
            Dims3{spliteDims.d[0]/2, 0, 0}, 
            Dims3{spliteDims.d[0]/2, spliteDims.d[1], spliteDims.d[2]}, 
            Dims3{1, 1, 1});
        assert(split2);

        auto in1 = addInstanceNorm2d(network, weightMap, *split1->getOutput(0), lname + "IN", 1e-5);
        auto bn1 = addBatchNorm2d(network, weightMap, *split2->getOutput(0), lname + "BN", 1e-5);

        ITensor* tensor1[] = {in1->getOutput(0), bn1->getOutput(0)};
        auto cat1 = network->addConcatenation(tensor1, 2);
        assert(cat1);
        return cat1;
    }

    IActivationLayer* bottleneck_ibn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const int inch, const int outch, const int stride, const std::string lname, const std::string ibn) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
        assert(conv1);

        IActivationLayer* relu1{nullptr};
        if (ibn == "a") {
            IConcatenationLayer* bn1 = addIBN(network, weightMap, *conv1->getOutput(0), lname + "bn1.");
            relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
            assert(relu1);
        } else {
            IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
            relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
            assert(relu1);
        }

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        assert(conv2);
        conv2->setStrideNd(DimsHW{stride, stride});
        conv2->setPaddingNd(DimsHW{1, 1});

        IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

        IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        assert(relu2);

        IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
        assert(conv3);

        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

        IElementWiseLayer* ew1;
        if (stride != 1 || inch != outch * 4) {
            IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
            assert(conv4);
            conv4->setStrideNd(DimsHW{stride, stride});

            IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
            ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        } else {
            ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        }
    
        IActivationLayer* relu3{nullptr};
        if (ibn == "b") {
            IScaleLayer* in1 = addInstanceNorm2d(network, weightMap, *ew1->getOutput(0), lname + "IN", 1e-5);
            relu3 = network->addActivation(*in1->getOutput(0), ActivationType::kRELU);
        } else {
            relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
        }

        assert(relu3);
        return relu3;
    }

}