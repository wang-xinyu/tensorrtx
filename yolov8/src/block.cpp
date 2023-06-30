#include "block.h"
#include "yololayer.h"
#include "config.h"
#include <iostream>
#include <assert.h>
#include <fstream>
#include <math.h>

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file){
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> WeightMap;

    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    int32_t count;
    input>>count ;
    assert(count > 0 && "Invalid weight map file.");

    while(count--){
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for(uint32_t x = 0, y = size; x < y; x++){
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
    }
    return WeightMap;
}


static nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
nvinfer1::ITensor& input, std::string lname, float eps){
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, pval, len };
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* output = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(output);
    return output;
}


nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, 
nvinfer1::ITensor& input, int ch, int k, int s, int p, std::string lname){
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname+".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname+".bn", 1e-5);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}


nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, 
nvinfer1::ITensor& input, int c1, int c2, bool shortcut, float e, std::string lname){
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c2, 3, 1, 1, lname+".cv1");
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *conv1->getOutput(0), c2, 3, 1, 1, lname+".cv2");
    
    if(shortcut && c1 == c2){
        nvinfer1::IElementWiseLayer* ew = network->addElementWise(input, *conv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    }
    return conv2;
}


nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, 
nvinfer1::ITensor& input, int c1, int c2, int n, bool shortcut, float e, std::string lname){
    int c_ = (float)c2 * e;
    
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, 2* c_, 1, 1, 0, lname+".cv1");
    nvinfer1::Dims d = conv1->getOutput(0)->getDimensions();
    
    nvinfer1::ISliceLayer* split1 = network->addSlice(*conv1->getOutput(0), nvinfer1::Dims3{0,0,0}, nvinfer1::Dims3{d.d[0]/2, d.d[1], d.d[2]}, nvinfer1::Dims3{1,1,1});
    nvinfer1::ISliceLayer* split2 = network->addSlice(*conv1->getOutput(0), nvinfer1::Dims3{d.d[0]/2,0,0}, nvinfer1::Dims3{d.d[0]/2, d.d[1], d.d[2]}, nvinfer1::Dims3{1,1,1});
    nvinfer1::ITensor* inputTensor0[] = {split1->getOutput(0), split2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor0, 2);
    nvinfer1::ITensor* y1 = split2->getOutput(0);
    for(int i = 0; i < n; i++){
        auto* b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, 1.0, lname+".m." + std::to_string(i));
        y1 = b->getOutput(0);

        nvinfer1::ITensor* inputTensors[] = {cat->getOutput(0), b->getOutput(0)};
        cat = network->addConcatenation(inputTensors, 2);
    }
    
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname+".cv2");
    
    return conv2;
}


nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, 
nvinfer1::ITensor& input, int c1, int c2, int k, std::string lname){
    int c_ = c1 / 2;
    
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c_, 1, 1, 0, lname+".cv1");
    
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k,k});
    pool1->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool1->setPaddingNd(nvinfer1::DimsHW{ k / 2, k / 2 });
    nvinfer1::IPoolingLayer* pool2 = network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k,k});
    pool2->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool2->setPaddingNd(nvinfer1::DimsHW{ k / 2, k / 2 });
    nvinfer1::IPoolingLayer* pool3 = network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k,k});
    pool3->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool3->setPaddingNd(nvinfer1::DimsHW{ k / 2, k / 2 });
    nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 4);
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname+".cv2");
    return conv2;
}


nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, 
nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname){

    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims3{4, 16, grid});
    shuffle1->setSecondTranspose(nvinfer1::Permutation{1, 0, 2});
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuffle1->getOutput(0));

    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*softmax->getOutput(0), 1, nvinfer1::DimsHW{1, 1}, weightMap[lname], bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});

    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setReshapeDimensions(nvinfer1::Dims2{4, grid});

    return shuffle2;
}


nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition *network, std::vector<nvinfer1::IConcatenationLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");

    nvinfer1::PluginField plugin_fields[1];
    int netinfo[4] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = nvinfer1::PluginFieldType::kFLOAT32;


    nvinfer1::PluginFieldCollection plugin_data;
    plugin_data.nbFields = 1;
    plugin_data.fields = plugin_fields;
    nvinfer1::IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}
