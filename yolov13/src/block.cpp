#include "block.h"
#include <assert.h>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include "config.h"
#include "model.h"
#include "yololayer.h"

using namespace std;


std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> WeightMap;

    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
        //std::cout << "===========name:              " << name << std::endl;
    }
    return WeightMap;
}

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    std::string lname, float eps) {
    //std::cout << "BatchNorm's name :             " << lname << endl;
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;


    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, shval, len };

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

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    int ch, std::vector<int> k, int s, std::string lname, int p, int g, int d) {

    nvinfer1::Weights bias_empty{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{ k[0], k[1] },
        weightMap[lname + ".conv.weight"], bias_empty);

    conv->setNbGroups(g);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{ s, s });
    int p0 = (p > 0 ? p : k[0] / 2);
    int p1 = (p > 0 ? p : k[1] / 2);
    conv->setPaddingNd(nvinfer1::DimsHW{ p0, p1 });
    conv->setDilationNd(nvinfer1::DimsHW{ d, d });
    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew =
        network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}



static nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    int c1, int c2, bool shortcut, std::vector<int> k1, std::vector<int> k2, float e,
    int g, std::string lname) {
    int c_ = (int)((float)c2 * e);
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c_, k1, 1, lname + ".cv1");
    nvinfer1::IElementWiseLayer* conv2 =
        convBnSiLU(network, weightMap, *conv1->getOutput(0), c2, k2, 1, lname + ".cv2", 0, g);


    if (shortcut && c1 == c2) {
        nvinfer1::IElementWiseLayer* ew =
            network->addElementWise(input, *conv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    }
    return conv2;
}



nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname) {

    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims4{ kBatchSize, 4, ch, grid });
    shuffle1->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1, 3 });
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuffle1->getOutput(0));
    softmax->setAxes(1 << 1);

    nvinfer1::Weights bias_empty{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::IConvolutionLayer* conv =
        network->addConvolutionNd(*softmax->getOutput(0), 1, nvinfer1::DimsHW{ 1, 1 }, weightMap[lname], bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW{ s, s });
    conv->setPaddingNd(nvinfer1::DimsHW{ p, p });


    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setReshapeDimensions(nvinfer1::Dims3{ kBatchSize, 4, grid });

    return shuffle2;
}

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
    std::vector<nvinfer1::IConcatenationLayer*> dets, const int* px_arry,
    int px_arry_num, bool is_segmentation, bool is_pose, bool is_obb) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const int netinfo_count = 9;  // Assuming the first 5 elements are for netinfo as per existing code.
    const int total_count = netinfo_count + px_arry_num;  // Total number of elements for netinfo and px_arry combined.

    std::vector<int> combinedInfo(total_count);
    int class_num = kNumClass;
    if (is_pose)
        class_num = kPoseNumClass;
    else if (is_obb)
        class_num = kObbNumClass;
    int input_w = kInputW;
    if (is_obb)
        input_w = kObbInputW;
    int input_h = kInputH;
    if (is_obb)
        input_h = kObbInputH;
    // Fill in the first 5 elements as per existing netinfo.
    combinedInfo[0] = class_num;
    combinedInfo[1] = kNumberOfPoints;
    combinedInfo[2] = kConfThreshKeypoints;
    combinedInfo[3] = input_w;
    combinedInfo[4] = input_h;
    combinedInfo[5] = kMaxNumOutputBbox;
    combinedInfo[6] = is_segmentation;
    combinedInfo[7] = is_pose;
    combinedInfo[8] = is_obb;

    // Copy the contents of px_arry into the combinedInfo vector after the initial 5 elements.
    std::copy(px_arry, px_arry + px_arry_num, combinedInfo.begin() + netinfo_count);

    // Now let's create the PluginField object to hold this combined information.
    nvinfer1::PluginField pluginField;
    pluginField.name = "combinedInfo";  // This can be any name that the plugin will recognize
    pluginField.data = combinedInfo.data();
    pluginField.type = nvinfer1::PluginFieldType::kINT32;
    pluginField.length = combinedInfo.size();

    // Create the PluginFieldCollection to hold the PluginField object.
    nvinfer1::PluginFieldCollection pluginFieldCollection;
    pluginFieldCollection.nbFields = 1;  // We have just one field, but it's a combined array
    pluginFieldCollection.fields = &pluginField;

    // Create the plugin object using the PluginFieldCollection.
    nvinfer1::IPluginV2* pluginObject = creator->createPlugin("yololayer", &pluginFieldCollection);

    // We assume that the plugin is to be added onto the network.
    // Prepare input tensors for the YOLO Layer.
    std::vector<nvinfer1::ITensor*> inputTensors;
    for (auto det : dets) {
        inputTensors.push_back(det->getOutput(0));  // Assuming each IConcatenationLayer has one output tensor.
    }

    // Add the plugin to the network using the prepared input tensors.
    nvinfer1::IPluginV2Layer* yoloLayer = network->addPluginV2(inputTensors.data(), inputTensors.size(), *pluginObject);

    return yoloLayer;  // Return the added YOLO layer.
}


nvinfer1::ILayer* Conv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int c_out, std::string lname, int k, int s, int padding,
    int g, bool act) {
    nvinfer1::Weights emptywts{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, c_out, nvinfer1::DimsHW{ k, k },
        weightMap[lname + ".conv.weight"], emptywts);


    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{ s, s });
    int p0 = (padding > 0 ? padding : k / 2);
    int p1 = (padding > 0 ? padding : k / 2);
    conv->setPaddingNd(nvinfer1::DimsHW{ p0, p1 });
    conv->setDilationNd(nvinfer1::DimsHW{ 1, 1 });
    conv->setNbGroups(g);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    if (act) {
        nvinfer1::IActivationLayer* sigmoid =
            network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
        nvinfer1::IElementWiseLayer* ew = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0),
            nvinfer1::ElementWiseOperation::kPROD);
        assert(ew);
        return ew;
    }
    else
        return bn;
}



nvinfer1::ILayer* DWConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int ch, std::vector<int> k, int s, std::string lname) {
    nvinfer1::Weights bias_empty{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{ k[0], k[1] },
        weightMap[lname + ".conv.weight"], bias_empty);

    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{ s, s });
    conv->setNbGroups(ch);
    // auto pad
    int p0 = k[0] / 2;
    int p1 = k[1] / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{ p0, p1 });

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew =
        network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

nvinfer1::IElementWiseLayer* C3k(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c2,
    std::string lname, int n, bool shortcut, int g, float e, int k) {
    int c_ = c2 * float(e);

    nvinfer1::IElementWiseLayer* cv1 = convBnSiLU(network, weightMap, input, c_, { 1, 1 }, 1, lname + ".cv1");
    nvinfer1::IElementWiseLayer* cv2 = convBnSiLU(network, weightMap, input, c_, { 1, 1 }, 1, lname + ".cv2");
    nvinfer1::ITensor* y = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        nvinfer1::ILayer* b = bottleneck(network, weightMap, *y, c_, c_, shortcut, { k, k }, { k, k }, 1.0, g,
            lname + ".m." + std::to_string(i));
        y = b->getOutput(0);
    }
    nvinfer1::ITensor* inputTensor[] = { y, cv2->getOutput(0) };
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor, 2);
    nvinfer1::IElementWiseLayer* cv3 =
        convBnSiLU(network, weightMap, *cat->getOutput(0), c2, { 1, 1 }, 1, lname + ".cv3");

    return cv3;
}


nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c2,
    int n, std::string lname, bool c3k, float e, int g, bool shortcut) {

    std::cout << "c3k' value: " << c3k << endl;
    int c = int(c2 * float(e));
    nvinfer1::ILayer* cv1 = Conv(network, weightMap, input, 2 * c, lname + ".cv1", 1, 1);
    nvinfer1::ISliceLayer* sl0 = network->addSlice(*cv1->getOutput(0), nvinfer1::Dims4{ 0, 0, 0, 0 },
        nvinfer1::Dims4{ cv1->getOutput(0)->getDimensions().d[0],
        cv1->getOutput(0)->getDimensions().d[1] / 2, cv1->getOutput(0)->getDimensions().d[2],
        cv1->getOutput(0)->getDimensions().d[3] }, nvinfer1::Dims4{ 1, 1, 1, 1 });
    nvinfer1::ISliceLayer* sl1 = network->addSlice(
        *cv1->getOutput(0), nvinfer1::Dims4{ 0, cv1->getOutput(0)->getDimensions().d[1] / 2, 0, 0 },
        nvinfer1::Dims4{ cv1->getOutput(0)->getDimensions().d[0], cv1->getOutput(0)->getDimensions().d[1] / 2,
                        cv1->getOutput(0)->getDimensions().d[2], cv1->getOutput(0)->getDimensions().d[3] },
        nvinfer1::Dims4{ 1, 1, 1, 1 });
    std::vector<nvinfer1::ITensor*> y = { sl0->getOutput(0), sl1->getOutput(0) };
    nvinfer1::ITensor* current = sl1->getOutput(0);
    for (int i = 0; i < n; i++) {
        if (c3k) {
            nvinfer1::IElementWiseLayer* m_ = C3k(network, weightMap, *current, c, lname + ".m." + std::to_string(i), 2, shortcut, g);
            current = m_->getOutput(0);
            y.push_back(current);
        }
        else {
            nvinfer1::ILayer* m_ = bottleneck(network, weightMap, *current, c, c, shortcut, { 3, 3 }, { 3, 3 }, 0.5, g,
                lname + ".m." + std::to_string(i));
            current = m_->getOutput(0);
            y.push_back(current);
        }
    }
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(y.data(), y.size());
    nvinfer1::IElementWiseLayer* cv2 =
        convBnSiLU(network, weightMap, *cat->getOutput(0), c2, { 1, 1 }, 1, lname + ".cv2");

    return cv2;
}


nvinfer1::IElementWiseLayer* DSConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    int c_in, int c_out, std::string lname, int k, int s, int p, int d, bool bias) {
    if (p == 0) {
        p = (d * (k - 1)) / 2;
    }
    nvinfer1::Weights emptywts{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::IConvolutionLayer* dw = network->addConvolutionNd(input, c_in, nvinfer1::DimsHW{ k, k }, weightMap[lname + ".dw.weight"], emptywts);
    dw->setStrideNd(nvinfer1::DimsHW{ s, s });
    dw->setPaddingNd(nvinfer1::DimsHW{ p, p });
    dw->setNbGroups(c_in);
    dw->setDilationNd(nvinfer1::DimsHW{ d, d });

    nvinfer1::IConvolutionLayer* pw =
        network->addConvolutionNd(*dw->getOutput(0), c_out, nvinfer1::DimsHW{ 1, 1 }, weightMap[lname + ".pw.weight"], emptywts);
    pw->setStrideNd(nvinfer1::DimsHW{ 1, 1 });
    pw->setPaddingNd(nvinfer1::DimsHW{ 0, 0 });
    pw->setNbGroups(1);
    pw->setDilationNd(nvinfer1::DimsHW{ 1, 1 });


    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *pw->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew =
        network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

nvinfer1::ILayer* DSBottleneck(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int c1, int c2, std::string lname, bool shortcut,
    float e, int k1, int k2, int d2) {
    int c_ = float(e) * c2;
    nvinfer1::IElementWiseLayer* cv1 = DSConv(network, weightMap, input, c1, c_, lname + ".cv1", k1, 1, 0, 1, false);
    nvinfer1::IElementWiseLayer* y =
        DSConv(network, weightMap, *cv1->getOutput(0), c_, c2, lname + ".cv2", k2, 1, 0, d2, false);
    if (c1 == c2 && shortcut) {
        nvinfer1::IElementWiseLayer* add =
            network->addElementWise(input, *y->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return add;
    }
    else
        return y;

}

nvinfer1::ILayer* DSC3k(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c2, int n, std::string lname,
    bool shortcut, int g, float e, int k1, int k2, int d2) {
    int c_ = float(e) * c2;
    nvinfer1::ILayer* cv1 = Conv(network, weightMap, input, c_, lname + ".cv1", 1, 1);
    nvinfer1::ILayer* cv2 = Conv(network, weightMap, input, c_, lname + ".cv2", 1, 1);
    nvinfer1::ITensor* current = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        nvinfer1::ILayer* m_ = DSBottleneck(network, weightMap, *current, c_, c_, lname + ".m." + std::to_string(i),
            shortcut, 1.0, k1, k2, d2);
        current = m_->getOutput(0);
    }
    nvinfer1::ITensor* inputTensors[] = { current, cv2->getOutput(0) };
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 2);
    nvinfer1::ILayer* cv3 = Conv(network, weightMap, *cat->getOutput(0), c2, lname + ".cv3", 1, 1);

    return cv3;
}


nvinfer1::ILayer* DSC3K2(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int c2, std::string lname, int n, bool dsc3k, float e, int g,
    bool shortcut, int k1, int k2, int d2) {
    int c = float(e) * c2;
    nvinfer1::ILayer* cv1 = Conv(network, weightMap, input, 2 * c, lname + ".cv1");
    nvinfer1::Dims dim_cv1 = cv1->getOutput(0)->getDimensions();
    nvinfer1::ISliceLayer* sl0 = network->addSlice(
        *cv1->getOutput(0), nvinfer1::Dims4{ 0, 0, 0, 0 },
        nvinfer1::Dims4{ dim_cv1.d[0], dim_cv1.d[1] / 2, dim_cv1.d[2], dim_cv1.d[3] }, nvinfer1::Dims4{ 1, 1, 1, 1 });
    nvinfer1::ISliceLayer* sl1 = network->addSlice(
        *cv1->getOutput(0), nvinfer1::Dims4{ 0, dim_cv1.d[1] / 2, 0, 0 },
        nvinfer1::Dims4{ dim_cv1.d[0], dim_cv1.d[1] / 2, dim_cv1.d[2], dim_cv1.d[3] }, nvinfer1::Dims4{ 1, 1, 1, 1 });
    std::vector<nvinfer1::ITensor*> y = { sl0->getOutput(0), sl1->getOutput(0) };
    nvinfer1::ITensor* current = sl1->getOutput(0);
    for (int i = 0; i < n; i++) {
        if (dsc3k) {
            nvinfer1::ILayer* m_ = DSC3k(network, weightMap, *current, c, 2,
                lname + ".m." + std::to_string(i), shortcut, g, 1.0, k1, k2, d2);
            current = m_->getOutput(0);
            y.push_back(current);
        }
        else {
            nvinfer1::ILayer* m_ = DSBottleneck(network, weightMap, *current, c, c,
                lname + ".m." + std::to_string(i), shortcut, 1.0, k1, k2, d2);
            current = m_->getOutput(0);
            y.push_back(current);
        }
    }
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(y.data(), y.size());
    nvinfer1::ILayer* cv2 = Conv(network, weightMap, *cat->getOutput(0), c2, lname + ".cv2");

    return cv2;

}


nvinfer1::ILayer* FuseModule(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    std::vector<nvinfer1::ITensor*>& input, int c_in, bool channel_adjust, std::string lname) {
    nvinfer1::IPoolingLayer* x1_ds =
        network->addPoolingNd(*input[0], nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{ 2, 2 });
    x1_ds->setStrideNd(nvinfer1::DimsHW{ 2, 2 });
    x1_ds->setPaddingNd(nvinfer1::DimsHW{ 0, 0 });

    nvinfer1::IResizeLayer* x3_up = network->addResize(*input[2]);
    float scale[] = { 1, 1, 2, 2 };
    x3_up->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    x3_up->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor[] = { x1_ds->getOutput(0), input[1], x3_up->getOutput(0) };
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor, 3);
    cat->setAxis(1);
    nvinfer1::ILayer* conv_out = Conv(network, weightMap, *cat->getOutput(0), c_in, lname + ".conv_out");
    return conv_out;

}

nvinfer1::ISoftMaxLayer* AdaHyperedgeGen(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int node_dim, int num_hyperedges, std::string lname, int num_heads,
    std::string context) {

    nvinfer1::Dims dim_input = input.getDimensions();
    int B = dim_input.d[0];
    int N = dim_input.d[1];
    int D = dim_input.d[2];
    int head_dim = node_dim / num_heads;
    nvinfer1::ITensor* context_cat = nullptr;
    if (context == "mean") {
        nvinfer1::IReduceLayer* context_mean = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, 1 << 1, false);
        context_cat = context_mean->getOutput(0);
    }
    else if (context == "max") {
        nvinfer1::IReduceLayer* context_max = network->addReduce(input, nvinfer1::ReduceOperation::kMAX, 1 << 1, false);
        context_cat = context_max->getOutput(0);
    }
    else {
        nvinfer1::IReduceLayer* context_mean = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, 1 << 1, false);
        nvinfer1::IReduceLayer* context_max = network->addReduce(input, nvinfer1::ReduceOperation::kMAX, 1 << 1, false);
        nvinfer1::ITensor* inputTensor[] = { context_mean->getOutput(0), context_max->getOutput(0) };
        nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor, 2);
        cat->setAxis(1 << 0);
        context_cat = cat->getOutput(0);
    }

    nvinfer1::IShuffleLayer* context_cat_dim4 = network->addShuffle(*context_cat);
    context_cat_dim4->setReshapeDimensions(nvinfer1::Dims4{ context_cat->getDimensions().d[0],
                                                           context_cat->getDimensions().d[1],
                                                           1, 1 });
    nvinfer1::IFullyConnectedLayer* prototypes_offsets_ = network->addFullyConnected(*context_cat_dim4->getOutput(0),
        num_hyperedges * node_dim, weightMap[lname + ".context_net.weight"], weightMap[lname + ".context_net.bias"]);
    nvinfer1::IShuffleLayer* prototypes_offsets = network->addShuffle(*prototypes_offsets_->getOutput(0));
    prototypes_offsets->setReshapeDimensions(nvinfer1::Dims3{ B, num_hyperedges, D });
    // prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  

    nvinfer1::Weights prototype_base_wts = weightMap[lname + ".prototype_base"];
    nvinfer1::IConstantLayer* prototype_base = network->addConstant(
        nvinfer1::Dims3{ 1, num_hyperedges, node_dim }, prototype_base_wts);
    nvinfer1::IElementWiseLayer* prototypes = network->addElementWise(*prototype_base->getOutput(0),
        *prototypes_offsets->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM);
    // prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

    nvinfer1::IShuffleLayer* input_dim4 = network->addShuffle(input);
    input_dim4->setReshapeDimensions(nvinfer1::Dims4{ B * N, D, 1, 1 });
    nvinfer1::IFullyConnectedLayer* X_proj = network->addFullyConnected(*input_dim4->getOutput(0), node_dim,
        weightMap[lname + ".pre_head_proj.weight"], weightMap[lname + ".pre_head_proj.bias"]);
    // X_proj = self.pre_head_proj(X) 

    nvinfer1::IShuffleLayer* X_heads = network->addShuffle(*X_proj->getOutput(0));
    X_heads->setReshapeDimensions(nvinfer1::Dims4{ B, N, num_heads, head_dim });
    X_heads->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1, 3 });
    // X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    nvinfer1::IShuffleLayer* proto_heads = network->addShuffle(*prototypes->getOutput(0));
    proto_heads->setReshapeDimensions(nvinfer1::Dims4{ B, num_hyperedges, num_heads, head_dim });
    proto_heads->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1, 3 });
    // proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    nvinfer1::IShuffleLayer* X_heads_flat = network->addShuffle(*X_heads->getOutput(0));
    X_heads_flat->setReshapeDimensions(nvinfer1::Dims3{ B * num_heads, N, head_dim });
    // X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)

    nvinfer1::IShuffleLayer* proto_heads_flat = network->addShuffle(*proto_heads->getOutput(0));
    proto_heads_flat->setReshapeDimensions(nvinfer1::Dims3{ B * num_heads, num_hyperedges, head_dim });
    proto_heads_flat->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1 });
    //proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)

    nvinfer1::IMatrixMultiplyLayer* logits = network->addMatrixMultiply(*X_heads_flat->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *proto_heads_flat->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    float* scales_ptr = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scales_ptr = sqrt(static_cast<float>(head_dim));
    nvinfer1::Weights scale_wts{ nvinfer1::DataType::kFLOAT, scales_ptr, 1 };
    nvinfer1::IConstantLayer* scale_layer = network->addConstant(nvinfer1::Dims3{ 1, 1, 1 }, scale_wts);
    // keep weight alive during build
    weightMap[lname + ".scaling"] = scale_wts;
    nvinfer1::IElementWiseLayer* logits_scale = network->addElementWise(*logits->getOutput(0), *scale_layer->getOutput(0),
        nvinfer1::ElementWiseOperation::kDIV);
    // logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 

    nvinfer1::IShuffleLayer* logits_scale_view = network->addShuffle(*logits_scale->getOutput(0));
    logits_scale_view->setReshapeDimensions(nvinfer1::Dims4{ B, num_heads, N, num_hyperedges });
    nvinfer1::IReduceLayer* logits_scale_view_mean =
        network->addReduce(*logits_scale_view->getOutput(0), nvinfer1::ReduceOperation::kAVG,
            1 << 1, false);


    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*logits_scale_view_mean->getOutput(0));
    softmax->setAxes(1 << 1);

    return softmax;
}

nvinfer1::IElementWiseLayer* GELU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input) {
    static float sqrt_2_over_pi = 0.797885f;  // 0.7978845608
    static float kappa = 0.044715f;
    static float one = 1.0f;
    static float half = 0.5f;

    nvinfer1::IElementWiseLayer* x3_layer = network->addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::ITensor* x2 = x3_layer->getOutput(0);
    x3_layer = network->addElementWise(*x2, input, nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::ITensor* x3 = x3_layer->getOutput(0);

    nvinfer1::Weights kappa_weight{ nvinfer1::DataType::kFLOAT, &kappa, 1 };
    nvinfer1::IConstantLayer* kappa_const = network->addConstant(nvinfer1::Dims4{ 1, 1, 1, 1 }, kappa_weight);
    nvinfer1::IElementWiseLayer* scaled_x3 =
        network->addElementWise(*x3, *kappa_const->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::IElementWiseLayer* inner_sum =
        network->addElementWise(input, *scaled_x3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    nvinfer1::ITensor* inner = inner_sum->getOutput(0);

    nvinfer1::Weights sqrt_weight{ nvinfer1::DataType::kFLOAT, &sqrt_2_over_pi, 1 };
    nvinfer1::IConstantLayer* sqrt_const = network->addConstant(nvinfer1::Dims4{ 1, 1, 1, 1 }, sqrt_weight);
    nvinfer1::IElementWiseLayer* scaled_inner =
        network->addElementWise(*inner, *sqrt_const->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::IActivationLayer* tanh_layer = network->addActivation(*scaled_inner->getOutput(0), nvinfer1::ActivationType::kTANH);

    nvinfer1::Weights one_weight{ nvinfer1::DataType::kFLOAT, &one, 1 };
    nvinfer1::IConstantLayer* one_const = network->addConstant(nvinfer1::Dims4{ 1, 1, 1, 1 }, one_weight);
    nvinfer1::IElementWiseLayer* add_one =
        network->addElementWise(*tanh_layer->getOutput(0), *one_const->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::IElementWiseLayer* half_x =
        network->addElementWise(input, *add_one->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::Weights half_weight{ nvinfer1::DataType::kFLOAT, &half, 1 };
    nvinfer1::IConstantLayer* half_const = network->addConstant(nvinfer1::Dims4{ 1, 1, 1, 1 }, half_weight);
    nvinfer1::IElementWiseLayer* gelu = network->addElementWise(*half_x->getOutput(0), *half_const->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD);
    return gelu;
}


nvinfer1::IElementWiseLayer* AdaHGConv(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    int embed_dim, std::string lname, int num_hyperedges, int num_heads, std::string context) {

    // {B, N, num_hyperedges}
    nvinfer1::ISoftMaxLayer* A = AdaHyperedgeGen(network, weightMap, input, embed_dim, num_hyperedges,
        lname + ".edge_generator", num_heads, context);
    nvinfer1::IMatrixMultiplyLayer* He = network->addMatrixMultiply(  // 486 layer
        *A->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE, input, nvinfer1::MatrixOperation::kNONE);
    nvinfer1::IShuffleLayer* He_dim4 = network->addShuffle(*He->getOutput(0));
    He_dim4->setReshapeDimensions(nvinfer1::Dims4{ He->getOutput(0)->getDimensions().d[1],
                                                  He->getOutput(0)->getDimensions().d[0],
                                                  He->getOutput(0)->getDimensions().d[2], 1 });

    nvinfer1::IFullyConnectedLayer* He_edge_proj_ = network->addFullyConnected(*He_dim4->getOutput(0),
        embed_dim, weightMap[lname + ".edge_proj.0.weight"], weightMap[lname + ".edge_proj.0.bias"]);
    nvinfer1::IElementWiseLayer* He_edge_proj = GELU(network, *He_edge_proj_->getOutput(0));
    nvinfer1::IShuffleLayer* He_edge_proj_dim2 = network->addShuffle(*He_edge_proj->getOutput(0));
    He_edge_proj_dim2->setReshapeDimensions(nvinfer1::Dims2{ He_edge_proj->getOutput(0)->getDimensions().d[0],
                                                            He_edge_proj->getOutput(0)->getDimensions().d[1] });
    nvinfer1::IShuffleLayer* A_dim2 = network->addShuffle(*A->getOutput(0));
    A_dim2->setReshapeDimensions(
        nvinfer1::Dims2{ A->getOutput(0)->getDimensions().d[1] * A->getOutput(0)->getDimensions().d[0], // keep the batch information
                        A->getOutput(0)->getDimensions().d[2] });
    nvinfer1::IMatrixMultiplyLayer* x_new_ = network->addMatrixMultiply(*A_dim2->getOutput(0),
        nvinfer1::MatrixOperation::kNONE, *He_edge_proj_dim2->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    nvinfer1::IShuffleLayer* x_new_dim4 = network->addShuffle(*x_new_->getOutput(0));
    x_new_dim4->setReshapeDimensions(nvinfer1::Dims4{ x_new_->getOutput(0)->getDimensions().d[0],
                                                     x_new_->getOutput(0)->getDimensions().d[1], 1, 1 });
    nvinfer1::IFullyConnectedLayer* x_new_node_proj_ = network->addFullyConnected(*x_new_dim4->getOutput(0), embed_dim,
        weightMap[lname + ".node_proj.0.weight"], weightMap[lname + ".node_proj.0.bias"]);
    nvinfer1::IElementWiseLayer* x_new_node_proj = GELU(network, *x_new_node_proj_->getOutput(0));
    nvinfer1::IShuffleLayer* x_new_finall = network->addShuffle(*x_new_node_proj->getOutput(0));
    x_new_finall->setReshapeDimensions(nvinfer1::Dims3{ 1, x_new_node_proj->getOutput(0)->getDimensions().d[0],
                                                       x_new_node_proj->getOutput(0)->getDimensions().d[1] });
    nvinfer1::IElementWiseLayer* add =
        network->addElementWise(*x_new_finall->getOutput(0), input, nvinfer1::ElementWiseOperation::kSUM);

    return add;
}

nvinfer1::IShuffleLayer* AdaHGComputation(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    int embed_dim, std::string lname, int num_hyperedges, int num_heads,
    std::string context) {
    nvinfer1::Dims dim = input.getDimensions();
    int B = dim.d[0];
    int C = dim.d[1];
    int H = dim.d[2];
    int W = dim.d[3];
    nvinfer1::IShuffleLayer* tokens = network->addShuffle(input);
    tokens->setReshapeDimensions(nvinfer1::Dims3{ B, C, H * W });
    tokens->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1 });
    nvinfer1::IElementWiseLayer* hgnn = AdaHGConv(network, weightMap, *tokens->getOutput(0), embed_dim, lname + ".hgnn",
        num_hyperedges, num_heads, context);

    nvinfer1::IShuffleLayer* x_out = network->addShuffle(*hgnn->getOutput(0));
    x_out->setFirstTranspose(nvinfer1::Permutation{ 0, 2, 1 });
    x_out->setReshapeDimensions(nvinfer1::Dims4{ B, C, H, W });

    return x_out;
}

nvinfer1::ILayer* C3AH(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int c2, std::string lname, float e, int num_hyperedges, std::string context) {
    int c_ = float(e) * c2;
    assert(c_ % 16 == 0 && "Dimension of AdaHGComputation should be a multiplt of 16");
    int num_heads = c_ / 16;
    nvinfer1::ILayer* cv1 = Conv(network, weightMap, input, c_, lname + ".cv1");
    nvinfer1::ILayer* cv2 = Conv(network, weightMap, input, c_, lname + ".cv2");

    nvinfer1::IShuffleLayer* m = AdaHGComputation(network, weightMap, *cv1->getOutput(0), c_,
        lname + ".m", num_hyperedges, num_heads, context);
    nvinfer1::ITensor* inputTensor[] = { m->getOutput(0), cv2->getOutput(0) };
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor, 2);
    nvinfer1::ILayer* cv3 = Conv(network, weightMap, *cat->getOutput(0), c2, lname + ".cv3");
    return cv3;
}

nvinfer1::ILayer* HyperACE(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    std::vector<nvinfer1::ITensor*> input, int c1, int c2, std::string lname, int n, int num_hyperedges,
    bool dsc3k, bool shortcut, float e1, float e2, std::string context, bool channel_adjust) {
    int c = int(c2 * e1);
    nvinfer1::ILayer* fuse = FuseModule(network, weightMap, input, c1, channel_adjust, lname + ".fuse");
    nvinfer1::ILayer* cv1 = Conv(network, weightMap, *fuse->getOutput(0), 3 * c, lname + ".cv1");
    nvinfer1::Dims d_cv1 = cv1->getOutput(0)->getDimensions();
    nvinfer1::ISliceLayer* sl0 = network->addSlice(*cv1->getOutput(0), nvinfer1::Dims4{ 0, 0, 0, 0 },
        nvinfer1::Dims4{ d_cv1.d[0], d_cv1.d[1] / 3, d_cv1.d[2], d_cv1.d[3] },
        nvinfer1::Dims4{ 1, 1, 1, 1 });
    nvinfer1::ISliceLayer* sl1 = network->addSlice(*cv1->getOutput(0), nvinfer1::Dims4{ 0, d_cv1.d[1] / 3, 0, 0 },
        nvinfer1::Dims4{ d_cv1.d[0], d_cv1.d[1] / 3, d_cv1.d[2], d_cv1.d[3] },
        nvinfer1::Dims4{ 1, 1, 1, 1 });
    nvinfer1::ISliceLayer* sl2 = network->addSlice(*cv1->getOutput(0), nvinfer1::Dims4{ 0, d_cv1.d[1] / 3 * 2, 0, 0 },
        nvinfer1::Dims4{ d_cv1.d[0], d_cv1.d[1] / 3, d_cv1.d[2], d_cv1.d[3] },
        nvinfer1::Dims4{ 1, 1, 1, 1 });
    std::vector<nvinfer1::ITensor*> y = { sl0->getOutput(0), sl1->getOutput(0), sl2->getOutput(0) };
    nvinfer1::ILayer* out1 = C3AH(network, weightMap, *y[1], c, lname + ".branch1", e2, num_hyperedges, context);
    nvinfer1::ILayer* out2 = C3AH(network, weightMap, *y[1], c, lname + ".branch2", e2, num_hyperedges, context);
    nvinfer1::ITensor* current = y[2];
    for (int i = 0; i < n; i++) {
        if (dsc3k) {
            nvinfer1::ILayer* m_ = DSC3k(network, weightMap, *current, c, 2,
                lname + ".m." + std::to_string(i), shortcut, 1, 0.5, 3, 7, 1);
            current = m_->getOutput(0);
        }
        else {
            nvinfer1::ILayer* m_ = DSBottleneck(network, weightMap, *current,
                c, c, lname + ".m." + std::to_string(i), shortcut);
            current = m_->getOutput(0);
        }
        y.push_back(current);
    }

    y[1] = out1->getOutput(0);
    y.push_back(out2->getOutput(0));

    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(y.data(), y.size());
    nvinfer1::ILayer* cv2 = Conv(network, weightMap, *cat->getOutput(0), c2, lname + ".cv2");

    return cv2;
}

nvinfer1::ILayer* DownsampleConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int in_channels, std::string lname, bool channel_adjust) {
    nvinfer1::IPoolingLayer* downsample =
        network->addPoolingNd(input, nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{ 2, 2 });
    downsample->setStrideNd(nvinfer1::DimsHW{ 2, 2 });
    downsample->setPaddingNd(nvinfer1::DimsHW{ 0, 0 });
    if (channel_adjust) {
        nvinfer1::ILayer* channel_adjust_ =
            Conv(network, weightMap, *downsample->getOutput(0), in_channels * 2, lname + ".channel_adjust");
        return channel_adjust_;
    }
    else
        return downsample;
}

nvinfer1::IElementWiseLayer* FullPad_Tunnel(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    std::vector<nvinfer1::ITensor*> input, std::string lname) {
    nvinfer1::Weights gate = weightMap[lname + ".gate"];
    nvinfer1::IConstantLayer* gate_constant = network->addConstant(nvinfer1::Dims4{ 1, 1, 1, 1 }, gate);
    nvinfer1::IElementWiseLayer* scaled_input_1 = network->addElementWise(*input[1],
        *gate_constant->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::IElementWiseLayer* add =
        network->addElementWise(*input[0], *scaled_input_1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    return add;
}

void cout_dim(nvinfer1::ITensor& input) {

    nvinfer1::Dims d = input.getDimensions();

    std::cout << "======================= Dimensions =================================" << std::endl;
    std::cout << "          " << d.d[0] << std::endl;
    std::cout << "          " << d.d[1] << std::endl;
    std::cout << "          " << d.d[2] << std::endl;
    std::cout << "          " << d.d[3] << std::endl;
    std::cout << "======================================================================" << std::endl;

}




nvinfer1::ILayer* AAttn(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int dim, int num_heads, std::string lname, int area) {

    nvinfer1::Dims d_input = input.getDimensions();
    int B = d_input.d[0];
    int C = d_input.d[1];
    int H = d_input.d[2];
    int W = d_input.d[3];
    int N = W * H;
    int head_dim = dim / num_heads;
    int all_head_dim = head_dim * num_heads;

    nvinfer1::ILayer* qk = Conv(network, weightMap, input, all_head_dim * 2, lname + ".qk", 1, 1, 0, 1, false);
    nvinfer1::IShuffleLayer* qk_flatten_t = network->addShuffle(*qk->getOutput(0));
    qk_flatten_t->setReshapeDimensions(nvinfer1::Dims3{ B, -1, N });
    qk_flatten_t->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1 });

    nvinfer1::ILayer* v = Conv(network, weightMap, input, all_head_dim, lname + ".v", 1, 1, 0, 1, false);
    nvinfer1::IShuffleLayer* v_flatten_t = network->addShuffle(*v->getOutput(0));
    v_flatten_t->setReshapeDimensions(nvinfer1::Dims3{ B, -1, N });
    v_flatten_t->setSecondTranspose(nvinfer1::Permutation{ 0, 2, 1 });  // (1, 6400, 64)

    nvinfer1::ILayer* pe = Conv(network, weightMap, *v->getOutput(0), dim, lname + ".pe", 5, 1, 2, dim, false);

    nvinfer1::ITensor* q_k = qk_flatten_t->getOutput(0);
    nvinfer1::ITensor* v_ = v_flatten_t->getOutput(0);
    if (area > 1) {
        B = B * area;
        N = N / area;

        nvinfer1::IShuffleLayer* qk_reshape = network->addShuffle(*qk_flatten_t->getOutput(0));
        qk_reshape->setReshapeDimensions(nvinfer1::Dims3{ B, N, C * 2 });
        nvinfer1::IShuffleLayer* v_reshape = network->addShuffle(*v_flatten_t->getOutput(0));
        v_reshape->setReshapeDimensions(nvinfer1::Dims3{ B, N, C });

        q_k = qk_reshape->getOutput(0);
        v_ = v_reshape->getOutput(0);
    }
    nvinfer1::Dims q_k_dim = q_k->getDimensions();
    nvinfer1::ISliceLayer* q =
        network->addSlice(*q_k, nvinfer1::Dims3{ 0, 0, 0 },
            nvinfer1::Dims3{ q_k_dim.d[0], q_k_dim.d[1], q_k_dim.d[2] / 2 }, nvinfer1::Dims3{ 1, 1, 1 });
    nvinfer1::ISliceLayer* k =
        network->addSlice(*q_k, nvinfer1::Dims3{ 0, 0, q_k_dim.d[2] / 2 },
            nvinfer1::Dims3{ q_k_dim.d[0], q_k_dim.d[1], q_k_dim.d[2] / 2 }, nvinfer1::Dims3{ 1, 1, 1 });

    nvinfer1::IShuffleLayer* q_reshape = network->addShuffle(*q->getOutput(0));
    q_reshape->setReshapeDimensions(nvinfer1::Dims4{ B, N, num_heads, head_dim });
    nvinfer1::IShuffleLayer* k_reshape = network->addShuffle(*k->getOutput(0));
    k_reshape->setReshapeDimensions(nvinfer1::Dims4{ B, N, num_heads, head_dim });
    nvinfer1::IShuffleLayer* v_reshape = network->addShuffle(*v_);
    v_reshape->setReshapeDimensions(nvinfer1::Dims4{ B, N, num_heads, head_dim });

    // (B, N, num_head, head_dim)--->(B, num_head, head_dim, N)
    nvinfer1::IShuffleLayer* q_t_view = network->addShuffle(*q_reshape->getOutput(0));
    q_t_view->setFirstTranspose(nvinfer1::Permutation{ 0, 2, 3, 1 });

    nvinfer1::IShuffleLayer* k_t_view = network->addShuffle(*k_reshape->getOutput(0));
    k_t_view->setFirstTranspose(nvinfer1::Permutation{ 0, 2, 3, 1 });
    nvinfer1::IShuffleLayer* v_t_view = network->addShuffle(*v_reshape->getOutput(0));
    v_t_view->setFirstTranspose(nvinfer1::Permutation{ 0, 2, 3, 1 });


    nvinfer1::IShuffleLayer* q_T = network->addShuffle(*q_t_view->getOutput(0));
    q_T->setFirstTranspose(nvinfer1::Permutation{ 0, 1, 3, 2 }); // (B, num_head, N, head_dim, N)
    nvinfer1::IMatrixMultiplyLayer* q_mul_k =
        network->addMatrixMultiply(*q_T->getOutput(0), nvinfer1::MatrixOperation::kNONE, *k_t_view->getOutput(0),
            nvinfer1::MatrixOperation::kNONE);

    float scale = 1.0 / sqrt(head_dim);
    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    scale_val[0] = scale;
    nvinfer1::Weights s_w{ nvinfer1::DataType::kFLOAT, scale_val, 1 };  // scale
    float* shift_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shift_val[0] = 0;
    nvinfer1::Weights sh_w{ nvinfer1::DataType::kFLOAT, shift_val, 1 };  // shift
    float* power_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    power_val[0] = 1;
    nvinfer1::Weights p_w{ nvinfer1::DataType::kFLOAT, power_val, 1 };  // power
    nvinfer1::IScaleLayer* q_mul_k_scale =
        network->addScale(*q_mul_k->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, sh_w, s_w, p_w);

    nvinfer1::IReduceLayer* attn_max =
        network->addReduce(*q_mul_k_scale->getOutput(0), nvinfer1::ReduceOperation::kMAX, 1 << 3, true);

    nvinfer1::IElementWiseLayer* attn_sub = network->addElementWise(
        *q_mul_k_scale->getOutput(0), *attn_max->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
    nvinfer1::IUnaryLayer* attn_exp = network->addUnary(*attn_sub->getOutput(0), nvinfer1::UnaryOperation::kEXP);
    nvinfer1::IReduceLayer* attn_sum =
        network->addReduce(*attn_exp->getOutput(0), nvinfer1::ReduceOperation::kSUM, 1 << 3, true);

    nvinfer1::IElementWiseLayer* attn_div = network->addElementWise(*attn_exp->getOutput(0), *attn_sum->getOutput(0),
        nvinfer1::ElementWiseOperation::kDIV);
    //cout_dim(*attn_div->getOutput(0));

    nvinfer1::IShuffleLayer* attn_t = network->addShuffle(*attn_div->getOutput(0));
    attn_t->setFirstTranspose(nvinfer1::Permutation{ 0, 1, 3, 2 });

    nvinfer1::IMatrixMultiplyLayer* attn_v =
        network->addMatrixMultiply(*v_t_view->getOutput(0), nvinfer1::MatrixOperation::kNONE, *attn_t->getOutput(0),
            nvinfer1::MatrixOperation::kNONE);

    nvinfer1::IShuffleLayer* attn_v_t = network->addShuffle(*attn_v->getOutput(0));
    attn_v_t->setFirstTranspose(nvinfer1::Permutation{ 0, 3, 1, 2 });
    nvinfer1::ITensor* attn_temp = attn_v_t->getOutput(0);
    if (area > 1) {
        B = B / area;
        N = N * area;

        nvinfer1::IShuffleLayer* attn_v_t_r = network->addShuffle(*attn_v_t->getOutput(0));
        attn_v_t_r->setReshapeDimensions(nvinfer1::Dims3{ B, N, C });
        attn_temp = attn_v_t_r->getOutput(0);
    }
    nvinfer1::IShuffleLayer* attn_x = network->addShuffle(*attn_temp);
    attn_x->setReshapeDimensions(nvinfer1::Dims4{ B, H, W, C });
    attn_x->setSecondTranspose(nvinfer1::Permutation{ 0, 3, 1, 2 });
    nvinfer1::IElementWiseLayer* x_add_pp =
        network->addElementWise(*attn_x->getOutput(0), *pe->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    nvinfer1::ILayer* proj = Conv(network, weightMap, *x_add_pp->getOutput(0), dim, lname + ".proj", 1, 1, 0, 1, false);

    return proj;
}




nvinfer1::IElementWiseLayer* ABlock(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
    int dim, int num_heads, std::string lname, float mlp_ratio, int area) {

    nvinfer1::ILayer* attn = AAttn(network, weightMap, input, dim, num_heads, lname + ".attn", area);
    nvinfer1::IElementWiseLayer* add1 = // x = x + self.attn(x)
        network->addElementWise(input, *attn->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    int mlp_hidden_dim = int(dim * mlp_ratio);

    nvinfer1::ILayer* mlp_0 =
        Conv(network, weightMap, *add1->getOutput(0), mlp_hidden_dim, lname + ".mlp.0", 1, 1, 0, 1, true);
    nvinfer1::ILayer* mlp_1 =
        Conv(network, weightMap, *mlp_0->getOutput(0), dim, lname + ".mlp.1", 1, 1, 0, 1, false);

    nvinfer1::IElementWiseLayer* result =
        network->addElementWise(*add1->getOutput(0), *mlp_1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    return result;

}


nvinfer1::ILayer* A2C2f(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
    nvinfer1::ITensor& input, int c2, int n, std::string lname, bool a2, int area, bool residual,
    float mlp_ratio, float e, int g, bool shortcut) {

    int c_ = static_cast<int>(c2 * e);
    assert(c_ % 32 == 0 && "Dimension of ABlock must be a multiple of 32");
    int num_heads = c_ / 32;

    nvinfer1::ILayer* cv1 = Conv(network, weightMap, input, c_, lname + ".cv1", 1, 1);
    std::vector<nvinfer1::ITensor*> y{ cv1->getOutput(0) };
    nvinfer1::ITensor* current = cv1->getOutput(0);

    for (int i = 0; i < n; i++) {
        if (a2) {
            nvinfer1::ILayer* m_0 = ABlock(network, weightMap, *current, c_, num_heads,
                lname + ".m." + std::to_string(i) + ".0", mlp_ratio, area);
            nvinfer1::ILayer* m_1 = ABlock(network, weightMap, *m_0->getOutput(0), c_, num_heads,
                lname + ".m." + std::to_string(i) + ".1", mlp_ratio, area);
            current = m_1->getOutput(0);
        }
        else {
            nvinfer1::ILayer* m =
                C3k(network, weightMap, *current, c_, lname + ".m." + std::to_string(i), 2, shortcut, g);
            current = m->getOutput(0);
        }
        y.push_back(current);
    }
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(y.data(), static_cast<int>(y.size()));
    cat->setAxis(1);
    nvinfer1::ILayer* cv2 = Conv(network, weightMap, *cat->getOutput(0), c2, lname + ".cv2", 1, 1);

    if (a2 && residual) {
        std::cout << lname << " applying residual connection with gamma" << std::endl;
        nvinfer1::Weights gamma = weightMap[lname + ".gamma"];
        nvinfer1::IConstantLayer* gamma_layer = network->addConstant(nvinfer1::Dims4{ 1, c2, 1, 1 }, gamma);
        nvinfer1::IElementWiseLayer* scaled_output = network->addElementWise(
            *gamma_layer->getOutput(0), *cv2->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        nvinfer1::IElementWiseLayer* result =
            network->addElementWise(input, *scaled_output->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return result;
    }
    else {
        return cv2;
    }
}
