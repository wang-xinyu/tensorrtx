#include "block.h"
#include "calibrator.h"
#include "config.h"
#include "yololayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

using namespace nvinfer1;
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
void PrintDim(const ILayer* layer, std::string log) {
    Dims dim = layer->getOutput(0)->getDimensions();
    std::cout << log << ": "
              << "\t\t\t\t";
    for (int i = 0; i < dim.nbDims; i++) {
        std::cout << dim.d[i] << " ";
    }
    std::cout << std::endl;
}

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

int get_width(int x, float gw, int divisor) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}
static nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                             std::map<std::string, nvinfer1::Weights> weightMap,
                                             nvinfer1::ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* output = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(output);
    return output;
}
nvinfer1::ILayer* convBnSiLU(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                             nvinfer1::ITensor& input, int ch, int k, int s, int p, std::string lname, int g) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});
    conv->setNbGroups(g);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid);
    auto ew = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}
nvinfer1::ILayer* convBnNoAct(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int ch,
                              int k, int s, int p, std::string lname, int g) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});
    conv->setNbGroups(g);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    return bn;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = kNumAnchor * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto* p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

ILayer* RepConvN(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2,
                 int k, int s, int p, int g, int d, bool act, bool bn, bool deploy, std::string lname) {
    assert(k == 3 && p == 1);
    ILayer* conv1 = convBnNoAct(network, weightMap, input, c2, k, s, p, lname + ".conv1", g);
    ILayer* conv2 = convBnNoAct(network, weightMap, input, c2, 1, s, p - k / 2, lname + ".conv2", g);
    ILayer* ew0 = network->addElementWise(*conv1->getOutput(0), *conv2->getOutput(0), ElementWiseOperation::kSUM);
    nvinfer1::IActivationLayer* sigmoid =
            network->addActivation(*ew0->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid);

    auto ew =
            network->addElementWise(*ew0->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer* RepNBottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                       int c2, bool shortcut, int k, int g, float e, std::string lname) {
    int c_ = int(c2 * e);
    assert(k == 3 && "RepVGG only support kernel size 3");
    auto cv1 = RepConvN(network, weightMap, input, c1, c_, k, 1, 1, g, 1, true, false, false, lname + ".cv1");
    auto cv2 = convBnSiLU(network, weightMap, *cv1->getOutput(0), c2, k, 1, 1, lname + ".cv2", g);
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* RepNCSP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2,
                int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = int(c2 * e);

    auto cv1 = convBnSiLU(network, weightMap, input, c_, 1, 1, 0, lname + ".cv1", 1);

    ILayer* m = cv1;
    for (int i = 0; i < n; i++) {
        m = RepNBottleneck(network, weightMap, *m->getOutput(0), c_, c_, shortcut, 3, g, 1.0,
                           lname + ".m." + std::to_string(i));
    }

    // auto m_0 = RepNBottleneck(network, weightMap, *cv1->getOutput(0), c_, c_, shortcut, 3, g, 1.0, lname + ".m.0");
    // auto m_1 = RepNBottleneck(network, weightMap, *m_0->getOutput(0), c_, c_, shortcut, 3, g, 1.0, lname + ".m.1");

    auto cv2 = convBnSiLU(network, weightMap, input, c_, 1, 1, 0, lname + ".cv2", 1);
    ITensor* inputTensors[] = {m->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname + ".cv3", 1);
    return cv3;
}

ILayer* ELAN1(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2,
              int c3, int c4, std::string lname) {
    auto cv1 = convBnSiLU(network, weightMap, input, c3, 1, 1, 0, lname + ".cv1", 1);
    // chunk(2, 1)

    nvinfer1::Dims d = cv1->getOutput(0)->getDimensions();
    nvinfer1::ISliceLayer* split1 =
            network->addSlice(*cv1->getOutput(0), nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{d.d[0] / 2, d.d[1], d.d[2]},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split2 =
            network->addSlice(*cv1->getOutput(0), nvinfer1::Dims3{d.d[0] / 2, 0, 0},
                              nvinfer1::Dims3{d.d[0] / 2, d.d[1], d.d[2]}, nvinfer1::Dims3{1, 1, 1});
    auto cv2 = convBnSiLU(network, weightMap, *split2->getOutput(0), c4, 3, 1, 1, lname + ".cv2", 1);

    auto cv3 = convBnSiLU(network, weightMap, *cv2->getOutput(0), c4, 3, 1, 1, lname + ".cv3", 1);

    ITensor* inputTensors[] = {split1->getOutput(0), split2->getOutput(0), cv2->getOutput(0), cv3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv4 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname + ".cv4", 1);
    return cv4;
}

ILayer* RepNCSPELAN4(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                     int c2, int c3, int c4, int c5, std::string lname) {

    auto cv1 = convBnSiLU(network, weightMap, input, c3, 1, 1, 0, lname + ".cv1", 1);
    // chunk(2, 1)

    nvinfer1::Dims d = cv1->getOutput(0)->getDimensions();
    nvinfer1::ISliceLayer* split1 =
            network->addSlice(*cv1->getOutput(0), nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{d.d[0] / 2, d.d[1], d.d[2]},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split2 =
            network->addSlice(*cv1->getOutput(0), nvinfer1::Dims3{d.d[0] / 2, 0, 0},
                              nvinfer1::Dims3{d.d[0] / 2, d.d[1], d.d[2]}, nvinfer1::Dims3{1, 1, 1});

    auto cv2_0 = RepNCSP(network, weightMap, *split2->getOutput(0), c3 / 2, c4, c5, true, 1, 0.5, lname + ".cv2.0");
    auto cv2_1 = convBnSiLU(network, weightMap, *cv2_0->getOutput(0), c4, 3, 1, 1, lname + ".cv2.1", 1);

    auto cv3_0 = RepNCSP(network, weightMap, *cv2_1->getOutput(0), c4, c4, c5, true, 1, 0.5, lname + ".cv3.0");
    auto cv3_1 = convBnSiLU(network, weightMap, *cv3_0->getOutput(0), c4, 3, 1, 1, lname + ".cv3.1", 1);

    ITensor* inputTensors[] = {split1->getOutput(0), split2->getOutput(0), cv2_1->getOutput(0), cv3_1->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv4 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname + ".cv4", 1);
    return cv4;
}

ILayer* AConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2,
              std::string lname) {
    auto pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{2, 2});
    pool->setStrideNd(DimsHW{1, 1});
    pool->setPaddingNd(DimsHW{0, 0});
    auto cv1 = convBnSiLU(network, weightMap, *pool->getOutput(0), c2, 3, 2, 1, lname + ".cv1", 1);
    return cv1;
}
ILayer* ADown(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2,
              std::string lname) {
    int c_ = c2 / 2;
    auto pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{2, 2});
    pool->setStrideNd(DimsHW{1, 1});
    pool->setPaddingNd(DimsHW{0, 0});

    nvinfer1::Dims d = pool->getOutput(0)->getDimensions();
    nvinfer1::ISliceLayer* split1 =
            network->addSlice(*pool->getOutput(0), nvinfer1::Dims3{0, 0, 0},
                              nvinfer1::Dims3{d.d[0] / 2, d.d[1], d.d[2]}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split2 =
            network->addSlice(*pool->getOutput(0), nvinfer1::Dims3{d.d[0] / 2, 0, 0},
                              nvinfer1::Dims3{d.d[0] / 2, d.d[1], d.d[2]}, nvinfer1::Dims3{1, 1, 1});

    // auto chunklayer = layer_split(1, pool->getOutput(0), network);
    auto cv1 = convBnSiLU(network, weightMap, *split1->getOutput(0), c_, 3, 2, 1, lname + ".cv1", 1);

    auto pool2 = network->addPoolingNd(*split2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool2->setStrideNd(DimsHW{2, 2});
    pool2->setPaddingNd(DimsHW{1, 1});
    auto cv2 = convBnSiLU(network, weightMap, *pool2->getOutput(0), c_, 1, 1, 0, lname + ".cv2", 1);

    ITensor* inputTensors[] = {cv1->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);
    return cat;
}

std::vector<ILayer*> CBLinear(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              std::vector<int> c2s, int k, int s, int p, int g, std::string lname) {

    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, std::accumulate(c2s.begin(), c2s.end(), 0), DimsHW{k, k},
                                      weightMap[lname + ".conv.weight"], weightMap[lname + ".conv.bias"]);
    assert(conv1);
    conv1->setName((lname + ".conv").c_str());
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    int h = input.getDimensions().d[1];
    int w = input.getDimensions().d[2];
    std::vector<ILayer*> slices(c2s.size());
    int start = 0;
    for (int i = 0; i < c2s.size(); i++) {
        slices[i] = network->addSlice(*conv1->getOutput(0), Dims3{start, 0, 0}, Dims3{c2s[i], h, w}, Dims3{1, 1, 1});
        start += c2s[i];
    }
    return slices;
}

ILayer* CBFuse(INetworkDefinition* network, std::vector<std::vector<ILayer*>> input, std::vector<int> idx,
               std::vector<int> strides) {
    ILayer** res = new ILayer*[input.size()];
    res[input.size() - 1] = input[input.size() - 1][0];

    for (int i = input.size() - 2; i >= 0; i--) {
        auto upsample = network->addResize(*input[i][idx[i]]->getOutput(0));
        upsample->setResizeMode(ResizeMode::kNEAREST);
        const float scales[] = {1, strides[i] / strides[strides.size() - 1], strides[i] / strides[strides.size() - 1]};
        upsample->setScales(scales, 3);
        res[i] = upsample;
    }

    for (int i = 1; i < input.size(); i++) {
        auto ew = network->addElementWise(*res[0]->getOutput(0), *res[i]->getOutput(0), ElementWiseOperation::kSUM);
        res[0] = ew;
    }
    return res[0];
}

ILayer* SP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int k, int s) {
    int p = k / 2;
    auto pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{k, k});
    pool->setPaddingNd(DimsHW{p, p});
    pool->setStrideNd(DimsHW{s, s});
    return pool;
}

ILayer* SPPELAN(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2,
                int c3, std::string lname) {
    auto cv1 = convBnSiLU(network, weightMap, input, c3, 1, 1, 0, lname + ".cv1", 1);
    auto cv2 = SP(network, weightMap, *cv1->getOutput(0), 5, 1);
    auto cv3 = SP(network, weightMap, *cv2->getOutput(0), 5, 1);
    auto cv4 = SP(network, weightMap, *cv3->getOutput(0), 5, 1);

    ITensor* inputTensors[] = {cv1->getOutput(0), cv2->getOutput(0), cv3->getOutput(0), cv4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv5 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname + ".cv5", 1);
    return cv5;
}

ILayer* DetectBbox_Conv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2,
                        int reg_max, std::string lname) {
    auto cv_0 = convBnSiLU(network, weightMap, input, c2, 3, 1, 1, lname + ".0", 1);
    auto cv_1 = convBnSiLU(network, weightMap, *cv_0->getOutput(0), c2, 3, 1, 1, lname + ".1", 4);
    auto cv_2 = network->addConvolutionNd(*cv_1->getOutput(0), reg_max * 4, DimsHW{1, 1},
                                          weightMap[lname + ".2.weight"], weightMap[lname + ".2.bias"]);
    cv_2->setName((lname + ".conv").c_str());
    cv_2->setStrideNd(DimsHW{1, 1});
    cv_2->setPaddingNd(DimsHW{0, 0});
    cv_2->setNbGroups(4);
    return cv_2;
}

ILayer* DetectCls_Conv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2,
                       int cls, std::string lname) {
    auto cv_0 = convBnSiLU(network, weightMap, input, c2, 3, 1, 1, lname + ".0", 1);
    auto cv_1 = convBnSiLU(network, weightMap, *cv_0->getOutput(0), c2, 3, 1, 1, lname + ".1", 1);
    auto cv_2 = network->addConvolutionNd(*cv_1->getOutput(0), cls, DimsHW{1, 1}, weightMap[lname + ".2.weight"],
                                          weightMap[lname + ".2.bias"]);
    cv_2->setName((lname + ".conv").c_str());
    cv_2->setStrideNd(DimsHW{1, 1});
    cv_2->setPaddingNd(DimsHW{0, 0});
    return cv_2;
}

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int k, int s, int p, std::string lname) {
    auto dim = input.getDimensions();
    int c = dim.d[0];
    int grid = dim.d[1] * dim.d[2];
    int split_num = c / ch;

    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims3{split_num, ch, grid});
    shuffle1->setSecondTranspose(nvinfer1::Permutation{1, 0, 2});
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuffle1->getOutput(0));
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*softmax->getOutput(0), 1, nvinfer1::DimsHW{1, 1},
                                                                  weightMap[lname + ".conv.weight"], bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});
    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setReshapeDimensions(nvinfer1::Dims2{4, grid});
    return shuffle2;
}

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                       std::vector<nvinfer1::IConcatenationLayer*> dets, bool is_segmentation) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");

    nvinfer1::PluginField plugin_fields[1];
    int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox, is_segmentation};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 5;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = nvinfer1::PluginFieldType::kFLOAT32;

    nvinfer1::PluginFieldCollection plugin_data;
    plugin_data.nbFields = 1;
    plugin_data.fields = plugin_fields;
    nvinfer1::IPluginV2* plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto det : dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}

std::vector<IConcatenationLayer*> DualDDetect(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                              std::vector<ILayer*> dets, int cls, std::vector<int> ch,
                                              std::string lname) {
    int c2 = std::max(int(ch[0] / 4), int(16 * 4));
    int c3 = std::max(ch[0], std::min(cls * 2, 128));
    int reg_max = 16;

    std::vector<ILayer*> bboxlayers;
    std::vector<ILayer*> clslayers;

    for (int i = 0; i < dets.size(); i++) {
        // Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)
        bboxlayers.push_back(DetectBbox_Conv(network, weightMap, *dets[i]->getOutput(0), c2, reg_max,
                                             lname + ".cv2." + std::to_string(i)));
        // Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nc, 1)
        auto cls_layer = DetectCls_Conv(network, weightMap, *dets[i]->getOutput(0), c3, cls,
                                        lname + ".cv3." + std::to_string(i));
        auto dim = cls_layer->getOutput(0)->getDimensions();
        nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*cls_layer->getOutput(0));
        shuffle->setReshapeDimensions(nvinfer1::Dims2{kNumClass, dim.d[1] * dim.d[2]});
        clslayers.push_back(shuffle);
    }

    std::vector<IConcatenationLayer*> ret;
    for (int i = 0; i < dets.size(); i++) {
        // softmax 16*4, w, h => 16, 4, w, h
        auto loc = DFL(network, weightMap, *bboxlayers[i]->getOutput(0), 16, 1, 1, 0, lname + ".dfl");
        nvinfer1::ITensor* inputTensor[] = {loc->getOutput(0), clslayers[i]->getOutput(0)};
        ret.push_back(network->addConcatenation(inputTensor, 2));
    }
    return ret;
}

std::vector<IConcatenationLayer*> DDetect(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                          std::vector<ILayer*> dets, int cls, std::vector<int> ch, std::string lname) {
    int c2 = std::max(int(ch[0] / 4), int(16 * 4));
    //  max((ch[0], min((self.nc * 2, 128))))
    // int c3 = std::max(ch[0], std::min(cls * 2, 128));
    int c3 = std::max(ch[0], std::min(cls, 128));
    int reg_max = 16;

    std::vector<ILayer*> bboxlayers;
    std::vector<ILayer*> clslayers;

    for (int i = 0; i < dets.size(); i++) {
        // Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)
        bboxlayers.push_back(DetectBbox_Conv(network, weightMap, *dets[i]->getOutput(0), c2, reg_max,
                                             lname + ".cv2." + std::to_string(i)));
        // Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nc, 1)
        auto cls_layer = DetectCls_Conv(network, weightMap, *dets[i]->getOutput(0), c3, cls,
                                        lname + ".cv3." + std::to_string(i));
        auto dim = cls_layer->getOutput(0)->getDimensions();
        nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*cls_layer->getOutput(0));
        shuffle->setReshapeDimensions(nvinfer1::Dims2{kNumClass, dim.d[1] * dim.d[2]});
        clslayers.push_back(shuffle);
    }

    std::vector<IConcatenationLayer*> ret;
    for (int i = 0; i < dets.size(); i++) {
        // softmax 16*4, w, h => 16, 4, w, h
        auto loc = DFL(network, weightMap, *bboxlayers[i]->getOutput(0), 16, 1, 1, 0, lname + ".dfl");
        nvinfer1::ITensor* inputTensor[] = {loc->getOutput(0), clslayers[i]->getOutput(0)};
        ret.push_back(network->addConcatenation(inputTensor, 2));
    }
    return ret;
}
