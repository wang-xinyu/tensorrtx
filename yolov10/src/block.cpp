#include "block.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "config.h"
#include "yololayer.h"

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> WeightMap;

    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
    }
    return WeightMap;
}

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                      std::string lname, float eps) {
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

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                        int ch, int k, int s, std::string lname, int g) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    int p = k / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});
    conv->setNbGroups(g);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew =
            network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

nvinfer1::ILayer* convBn(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, int k, int s, std::string lname, int g = 1) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    int p = k / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});
    conv->setNbGroups(g);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    return bn;
}

nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int c1, int c2, bool shortcut, float e, std::string lname) {
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c2, 3, 1, lname + ".cv1");
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *conv1->getOutput(0), c2, 3, 1, lname + ".cv2");

    if (shortcut && c1 == c2) {
        nvinfer1::IElementWiseLayer* ew =
                network->addElementWise(input, *conv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    }
    return conv2;
}

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                 int c2, int n, bool shortcut, float e, std::string lname) {
    int c_ = (float)c2 * e;

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, 2 * c_, 1, 1, lname + ".cv1");
    nvinfer1::Dims d = conv1->getOutput(0)->getDimensions();

    nvinfer1::ISliceLayer* split1 =
            network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                              nvinfer1::Dims4{d.d[0], d.d[1] / 2, d.d[2], d.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* split2 =
            network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, d.d[1] / 2, 0, 0},
                              nvinfer1::Dims4{d.d[0], d.d[1] / 2, d.d[2], d.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ITensor* inputTensor0[] = {split1->getOutput(0), split2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor0, 2);
    nvinfer1::ITensor* y1 = split2->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto* b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);

        nvinfer1::ITensor* inputTensors[] = {cat->getOutput(0), b->getOutput(0)};
        cat = network->addConcatenation(inputTensors, 2);
    }

    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, lname + ".cv2");

    return conv2;
}

nvinfer1::IElementWiseLayer* C2(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c1,
                                int c2, int n, bool shortcut, float e, std::string lname) {
    assert(network != nullptr);
    int hidden_channels = static_cast<int>(c2 * e);

    // cv1 branch
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, input, 2 * hidden_channels, 1, 1, lname + ".cv1");
    nvinfer1::ITensor* cv1_out = conv1->getOutput(0);

    // Split the output of cv1 into two tensors
    nvinfer1::Dims dims = cv1_out->getDimensions();
    nvinfer1::ISliceLayer* split1 = network->addSlice(*cv1_out, nvinfer1::Dims4{0, 0, 0, 0},
                                                      nvinfer1::Dims4{dims.d[0], dims.d[1] / 2, dims.d[2], dims.d[3]},
                                                      nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* split2 = network->addSlice(*cv1_out, nvinfer1::Dims4{0, dims.d[1] / 2, 0, 0},
                                                      nvinfer1::Dims4{dims.d[0], dims.d[1] / 2, dims.d[2], dims.d[3]},
                                                      nvinfer1::Dims4{1, 1, 1, 1});

    // Create y1 bottleneck sequence
    nvinfer1::ITensor* y1 = split1->getOutput(0);
    for (int i = 0; i < n; ++i) {
        auto* bottleneck_layer = bottleneck(network, weightMap, *y1, hidden_channels, hidden_channels, shortcut, 1.0,
                                            lname + ".m." + std::to_string(i));
        y1 = bottleneck_layer->getOutput(0);  // update 'y1' to be the output of the current bottleneck
    }

    // Concatenate y1 with the second split of cv1
    nvinfer1::ITensor* concatInputs[2] = {y1, split2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(concatInputs, 2);

    // cv2 to produce the final output
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, lname + ".cv2");

    return conv2;
}

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int k, std::string lname) {
    int c_ = c1 / 2;
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c_, 1, 1, lname + ".cv1");
    nvinfer1::IPoolingLayer* pool1 =
            network->addPoolingNd(*conv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k, k});
    pool1->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k / 2, k / 2});
    nvinfer1::IPoolingLayer* pool2 =
            network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k, k});
    pool2->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool2->setPaddingNd(nvinfer1::DimsHW{k / 2, k / 2});
    nvinfer1::IPoolingLayer* pool3 =
            network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k, k});
    pool3->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool3->setPaddingNd(nvinfer1::DimsHW{k / 2, k / 2});
    nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0),
                                         pool3->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 4);
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, lname + ".cv2");
    return conv2;
}

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname) {

    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims4{kBatchSize, 4, 16, grid});
    shuffle1->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuffle1->getOutput(0));
    softmax->setAxes(1 << 1);

    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(*softmax->getOutput(0), 1, nvinfer1::DimsHW{1, 1}, weightMap[lname], bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});

    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setReshapeDimensions(nvinfer1::Dims3{kBatchSize, 4, grid});

    return shuffle2;
}

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network, std::vector<nvinfer1::ILayer*> dets,
                                       const int* px_arry, int px_arry_num) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const int netinfo_count = 4;  // Assuming the first 5 elements are for netinfo as per existing code.
    const int total_count = netinfo_count + px_arry_num;  // Total number of elements for netinfo and px_arry combined.

    std::vector<int> combinedInfo(total_count);
    // Fill in the first 5 elements as per existing netinfo.
    combinedInfo[0] = kNumClass;
    combinedInfo[1] = kInputW;
    combinedInfo[2] = kInputH;
    combinedInfo[3] = kMaxNumOutputBbox;

    // Copy the contents of px_arry into the combinedInfo vector after the initial 5 elements.
    std::copy(px_arry, px_arry + px_arry_num, combinedInfo.begin() + netinfo_count);

    // Now let's create the PluginField object to hold this combined information.
    nvinfer1::PluginField pluginField;
    pluginField.name = "combinedInfo";  // This can be any name that the plugin will recognize
    pluginField.data = combinedInfo.data();
    pluginField.type = nvinfer1::PluginFieldType::kINT32;
    pluginField.length = combinedInfo.size();

    // Create the PluginFieldCollection to hold the PluginField object.
    nvinfer1::PluginFieldCollection pluginFieldCollection{};
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

nvinfer1::ILayer* SCDown(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, int k, int s, std::string lname) {
    auto* conv1 = convBnSiLU(network, weightMap, input, ch, 1, 1, lname + ".cv1");

    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), ch, nvinfer1::DimsHW{k, k},
                                                                   weightMap[lname + ".cv2.conv.weight"], bias_empty);
    assert(conv2);
    conv2->setStrideNd(nvinfer1::DimsHW{s, s});
    int p = k / 2;
    conv2->setPaddingNd(nvinfer1::DimsHW{p, p});
    conv2->setNbGroups(ch);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".cv2.bn", 1e-3);
    assert(bn);
    return bn;
}

nvinfer1::ILayer* Attention(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                            nvinfer1::ITensor& input, int dim, int num_heads, float attn_ratio, std::string lname) {
    int head_dim = dim / num_heads;
    int key_dim = head_dim * attn_ratio;
    float scale = pow(key_dim, -0.5);
    int nh_kd = key_dim * num_heads;
    int h = dim + nh_kd * 2;

    auto d = input.getDimensions();
    int B = d.d[0];
    int H = d.d[2];
    int W = d.d[3];
    int N = H * W;
    auto* qkv = convBn(network, weightMap, input, h, 1, 1, lname + ".qkv");
    // qkv.view(B, self.num_heads, -1, N)
    auto shuffle = network->addShuffle(*qkv->getOutput(0));
    shuffle->setReshapeDimensions(nvinfer1::Dims4{B, num_heads, -1, N});
    // q, k, v = .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    auto d1 = shuffle->getOutput(0)->getDimensions();
    auto q = network->addSlice(*shuffle->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                               nvinfer1::Dims4{d1.d[0], d1.d[1], key_dim, d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    auto k = network->addSlice(*shuffle->getOutput(0), nvinfer1::Dims4{0, 0, key_dim, 0},
                               nvinfer1::Dims4{d1.d[0], d1.d[1], key_dim, d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    auto v = network->addSlice(*shuffle->getOutput(0), nvinfer1::Dims4{0, 0, key_dim * 2, 0},
                               nvinfer1::Dims4{d1.d[0], d1.d[1], head_dim, d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    // attn = ((q.transpose(-2, -1) @ k) * self.scale)
    auto qT = network->addShuffle(*q->getOutput(0));
    qT->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});
    auto matmul = network->addMatrixMultiply(*qT->getOutput(0), nvinfer1::MatrixOperation::kNONE, *k->getOutput(0),
                                             nvinfer1::MatrixOperation::kNONE);
    // There are not many memory leaks, and I will change it when I have time
    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    scale_val[0] = scale;
    nvinfer1::Weights s_w{nvinfer1::DataType::kFLOAT, scale_val, 1};
    float* shift_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shift_val[0] = 0;
    nvinfer1::Weights sh_w{nvinfer1::DataType::kFLOAT, shift_val, 1};
    float* power_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    power_val[0] = 1;
    nvinfer1::Weights p_w{nvinfer1::DataType::kFLOAT, power_val, 1};
    nvinfer1::IScaleLayer* scaleLayer =
            network->addScale(*matmul->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, sh_w, s_w, p_w);
    // attn = attn.softmax(dim=-1)
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*scaleLayer->getOutput(0));
    softmax->setAxes(1 << 3);
    // x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) + self.pe(v.reshape(B, -1, H, W))
    auto attnT = network->addShuffle(*softmax->getOutput(0));
    attnT->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});
    auto matmul2 = network->addMatrixMultiply(*v->getOutput(0), nvinfer1::MatrixOperation::kNONE, *attnT->getOutput(0),
                                              nvinfer1::MatrixOperation::kNONE);
    auto reshape = network->addShuffle(*matmul2->getOutput(0));
    reshape->setReshapeDimensions(nvinfer1::Dims4{B, -1, H, W});
    auto v_reshape = network->addShuffle(*v->getOutput(0));
    v_reshape->setReshapeDimensions(nvinfer1::Dims4{B, -1, H, W});
    // self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
    auto pe = convBn(network, weightMap, *v_reshape->getOutput(0), dim, 3, 1, lname + ".pe", dim);
    auto sum = network->addElementWise(*reshape->getOutput(0), *pe->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    // x = self.proj(x)
    // self.proj = Conv(dim, dim, 1, act=False)
    auto proj = convBn(network, weightMap, *sum->getOutput(0), dim, 1, 1, lname + ".proj");
    return proj;
}

nvinfer1::ILayer* PSA(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                      nvinfer1::ITensor& input, int ch, std::string lname) {
    int c = int(ch * 0.5);
    auto conv1 = convBnSiLU(network, weightMap, input, c * 2, 1, 1, lname + ".cv1");
    // a, b = split((self.c, self.c), dim=1)
    auto d1 = conv1->getOutput(0)->getDimensions();
    auto a = network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                               nvinfer1::Dims4{d1.d[0], c, d1.d[2], d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    auto b = network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, c, 0, 0},
                               nvinfer1::Dims4{d1.d[0], c, d1.d[2], d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    // b = b + self.attn(b)
    auto attn = Attention(network, weightMap, *b->getOutput(0), c, c / 64, 0.5f, lname + ".attn");
    auto sum = network->addElementWise(*b->getOutput(0), *attn->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    // b = b + self.ffn(b)
    // self.ffn = nn.Sequential(
    //			Conv(self.c, self.c * 2, 1),
    //			Conv(self.c * 2, self.c, 1, act=False)
    //		)
    auto ffn1 = convBnSiLU(network, weightMap, *sum->getOutput(0), c * 2, 1, 1, lname + ".ffn.0");
    auto ffn2 = convBn(network, weightMap, *ffn1->getOutput(0), c, 1, 1, lname + ".ffn.1");
    auto sum2 = network->addElementWise(*sum->getOutput(0), *ffn2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    // self.cv2(torch.cat((a, b), 1))
    nvinfer1::ITensor* inputTensors[] = {a->getOutput(0), sum2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 2);
    auto conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), ch, 1, 1, lname + ".cv2");
    return conv2;
}

nvinfer1::ILayer* RepVGGDW(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                           nvinfer1::ITensor& input, int ch, std::string lname) {
    // self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
    // self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
    // self.dim = ed
    // self.act = nn.SiLU()
    // return self.act(self.conv(x) + self.conv1(x))
    auto conv = convBn(network, weightMap, input, ch, 7, 1, lname + ".conv", ch);
    auto conv1 = convBn(network, weightMap, input, ch, 3, 1, lname + ".conv1", ch);
    auto ew = network->addElementWise(*conv->getOutput(0), *conv1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    auto sigmoid = network->addActivation(*ew->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    auto ew_silu =
            network->addElementWise(*ew->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew_silu);
    return ew_silu;
}

nvinfer1::ILayer* CIB(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                      nvinfer1::ITensor& input, int c1, int c2, bool shortcut, float e, bool lk, std::string lname) {
    // self.cv1 = nn.Sequential(
    //			Conv(c1, c1, 3, g=c1),
    //			Conv(c1, 2 * c_, 1),
    //			Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
    //			Conv(2 * c_, c2, 1),
    //			Conv(c2, c2, 3, g=c2),
    //		)
    int c_ = (float)c2 * e;
    auto* conv1 = convBnSiLU(network, weightMap, input, c1, 3, 1, lname + ".cv1.0", c1);
    auto* conv2 = convBnSiLU(network, weightMap, *conv1->getOutput(0), 2 * c_, 1, 1, lname + ".cv1.1");
    nvinfer1::ILayer* conv3;
    if (!lk) {
        conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0), 2 * c_, 3, 1, lname + ".cv1.2", 2 * c_);
    } else {
        conv3 = RepVGGDW(network, weightMap, *conv2->getOutput(0), 2 * c_, lname + ".cv1.2");
    }
    auto* conv4 = convBnSiLU(network, weightMap, *conv3->getOutput(0), c2, 1, 1, lname + ".cv1.3");
    auto* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0), c2, 3, 1, lname + ".cv1.4", c2);
    if (shortcut && c1 == c2) {
        auto* ew = network->addElementWise(input, *conv5->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    } else {
        return conv5;
    }
}

nvinfer1::ILayer* C2fCIB(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int c1, int c2, int n, bool shortcut, bool lk, float e,
                         std::string lname) {
    int c_ = (float)c2 * e;

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, 2 * c_, 1, 1, lname + ".cv1");
    nvinfer1::Dims d = conv1->getOutput(0)->getDimensions();

    nvinfer1::ISliceLayer* split1 =
            network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                              nvinfer1::Dims4{d.d[0], d.d[1] / 2, d.d[2], d.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* split2 =
            network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, d.d[1] / 2, 0, 0},
                              nvinfer1::Dims4{d.d[0], d.d[1] / 2, d.d[2], d.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ITensor* inputTensor0[] = {split1->getOutput(0), split2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor0, 2);
    nvinfer1::ITensor* y1 = split2->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto* b = CIB(network, weightMap, *y1, c_, c_, shortcut, 1.0, lk, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);

        nvinfer1::ITensor* inputTensors[] = {cat->getOutput(0), b->getOutput(0)};
        cat = network->addConcatenation(inputTensors, 2);
    }

    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, lname + ".cv2");

    return conv2;
}
