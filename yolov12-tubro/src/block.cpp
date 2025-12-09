#include "block.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "config.h"
#include "model.h"
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
        // std::cout << "===========name:              " << name << std::endl;
    }
    return WeightMap;
}

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                      std::string lname, float eps) {
    // cout << "BatchNorm's name :             " << lname << endl;
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
                                        int ch, std::vector<int> k, int s, std::string lname, int p, int g, int d) {
  
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k[0], k[1]},
                                                                  weightMap[lname + ".conv.weight"], bias_empty);

    conv->setNbGroups(g);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    // auto pad
    int p0 = k[0] / 2;
    int p1 = k[1] / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p0, p1});
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
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, c_out, nvinfer1::DimsHW{k, k},
                                                                  weightMap[lname + ".conv.weight"], emptywts);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    // auto pad
    int p0 = k / 2;
    int p1 = k / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p0, p1});
    conv->setNbGroups(g);

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    if (act) {
        nvinfer1::IActivationLayer* sigmoid =
                network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
        nvinfer1::IElementWiseLayer* ew = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0),
                                                                  nvinfer1::ElementWiseOperation::kPROD);
        assert(ew);
        return ew;
    } else
        return bn;
}



nvinfer1::ILayer* DWConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, std::vector<int> k, int s, std::string lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k[0], k[1]},
                                                                  weightMap[lname + ".conv.weight"], bias_empty);

    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setNbGroups(ch);
    // auto pad
    int p0 = k[0] / 2;
    int p1 = k[1] / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p0, p1});

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
    
    nvinfer1::IElementWiseLayer* cv1 = convBnSiLU(network, weightMap, input, c_, {1, 1}, 1, lname + ".cv1");
    nvinfer1::IElementWiseLayer* cv2 = convBnSiLU(network, weightMap, input, c_, {1, 1}, 1, lname + ".cv2");
    nvinfer1::ITensor* y = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        nvinfer1::ILayer* b = bottleneck(network, weightMap, *y, c_, c_, shortcut, {k, k}, {k, k}, 1.0, g,
                                          lname + ".m." + std::to_string(i));
        y = b->getOutput(0);
    }
    nvinfer1::ITensor* inputTensor[] = {y, cv2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor, 2);
    nvinfer1::IElementWiseLayer* cv3 =
            convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv3");

    return cv3;
}


nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
    std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c2,
                                  int n, std::string lname, bool c3k , float e, int g, bool shortcut) {
    int c = int(c2 * float(e));
    nvinfer1::ILayer* cv1 = Conv(network, weightMap, input, 2 * c, lname + ".cv1", 1, 1);
    nvinfer1::ISliceLayer* sl0 = network->addSlice(*cv1->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0}, 
        nvinfer1::Dims4{cv1->getOutput(0)->getDimensions().d[0],
        cv1->getOutput(0)->getDimensions().d[1] / 2, cv1->getOutput(0)->getDimensions().d[2],
        cv1->getOutput(0)->getDimensions().d[3]}, nvinfer1::Dims4 {1, 1, 1, 1});
    nvinfer1::ISliceLayer* sl1 = network->addSlice(
            *cv1->getOutput(0), nvinfer1::Dims4{0, cv1->getOutput(0)->getDimensions().d[1] / 2, 0, 0},
            nvinfer1::Dims4{cv1->getOutput(0)->getDimensions().d[0], cv1->getOutput(0)->getDimensions().d[1] / 2,
                            cv1->getOutput(0)->getDimensions().d[2], cv1->getOutput(0)->getDimensions().d[3]},
            nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ITensor *inputTensor0[] = {sl0->getOutput(0), sl1->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat = network->addConcatenation(inputTensor0, 2);
    nvinfer1::ITensor* current = sl1->getOutput(0);
   
    for (int i = 0; i < n; i++) {
        nvinfer1::ILayer *b;
        if (c3k) {
            b = C3k(network, weightMap, *current, c, lname + ".m." + std::to_string(i), 2, shortcut, g);
        } else {
            b = bottleneck(network, weightMap, *current, c, c, shortcut, {3, 3}, {3, 3}, 0.5, g,
                                              lname + ".m." + std::to_string(i));
        }
        current = b->getOutput(0);
        nvinfer1::ITensor* inputTensors[] = {cat->getOutput(0), b->getOutput(0)};
        cat = network->addConcatenation(inputTensors, 2);
    }
    nvinfer1::IElementWiseLayer* cv2 =
            convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv2");
    return cv2;
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
    qk_flatten_t->setReshapeDimensions(nvinfer1::Dims3{B, -1, N});
    qk_flatten_t->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});

    nvinfer1::ILayer* v = Conv(network, weightMap, input, all_head_dim, lname + ".v", 1, 1, 0, 1, false);
    nvinfer1::IShuffleLayer* v_flatten_t = network->addShuffle(*v->getOutput(0));
    v_flatten_t->setReshapeDimensions(nvinfer1::Dims3{B, -1, N});
    v_flatten_t->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});  // (1, 6400, 64)

    nvinfer1::ILayer* pe = Conv(network, weightMap, *v->getOutput(0), dim, lname + ".pe", 5, 1, 2, dim, false);

    nvinfer1::ITensor* q_k = qk_flatten_t->getOutput(0);
    nvinfer1::ITensor* v_ = v_flatten_t->getOutput(0);
    if (area > 1) {
        B = B * area;
        N = N / area;

        nvinfer1::IShuffleLayer* qk_reshape = network->addShuffle(*qk_flatten_t->getOutput(0));
        qk_reshape->setReshapeDimensions(nvinfer1::Dims3{B, N, C * 2});
        nvinfer1::IShuffleLayer* v_reshape = network->addShuffle(*v_flatten_t->getOutput(0));
        v_reshape->setReshapeDimensions(nvinfer1::Dims3{B, N, C});

        q_k = qk_reshape->getOutput(0);
        v_ = v_reshape->getOutput(0);
    }
    nvinfer1::Dims q_k_dim = q_k->getDimensions();
    nvinfer1::ISliceLayer* q =
            network->addSlice(*q_k, nvinfer1::Dims3{0, 0, 0},
                              nvinfer1::Dims3{q_k_dim.d[0], q_k_dim.d[1], q_k_dim.d[2] / 2}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* k =
            network->addSlice(*q_k, nvinfer1::Dims3{0, 0, q_k_dim.d[2] / 2},
                              nvinfer1::Dims3{q_k_dim.d[0], q_k_dim.d[1], q_k_dim.d[2] / 2}, nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* q_reshape = network->addShuffle(*q->getOutput(0));
    q_reshape->setReshapeDimensions(nvinfer1::Dims4{B, N, num_heads, head_dim});
    nvinfer1::IShuffleLayer* k_reshape = network->addShuffle(*k->getOutput(0));
    k_reshape->setReshapeDimensions(nvinfer1::Dims4{B, N, num_heads, head_dim});
    nvinfer1::IShuffleLayer* v_reshape = network->addShuffle(*v_);
    v_reshape->setReshapeDimensions(nvinfer1::Dims4{B, N, num_heads, head_dim});
    
    // (B, N, num_head, head_dim)--->(B, num_head, head_dim, N)
    nvinfer1::IShuffleLayer* q_t_view = network->addShuffle(*q_reshape->getOutput(0));
    q_t_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    
    nvinfer1::IShuffleLayer* k_t_view = network->addShuffle(*k_reshape->getOutput(0));
    k_t_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    nvinfer1::IShuffleLayer* v_t_view = network->addShuffle(*v_reshape->getOutput(0));
    v_t_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});


    nvinfer1::IShuffleLayer* q_T = network->addShuffle(*q_t_view->getOutput(0));
    q_T->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2}); // (B, num_head, N, head_dim, N)
    nvinfer1::IMatrixMultiplyLayer* q_mul_k =
            network->addMatrixMultiply(*q_T->getOutput(0), nvinfer1::MatrixOperation::kNONE, *k_t_view->getOutput(0),
                                       nvinfer1::MatrixOperation::kNONE);

    float scale = 1.0 / sqrt(head_dim);
    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    scale_val[0] = scale;
    nvinfer1::Weights s_w{nvinfer1::DataType::kFLOAT, scale_val, 1};  // scale
    float* shift_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shift_val[0] = 0;
    nvinfer1::Weights sh_w{nvinfer1::DataType::kFLOAT, shift_val, 1};  // shift
    float* power_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    power_val[0] = 1;
    nvinfer1::Weights p_w{nvinfer1::DataType::kFLOAT, power_val, 1};  // power
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
    cout_dim(*attn_div->getOutput(0));

    nvinfer1::IShuffleLayer* attn_t = network->addShuffle(*attn_div->getOutput(0));
    attn_t->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});

    nvinfer1::IMatrixMultiplyLayer* attn_v =
            network->addMatrixMultiply(*v_t_view->getOutput(0), nvinfer1::MatrixOperation::kNONE, *attn_t->getOutput(0),
                                       nvinfer1::MatrixOperation::kNONE);

    nvinfer1::IShuffleLayer* attn_v_t = network->addShuffle(*attn_v->getOutput(0));
    attn_v_t->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    nvinfer1::ITensor* attn_temp = attn_v_t->getOutput(0);
    if (area > 1) {
        B = B / area;
        N = N * area;

        nvinfer1::IShuffleLayer* attn_v_t_r = network->addShuffle(*attn_v_t->getOutput(0));
        attn_v_t_r->setReshapeDimensions(nvinfer1::Dims3{B, N, C});
        attn_temp = attn_v_t_r->getOutput(0);
    }
    nvinfer1::IShuffleLayer* attn_x = network->addShuffle(*attn_temp);
    attn_x->setReshapeDimensions(nvinfer1::Dims4{B, H, W, C});
    attn_x->setSecondTranspose(nvinfer1::Permutation{0, 3, 1, 2});
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
    std::vector<nvinfer1::ITensor*> y{cv1->getOutput(0)};
    nvinfer1::ITensor* current = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        if (a2) {
            nvinfer1::ILayer* m_0 = ABlock(network, weightMap, *current, c_, num_heads,
                                           lname + ".m." + std::to_string(i) + ".0", mlp_ratio, area);
            nvinfer1::ILayer* m_1 = ABlock(network, weightMap, *m_0->getOutput(0), c_, num_heads,
                                           lname + ".m." + std::to_string(i) + ".1", mlp_ratio, area);
            current = m_1->getOutput(0);
        } else {
            // C3k 
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

        nvinfer1::IConstantLayer* gamma_layer = network->addConstant(nvinfer1::Dims4{1, c2, 1, 1}, gamma);
        nvinfer1::IElementWiseLayer* scaled_output = network->addElementWise(
                *gamma_layer->getOutput(0), *cv2->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        nvinfer1::IElementWiseLayer* result =
                network->addElementWise(input, *scaled_output->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

        return result;
    } else {
        
        return cv2;
    }
}


