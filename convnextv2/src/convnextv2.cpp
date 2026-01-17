#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "logging.h"
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include "LayerNormPlugin.h"
#include <opencv2/opencv.hpp>

static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "output";

struct ConvNextConfig {
    int depths[4];
    int dims[4];
    int input_h;
    int input_w;
};

// Simple parser for YAML-like config (key: [v1, v2..] or key: value)
ConvNextConfig loadConfig(const std::string& configPath) {
    ConvNextConfig cfg;
    // Default to Nano
    cfg.depths[0]=2; cfg.depths[1]=2; cfg.depths[2]=8; cfg.depths[3]=2;
    cfg.dims[0]=80; cfg.dims[1]=160; cfg.dims[2]=320; cfg.dims[3]=640;
    cfg.input_h=224; cfg.input_w=224;

    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file " << configPath << ". Using default Nano config." << std::endl;
        return cfg;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string key;
        std::getline(ss, key, ':');
        
        // Trim key
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);

        if (key == "depths" || key == "dims") {
            // format: [v1, v2, v3, v4]
            std::string valStr;
            std::getline(ss, valStr);
             // Simple parse: remove [ ] and split by ,
            size_t start = valStr.find('[');
            size_t end = valStr.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string nums = valStr.substr(start + 1, end - start - 1);
                std::stringstream ssNums(nums);
                std::string segment;
                int idx = 0;
                while (std::getline(ssNums, segment, ',') && idx < 4) {
                    if (key == "depths") cfg.depths[idx++] = std::stoi(segment);
                    else cfg.dims[idx++] = std::stoi(segment);
                }
            }
        } else if (key == "input_h") {
            int val; ss >> val; cfg.input_h = val;
        } else if (key == "input_w") {
            int val; ss >> val; cfg.input_w = val;
        }
    }
    std::cout << "Loaded Config - Depths: [" << cfg.depths[0] << "," << cfg.depths[1] << "," << cfg.depths[2] << "," << cfg.depths[3] << "]"
              << " Dims: [" << cfg.dims[0] << "," << cfg.dims[1] << "," << cfg.dims[2] << "," << cfg.dims[3] << "]" 
              << " Input: " << cfg.input_h << "x" << cfg.input_w << std::endl;
    return cfg;
}

// Global config
static ConvNextConfig g_config;
// Macros/Consts replaced by g_config members
#define DEPTHS g_config.depths
#define DIMS g_config.dims
#define INPUT_H g_config.input_h
#define INPUT_W g_config.input_w

using namespace nvinfer1;

static Logger gLogger;

// Global variables for paths
std::string g_wts_path = "convnextv2.wts";
std::string g_engine_path = "convnextv2.engine";

// Weights utils
std::map<std::string, Weights> loadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;
        uint32_t* val = new uint32_t[size];
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap, std::string name, float eps) {
    float* gamma = (float*)weightMap[name + ".weight"].values;
    float* beta = (float*)weightMap[name + ".bias"].values;
    float* mean = (float*)weightMap[name + ".running_mean"].values;
    float* var = (float*)weightMap[name + ".running_var"].values;
    int len = weightMap[name + ".running_var"].count;

    float* scval = new float[len];
    float* shval = new float[len];
    float* pval = new float[len];

    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        pval[i] = 1.0;
    }
    Weights wsc{DataType::kFLOAT, scval, len};
    Weights wsh{DataType::kFLOAT, shval, len};
    Weights wpower{DataType::kFLOAT, pval, len};
    
    IScaleLayer* scale = network->addScale(input, ScaleMode::kCHANNEL, wsh, wsc, wpower);
    assert(scale);
    return scale;
}

ITensor* convNextBlock(INetworkDefinition* network, ITensor* input, int dim, std::string name, std::map<std::string, Weights>& weightMap) {
    // Input is NCHW
    
    // 1. DWConv 7x7
    Weights empty{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* dwconv = network->addConvolutionNd(*input, dim, DimsHW{7, 7}, weightMap[name + ".dwconv.weight"], weightMap[name + ".dwconv.bias"]);
    assert(dwconv);
    dwconv->setStrideNd(DimsHW{1, 1});
    dwconv->setPaddingNd(DimsHW{3, 3});
    dwconv->setNbGroups(dim);
    ITensor* x = dwconv->getOutput(0);

    // 2. Permute NCHW -> NHWC for LayerNorm
    IShuffleLayer* p1 = network->addShuffle(*x);
    p1->setSecondTranspose({0, 2, 3, 1});
    x = p1->getOutput(0);

    // 3. LayerNorm (Plugin)
    auto creator = getPluginRegistry()->getPluginCreator("LayerNorm", "1");
    PluginFieldCollection pfc;
    float eps = 1e-6f;
    PluginField pf("epsilon", &eps, PluginFieldType::kFLOAT32, 1);
    pfc.nbFields = 1;
    pfc.fields = &pf;
    IPluginV2* plugin = creator->createPlugin(name.c_str(), &pfc);

    // Pass gamma/beta (1D of size C) as plugin inputs along with x (N,H,W,C)
    auto w_ln_w = weightMap[name + ".norm.weight"];
    auto w_ln_b = weightMap[name + ".norm.bias"];
    IConstantLayer* c_gamma = network->addConstant(Dims{1, {w_ln_w.count}}, w_ln_w);
    IConstantLayer* c_beta = network->addConstant(Dims{1, {w_ln_b.count}}, w_ln_b);
    
    ITensor* inputs[] = {x, c_gamma->getOutput(0), c_beta->getOutput(0)};
    IPluginV2Layer* ln = network->addPluginV2(inputs, 3, *plugin);
    x = ln->getOutput(0);

    // 4. Permute NHWC -> NCHW
    IShuffleLayer* p2 = network->addShuffle(*x);
    p2->setSecondTranspose({0, 3, 1, 2});
    x = p2->getOutput(0);

    // 5. PWConv1 (1x1)
    IConvolutionLayer* pw1 = network->addConvolutionNd(*x, 4 * dim, DimsHW{1, 1}, weightMap[name + ".pwconv1.weight"], weightMap[name + ".pwconv1.bias"]);
    x = pw1->getOutput(0);

    // 6. GELU
    // Manual GELU implementation: 0.5 * x * (1 + erf(x / sqrt(2)))
    float* sqrt2_inv = new float[1]; *sqrt2_inv = 1.0f / std::sqrt(2.0f);
    Weights w_sqrt2{DataType::kFLOAT, sqrt2_inv, 1};
    IConstantLayer* c_sqrt2 = network->addConstant(Dims4{1,1,1,1}, w_sqrt2); // Broadcast
    
    IElementWiseLayer* div = network->addElementWise(*x, *c_sqrt2->getOutput(0), ElementWiseOperation::kPROD);
    IUnaryLayer* erf = network->addUnary(*div->getOutput(0), UnaryOperation::kERF);
    
    float* one = new float[1]; *one = 1.0f;
    Weights w_one{DataType::kFLOAT, one, 1};
    IConstantLayer* c_one = network->addConstant(Dims4{1,1,1,1}, w_one);
    
    IElementWiseLayer* add_erf = network->addElementWise(*erf->getOutput(0), *c_one->getOutput(0), ElementWiseOperation::kSUM);
    
    float* half = new float[1]; *half = 0.5f;
    Weights w_half{DataType::kFLOAT, half, 1};
    IConstantLayer* c_half = network->addConstant(Dims4{1,1,1,1}, w_half);
    
    IElementWiseLayer* mul_half = network->addElementWise(*x, *c_half->getOutput(0), ElementWiseOperation::kPROD);
    
    IElementWiseLayer* gelu = network->addElementWise(*mul_half->getOutput(0), *add_erf->getOutput(0), ElementWiseOperation::kPROD);
    x = gelu->getOutput(0);
    
    // 7. GRN (implemented in NCHW). X shape: [N, 4*dim, H, W], gx -> [N, C, 1, 1]
    
    // x*x
    IElementWiseLayer* sq = network->addElementWise(*x, *x, ElementWiseOperation::kPROD);
    ITensor* x_sq = sq->getOutput(0);
    
    // Sum over H,W (axes 2, 3 = 4 | 8 = 12)
    IReduceLayer* red_sum = network->addReduce(*x_sq, ReduceOperation::kSUM, 12, true); 
    ITensor* sum_x = red_sum->getOutput(0);
    
    // Sqrt
    IUnaryLayer* sqrt_layer = network->addUnary(*sum_x, UnaryOperation::kSQRT);
    ITensor* gx = sqrt_layer->getOutput(0); // [N, C, 1, 1]
    
    // Normalize GRN: nx = gx / (mean(gx, dim=1) + eps)
    // Mean over C (axis 1)
    IReduceLayer* red_mean = network->addReduce(*gx, ReduceOperation::kAVG, 2, true); // bit 1 set -> axis 1
    ITensor* mean_gx = red_mean->getOutput(0); // [N, 1, 1, 1]
    
    // Add eps
    float eps_val = 1e-6f;
    Weights w_eps{DataType::kFLOAT, &eps_val, 1};

    // Creating scalar constant [1,1,1,1]
    float* eps_ptr = new float[1]; eps_ptr[0]=1e-6f;
    Weights eps_w{DataType::kFLOAT, eps_ptr, 1};
    IConstantLayer* c_eps = network->addConstant(Dims4{1,1,1,1}, eps_w);
    
    IElementWiseLayer* add_eps = network->addElementWise(*mean_gx, *c_eps->getOutput(0), ElementWiseOperation::kSUM);
    ITensor* denom = add_eps->getOutput(0);
    
    // Div
    IElementWiseLayer* div_grn = network->addElementWise(*gx, *denom, ElementWiseOperation::kDIV);
    ITensor* nx = div_grn->getOutput(0); // [N, C, 1, 1]
    
    // Scale X by nx
    IElementWiseLayer* scale_x = network->addElementWise(*x, *nx, ElementWiseOperation::kPROD);
    ITensor* x_norm = scale_x->getOutput(0);
    
    // Apply Gamma/Beta for GRN (channel-wise scale) then add residual from GELU input
    Weights w_grn_g = weightMap[name + ".grn.gamma"];
    Weights w_grn_b = weightMap[name + ".grn.beta"];
    Weights w_power{DataType::kFLOAT, nullptr, 0};
    IScaleLayer* grn_scale = network->addScale(*x_norm, ScaleMode::kCHANNEL, w_grn_b, w_grn_g, w_power);
    x = grn_scale->getOutput(0);

    // Residual: x = grn_scaled + gelu_output
    ITensor* x_in = gelu->getOutput(0);
    IElementWiseLayer* add_grn = network->addElementWise(*x, *x_in, ElementWiseOperation::kSUM);
    x = add_grn->getOutput(0);
    
    
    // 8. PWConv2 (1x1)
    IConvolutionLayer* pw2 = network->addConvolutionNd(*x, dim, DimsHW{1, 1}, weightMap[name + ".pwconv2.weight"], weightMap[name + ".pwconv2.bias"]);
    x = pw2->getOutput(0);
    
    // 9. DropPath (Ignored in inference)
    
    // 10. Residual
    IElementWiseLayer* res = network->addElementWise(*input, *x, ElementWiseOperation::kSUM);
    return res->getOutput(0);
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // Create input
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{maxBatchSize, 3, INPUT_H, INPUT_W});
    assert(data);

    // Load weights from the path provided via command line (g_wts_path)
    std::map<std::string, Weights> weightMap = loadWeights(g_wts_path);

    // Initialize Stem
    // downsample_layers.0: Conv 4x4, s=4 -> LN
    // Conv
    IConvolutionLayer* conv0 = network->addConvolutionNd(*data, DIMS[0], DimsHW{4, 4}, weightMap["downsample_layers.0.0.weight"], weightMap["downsample_layers.0.0.bias"]);
    assert(conv0);
    conv0->setStrideNd(DimsHW{4, 4});

    ITensor* x = conv0->getOutput(0);
    
    // LN
    // Transpose to NHWC
    IShuffleLayer* p0 = network->addShuffle(*x);
    p0->setSecondTranspose({0, 2, 3, 1});
    x = p0->getOutput(0);
    
    // Plugin
    auto creator = getPluginRegistry()->getPluginCreator("LayerNorm", "1");
    PluginFieldCollection pfc;
    float eps = 1e-6f;
    PluginField pf("epsilon", &eps, PluginFieldType::kFLOAT32, 1);
    pfc.nbFields = 1; pfc.fields = &pf;
    IPluginV2* plugin = creator->createPlugin("stem_ln", &pfc);
    
    auto w_ln0_w = weightMap["downsample_layers.0.1.weight"];
    auto w_ln0_b = weightMap["downsample_layers.0.1.bias"];
    IConstantLayer* c_g0 = network->addConstant(Dims{1, {w_ln0_w.count}}, w_ln0_w);
    IConstantLayer* c_b0 = network->addConstant(Dims{1, {w_ln0_b.count}}, w_ln0_b);
    ITensor* in0[] = {x, c_g0->getOutput(0), c_b0->getOutput(0)};
    IPluginV2Layer* ln0 = network->addPluginV2(in0, 3, *plugin);
    x = ln0->getOutput(0);
    
    // Transpose back
    IShuffleLayer* p0_back = network->addShuffle(*x);
    p0_back->setSecondTranspose({0, 3, 1, 2});
    x = p0_back->getOutput(0);
    
    // Stages
    for (int i = 0; i < 4; i++) {
        // Downsample layer (except first stage which is stem)
        if (i > 0) {
            std::string ds_name = "downsample_layers." + std::to_string(i);
            // LN -> Conv 2x2 s=2
            // LN (NHWC)
            IShuffleLayer* p_ds = network->addShuffle(*x);
            p_ds->setSecondTranspose({0, 2, 3, 1});
            x = p_ds->getOutput(0);
            
            auto creator = getPluginRegistry()->getPluginCreator("LayerNorm", "1");
            PluginFieldCollection pfc_ds;
            float eps_ds = 1e-6f;
            PluginField pf_ds("epsilon", &eps_ds, PluginFieldType::kFLOAT32, 1);
            pfc_ds.nbFields = 1; pfc_ds.fields = &pf_ds;
            IPluginV2* plugin_ds = creator->createPlugin((ds_name + "_ln").c_str(), &pfc_ds);
            
            auto w_ds_w = weightMap[ds_name + ".0.weight"];
            auto w_ds_b = weightMap[ds_name + ".0.bias"];
            IConstantLayer* c_ds_g = network->addConstant(Dims{1, {w_ds_w.count}}, w_ds_w);
            IConstantLayer* c_ds_b = network->addConstant(Dims{1, {w_ds_b.count}}, w_ds_b);
            ITensor* in_ds[] = {x, c_ds_g->getOutput(0), c_ds_b->getOutput(0)};
            IPluginV2Layer* ln_ds = network->addPluginV2(in_ds, 3, *plugin_ds);
            x = ln_ds->getOutput(0);
            
            IShuffleLayer* p_ds_back = network->addShuffle(*x);
            p_ds_back->setSecondTranspose({0, 3, 1, 2});
            x = p_ds_back->getOutput(0);
            
            // Conv 2x2, s=2
            IConvolutionLayer* conv_ds = network->addConvolutionNd(*x, DIMS[i], DimsHW{2, 2}, weightMap[ds_name + ".1.weight"], weightMap[ds_name + ".1.bias"]);
            conv_ds->setStrideNd(DimsHW{2, 2});
            x = conv_ds->getOutput(0);
        }
        
        // Blocks
        for (int j = 0; j < DEPTHS[i]; j++) {
            std::string block_name = "stages." + std::to_string(i) + "." + std::to_string(j);
            x = convNextBlock(network, x, DIMS[i], block_name, weightMap);
        }
    }
    
    // Final Norm (Global Avg Pooling -> LayerNorm -> Head)
    
    // Global Avg Pooling
    IReduceLayer* gap = network->addReduce(*x, ReduceOperation::kAVG, 12, true); // sum H,W (indices 2,3)
    x = gap->getOutput(0); // [N, C, 1, 1]
    
    // Reshape to [N,1,1,C] so LayerNorm plugin sees channels as last dimension
    IShuffleLayer* p_fin = network->addShuffle(*x);
    p_fin->setReshapeDimensions(Dims4{maxBatchSize, 1, 1, DIMS[3]});
    x = p_fin->getOutput(0);
    
    auto creator_fin = getPluginRegistry()->getPluginCreator("LayerNorm", "1");
    PluginFieldCollection pfc_fin;
    float eps_fin = 1e-6f;
    PluginField pf_fin("epsilon", &eps_fin, PluginFieldType::kFLOAT32, 1);
    pfc_fin.nbFields = 1; pfc_fin.fields = &pf_fin;
    IPluginV2* plugin_fin = creator_fin->createPlugin("final_norm", &pfc_fin);
    
    // norm.weight / norm.bias
    auto w_fn_w = weightMap["norm.weight"];
    auto w_fn_b = weightMap["norm.bias"];
    IConstantLayer* c_fn_g = network->addConstant(Dims{1, {w_fn_w.count}}, w_fn_w);
    IConstantLayer* c_fn_b = network->addConstant(Dims{1, {w_fn_b.count}}, w_fn_b);
    ITensor* in_fn[] = {x, c_fn_g->getOutput(0), c_fn_b->getOutput(0)};
    IPluginV2Layer* ln_fn = network->addPluginV2(in_fn, 3, *plugin_fin);
    x = ln_fn->getOutput(0);
    
    // Reshape back to [N, C, 1, 1] for 1x1 conv.
    IShuffleLayer* p_fin_b = network->addShuffle(*x);
    p_fin_b->setReshapeDimensions(Dims4{maxBatchSize, DIMS[3], 1, 1});
    x = p_fin_b->getOutput(0);
    
    Weights head_w = weightMap["head.weight"];
    Weights head_b = weightMap["head.bias"];
    // Check num classes
    int num_classes = head_w.count / DIMS[3];
    
    IConvolutionLayer* head = network->addConvolutionNd(*x, num_classes, DimsHW{1, 1}, head_w, head_b);
    x = head->getOutput(0);
    
    x->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*x);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    // Workspace size configured below depending on TRT version
    #if (NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR) >= 86
    // setMemoryPoolLimit
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB
    #else
    config->setMaxWorkspaceSize(1 << 30); // 1GB
    #endif

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    
    delete network;
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);
    (*modelStream) = engine->serialize();
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void inference(const std::string& engine_file, const std::string& image_file) {
    std::cout << "Running inference..." << std::endl;
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Engine file not found" << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Load image
    cv::Mat img = cv::imread(image_file);
    if (img.empty()) {
        std::cerr << "Error: Image not found" << std::endl;
        return;
    }
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    img.convertTo(img, CV_32F);
    
    // Normalize (Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225])
    // OpenCV is BGR. Pytorch expects RGB.
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img /= 255.0;
    
    float mean[] = {0.485, 0.456, 0.406};
    float std[] = {0.229, 0.224, 0.225};
    
    // HWC -> NCHW and Normalize
    float* hostData = new float[3 * INPUT_H * INPUT_W];
    for (int h = 0; h < INPUT_H; ++h) {
        for (int w = 0; w < INPUT_W; ++w) {
            for (int c = 0; c < 3; ++c) {
                float val = img.at<cv::Vec3f>(h, w)[c];
                hostData[c * INPUT_H * INPUT_W + h * INPUT_W + w] = (val - mean[c]) / std[c];
            }
        }
    }

    void* deviceData;
    cudaMalloc(&deviceData, 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMemcpy(deviceData, hostData, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice);

    // Output buffer
    // Determine output size. 
    int outputSize = 1000; // Default ImageNet
    // Check binding dimensions
    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    Dims outDims = engine->getBindingDimensions(outputIndex);
    // outputSize = outDims.d[1];
    
    float* hostOutput = new float[outputSize];
    void* deviceOutput;
    cudaMalloc(&deviceOutput, outputSize * sizeof(float));

    void* bindings[] = {deviceData, deviceOutput};
    
    // Execute
    context->executeV2(bindings);

    // Copy back
    cudaMemcpy(hostOutput, deviceOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Softmax and Argmax
    float maxVal = -1e9;
    int maxIdx = -1;
    for (int i = 0; i < outputSize; ++i) {
        if (hostOutput[i] > maxVal) {
            maxVal = hostOutput[i];
            maxIdx = i;
        }
    }
    std::cout << "Predicted Class: " << maxIdx << " (Score: " << maxVal << ")" << std::endl;

    cudaFree(deviceData);
    cudaFree(deviceOutput);
    delete[] hostData;
    delete[] hostOutput;
    delete context;
    delete engine;
    delete runtime;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <wts_path> <engine_path> [config_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " convnextv2.wts convnextv2.engine config.yaml" << std::endl;
        return -1;
    }

    g_wts_path = argv[1];
    g_engine_path = argv[2];
    std::string config_path = (argc >= 4) ? argv[3] : "config.yaml";
    g_config = loadConfig(config_path);

    // Register Plugin manually if needed
    auto* lnCreator = new LayerNormPluginCreator();
    getPluginRegistry()->registerCreator(*lnCreator, "");
    
    // Generate engine
    IHostMemory* modelStream{nullptr};
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p(g_engine_path, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    std::cout << "Engine generated successfully: " << g_engine_path << std::endl;
    
    return 0;
}
