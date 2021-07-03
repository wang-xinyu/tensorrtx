#pragma once
#include <iostream>
#include <unordered_map>
#include "./logging.h"
#include "backbone.hpp"

#define DEVICE 0
#define BATCH_SIZE 1

// 1 / math.sqrt(head_dim) https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/functional/activation.h#623
static const float SCALING = 0.17677669529663687;
static const float MIN_SIZE = 800.0;
static const int INPUT_H = 800;
static const int INPUT_W = 1066;
static const int NUM_CLASS = 92;  // include background
static const float SCALING_ONE = 1.0;
static const float SHIFT_ZERO = 0.0;
static const float POWER_TWO = 2.0;
static const float EPS = 0.00001;
static const int D_MODEL = 256;
static const int NHEAD = 8;
static const int DIM_FEEDFORWARD = 2048;
static const int NUM_ENCODE_LAYERS = 6;
static const int NUM_DECODE_LAYERS = 6;
static const int NUM_QUERIES = 100;
static const float SCORE_THRESH = 0.5;

const char* INPUT_NODE_NAME = "images";
const std::vector<std::string> OUTPUT_NAMES = { "scores", "boxes"};

void preprocessImg(cv::Mat& img) {
    // convert to rgb
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    float ratio = static_cast<float>(MIN_SIZE) / std::min(img.rows, img.cols);
    int newh = 0, neww = 0;
    if (img.rows < img.cols) {
        newh = MIN_SIZE;
        neww = ratio * img.cols;
    } else {
        newh = ratio * img.rows;
        neww = MIN_SIZE;
    }
    cv::resize(img, img, cv::Size(neww, newh));
    img.convertTo(img, CV_32FC3);
    img /= 255;
    img -= cv::Scalar(0.485, 0.456, 0.406);
    img /= cv::Scalar(0.229, 0.224, 0.225);
}

ITensor* PositionEmbeddingSine(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
ITensor& input,
int num_pos_feats = 64,
int temperature = 10000
) {
    // refer to https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py#12
    // TODO: improve this implementation
    auto mask_dim = input.getDimensions();
    int h = mask_dim.d[1], w = mask_dim.d[2];
    std::vector<std::vector<float>> y_embed(h);
    for (int i = 0; i < h; i++)
        y_embed[i] = std::vector<float>(w, i + 1);
    std::vector<float> sub_embed(w, 0);
    for (int i = 0; i < w; i++)
        sub_embed[i] = i + 1;
    std::vector<std::vector<float>> x_embed(h, sub_embed);

    // normalize
    float eps = 1e-6, scale = 2.0 * 3.1415926;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            y_embed[i][j] = y_embed[i][j] / (h + eps) * scale;
            x_embed[i][j] = x_embed[i][j] / (w + eps) * scale;
        }
    }

    // dim_t
    std::vector<float> dim_t(num_pos_feats, 0);
    for (int i = 0; i < num_pos_feats; i++) {
        dim_t[i] = pow(temperature, (2 * (i / 2) / static_cast<float>(num_pos_feats)));
    }

    // pos_x, pos_y
    std::vector<std::vector<std::vector<float>>> pos_x(h,
    std::vector<std::vector<float>>(w,
    std::vector<float>(num_pos_feats, 0)));

    std::vector<std::vector<std::vector<float>>> pos_y(h,
    std::vector<std::vector<float>>(w,
    std::vector<float>(num_pos_feats, 0)));
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < num_pos_feats; k++) {
                float value_x = x_embed[i][j] / dim_t[k];
                float value_y = y_embed[i][j] / dim_t[k];
                if (k & 1) {
                    pos_x[i][j][k] = std::cos(value_x);
                    pos_y[i][j][k] = std::cos(value_y);
                } else {
                    pos_x[i][j][k] = std::sin(value_x);
                    pos_y[i][j][k] = std::sin(value_y);
                }
            }
        }
    }

    // pos
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * h * w * num_pos_feats * 2));
    float *pNext = pval;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < num_pos_feats; k++) {
                *pNext = pos_y[i][j][k];
                ++pNext;
            }
            for (int k = 0; k < num_pos_feats; k++) {
                *pNext = pos_x[i][j][k];
                ++pNext;
            }
        }
    }
    Weights pos_embed_weight{ DataType::kFLOAT, pval, h * w * num_pos_feats * 2 };
    weightMap["pos"] = pos_embed_weight;
    auto pos_embed = network->addConstant(Dims4{ h * w, num_pos_feats * 2, 1, 1 }, pos_embed_weight);
    assert(pos_embed);
    return pos_embed->getOutput(0);
}

ITensor* MultiHeadAttention(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& query,
ITensor& key,
ITensor& value,
int embed_dim = 256,
int num_heads = 8
) {
    int tgt_len = query.getDimensions().d[0];
    int head_dim = embed_dim / num_heads;

    // q
    auto linear_q = network->addFullyConnected(
        query,
        embed_dim,
        weightMap[lname + ".in_proj_weight_q"],
        weightMap[lname + ".in_proj_bias_q"]);
    assert(linear_q);

    // k
    auto linear_k = network->addFullyConnected(
        key,
        embed_dim,
        weightMap[lname + ".in_proj_weight_k"],
        weightMap[lname + ".in_proj_bias_k"]);
    assert(linear_k);

    // v
    auto linear_v = network->addFullyConnected(
        value,
        embed_dim,
        weightMap[lname + ".in_proj_weight_v"],
        weightMap[lname + ".in_proj_bias_v"]);
    assert(linear_v);

    auto scaling_t = network->addConstant(Dims4{ 1, 1, 1, 1 }, Weights{ DataType::kFLOAT, &SCALING, 1 });
    assert(scaling_t);
    auto q_scaling = network->addElementWise(
        *linear_q->getOutput(0),
        *scaling_t->getOutput(0),
        ElementWiseOperation::kPROD);
    assert(q_scaling);

    auto q_shuffle = network->addShuffle(*q_scaling->getOutput(0));
    assert(q_shuffle);
    q_shuffle->setName((lname + ".q_shuffle").c_str());
    q_shuffle->setReshapeDimensions(Dims3{ -1, num_heads, head_dim });
    q_shuffle->setSecondTranspose(Permutation{1, 0, 2});

    auto k_shuffle = network->addShuffle(*linear_k->getOutput(0));
    assert(k_shuffle);
    k_shuffle->setName((lname + ".k_shuffle").c_str());
    k_shuffle->setReshapeDimensions(Dims3{ -1, num_heads, head_dim });
    k_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 });

    auto v_shuffle = network->addShuffle(*linear_v->getOutput(0));
    assert(v_shuffle);
    v_shuffle->setName((lname + ".v_shuffle").c_str());
    v_shuffle->setReshapeDimensions(Dims3{ -1, num_heads, head_dim });
    v_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 });

    auto q_product_k = network->addMatrixMultiply(*q_shuffle->getOutput(0), false, *k_shuffle->getOutput(0), true);
    assert(q_product_k);

    // src_key_padding_mask are all false, so do nothing here
    // see https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/functional/activation.h#826-#839

    auto softmax = network->addSoftMax(*q_product_k->getOutput(0));
    assert(softmax);
    softmax->setAxes(4);

    auto attn_product_v = network->addMatrixMultiply(*softmax->getOutput(0), false, *v_shuffle->getOutput(0), false);
    assert(attn_product_v);

    auto attn_shuffle = network->addShuffle(*attn_product_v->getOutput(0));
    assert(attn_shuffle);
    attn_shuffle->setName((lname + ".attn_shuffle").c_str());
    attn_shuffle->setFirstTranspose(Permutation{ 1, 0, 2 });
    attn_shuffle->setReshapeDimensions(Dims4{ tgt_len, -1, 1, 1 });

    auto linear_attn = network->addFullyConnected(
        *attn_shuffle->getOutput(0),
        embed_dim,
        weightMap[lname + ".out_proj.weight"],
        weightMap[lname + ".out_proj.bias"]);
    assert(linear_attn);

    return linear_attn->getOutput(0);
}

ITensor* LayerNorm(
INetworkDefinition *network,
ITensor& input,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
int d_model = 256
) {
    // TODO: maybe a better implementation https://github.com/NVIDIA/TensorRT/blob/master/plugin/common/common.cuh#212
    auto mean = network->addReduce(input, ReduceOperation::kAVG, 2, true);
    assert(mean);

    auto sub_mean = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(sub_mean);

    // implement pow2 with scale
    Weights scale{ DataType::kFLOAT, &SCALING_ONE, 1 };
    Weights shift{ DataType::kFLOAT, &SHIFT_ZERO, 1 };
    Weights power{ DataType::kFLOAT, &POWER_TWO, 1 };
    auto pow2 = network->addScaleNd(*sub_mean->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power, 0);
    assert(pow2);

    auto pow_mean = network->addReduce(*pow2->getOutput(0), ReduceOperation::kAVG, 2, true);
    assert(pow_mean);

    auto eps = network->addConstant(Dims4{ 1, 1, 1, 1 }, Weights{ DataType::kFLOAT, &EPS, 1 });
    assert(eps);

    auto add_eps = network->addElementWise(*pow_mean->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
    assert(add_eps);

    auto sqrt = network->addUnary(*add_eps->getOutput(0), UnaryOperation::kSQRT);
    assert(sqrt);

    auto div = network->addElementWise(*sub_mean->getOutput(0), *sqrt->getOutput(0), ElementWiseOperation::kDIV);
    assert(div);

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * d_model));
    for (int i = 0; i < d_model; i++) {
        pval[i] = 1.0;
    }
    Weights norm1_power{ DataType::kFLOAT, pval, d_model };
    weightMap[lname + ".power"] = norm1_power;
    auto affine = network->addScaleNd(
        *div->getOutput(0),
        ScaleMode::kCHANNEL,
        weightMap[lname + ".bias"],
        weightMap[lname + ".weight"],
        norm1_power,
        1);
    assert(affine);
    return affine->getOutput(0);
}

ITensor* TransformerEncoderLayer(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& src,
ITensor& pos,
int d_model = 256,
int nhead = 8,
int dim_feedforward = 2048
) {
    auto pos_embed = network->addElementWise(src, pos, ElementWiseOperation::kSUM);
    assert(pos_embed);

    ITensor* src2 = MultiHeadAttention(
        network,
        weightMap,
        lname + ".self_attn",
        *pos_embed->getOutput(0),
        *pos_embed->getOutput(0),
        src,
        d_model,
        nhead);

    auto shortcut1 = network->addElementWise(src, *src2, ElementWiseOperation::kSUM);
    assert(shortcut1);

    ITensor* norm1 = LayerNorm(network, *shortcut1->getOutput(0), weightMap, lname + ".norm1");

    auto linear1 = network->addFullyConnected(
        *norm1,
        dim_feedforward,
        weightMap[lname + ".linear1.weight"],
        weightMap[lname + ".linear1.bias"]);
    assert(linear1);

    auto relu = network->addActivation(*linear1->getOutput(0), ActivationType::kRELU);
    assert(relu);

    auto linear2 = network->addFullyConnected(
        *relu->getOutput(0),
        d_model,
        weightMap[lname + ".linear2.weight"],
        weightMap[lname + ".linear2.bias"]);
    assert(linear2);

    auto shortcut2 = network->addElementWise(*norm1, *linear2->getOutput(0), ElementWiseOperation::kSUM);
    assert(shortcut2);

    ITensor* norm2 = LayerNorm(network, *shortcut2->getOutput(0), weightMap, lname + ".norm2");
    return norm2;
}

ITensor* TransformerEncoder(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& src,
ITensor& pos,
int num_layers = 6
) {
    ITensor* out = &src;
    for (int i = 0; i < num_layers; i++) {
        std::string layer_name = lname + ".layers." + std::to_string(i);
        out = TransformerEncoderLayer(network, weightMap, layer_name, *out, pos);
    }
    return out;
}

ITensor* TransformerDecoderLayer(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& tgt,
ITensor& memory,
ITensor& pos,
ITensor& query_pos,
int d_model = 256,
int nhead = 8,
int dim_feedforward = 2048
) {
    auto pos_embed = network->addElementWise(tgt, query_pos, ElementWiseOperation::kSUM);
    assert(pos_embed);

    ITensor* tgt2 = MultiHeadAttention(
        network,
        weightMap,
        lname + ".self_attn",
        *pos_embed->getOutput(0),
        *pos_embed->getOutput(0),
        tgt);

    auto shortcut1 = network->addElementWise(tgt, *tgt2, ElementWiseOperation::kSUM);
    assert(shortcut1);
    ITensor* norm1 = LayerNorm(network, *shortcut1->getOutput(0), weightMap, lname + ".norm1");

    auto query_embed = network->addElementWise(*norm1, query_pos, ElementWiseOperation::kSUM);
    assert(query_embed);

    auto key_embed = network->addElementWise(memory, pos, ElementWiseOperation::kSUM);
    assert(key_embed);

    ITensor* mha2 = MultiHeadAttention(
        network,
        weightMap,
        lname + ".multihead_attn",
        *query_embed->getOutput(0),
        *key_embed->getOutput(0),
        memory);

    auto shortcut2 = network->addElementWise(*norm1, *mha2, ElementWiseOperation::kSUM);
    assert(shortcut2);

    ITensor* norm2 = LayerNorm(network, *shortcut2->getOutput(0), weightMap, lname + ".norm2");

    auto linear1 = network->addFullyConnected(
        *norm2,
        dim_feedforward,
        weightMap[lname + ".linear1.weight"],
        weightMap[lname + ".linear1.bias"]);
    assert(linear1);

    auto relu = network->addActivation(*linear1->getOutput(0), ActivationType::kRELU);
    assert(relu);

    auto linear2 = network->addFullyConnected(
        *relu->getOutput(0),
        d_model,
        weightMap[lname + ".linear2.weight"],
        weightMap[lname + ".linear2.bias"]);
    assert(linear2);

    auto shortcut3 = network->addElementWise(*norm2, *linear2->getOutput(0), ElementWiseOperation::kSUM);
    assert(shortcut3);

    ITensor* norm3 = LayerNorm(network, *shortcut3->getOutput(0), weightMap, lname + ".norm3");

    return norm3;
}

ITensor* TransformerDecoder(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& tgt,
ITensor& memory,
ITensor& pos,
ITensor& query_pos,
int num_layers = 6,
int d_model = 256,
int nhead = 8,
int dim_feedforward = 2048
) {
    ITensor* out = &tgt;
    for (int i = 0; i < num_layers; i++) {
        std::string layer_name = lname + ".layers." + std::to_string(i);
        out = TransformerDecoderLayer(
            network,
            weightMap,
            layer_name,
            *out,
            memory,
            pos,
            query_pos,
            d_model,
            nhead,
            dim_feedforward);
    }
    ITensor* norm = LayerNorm(network, *out, weightMap, lname + ".norm", d_model);
    return norm;
}

ITensor* Transformer(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& src,
ITensor& pos_embed,
int num_queries = 100,
int num_encoder_layers = 6,
int num_decoder_layers = 6,
int d_model = 256,
int nhead = 8,
int dim_feedforward = 2048
) {
    auto memory = TransformerEncoder(network, weightMap, lname + ".encoder", src, pos_embed, num_encoder_layers);

    // construct tgt
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * num_queries * d_model));
    for (int i = 0; i < num_queries * d_model; i++) {
        pval[i] = 0.0;
    }
    Weights tgt_weight{ DataType::kFLOAT, pval, num_queries * d_model };
    weightMap[lname + ".tgt_weight"] = tgt_weight;
    auto tgt = network->addConstant(Dims4{ num_queries, d_model, 1, 1 }, tgt_weight);
    assert(tgt);
    // construct query_pos
    auto query_pos = network->addConstant(Dims4{ num_queries, d_model, 1, 1 }, weightMap["query_embed.weight"]);
    assert(query_pos);

    auto out = TransformerDecoder(
        network,
        weightMap,
        lname + ".decoder",
        *tgt->getOutput(0),
        *memory, pos_embed,
        *query_pos->getOutput(0),
        num_decoder_layers,
        d_model,
        nhead,
        dim_feedforward);
    return out;
}

ITensor* MLP(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& src,
int num_layers = 3,
int hidden_dim = 256,
int output_dim = 4
) {
    ITensor* out = &src;
    for (int i = 0; i < num_layers; i++) {
        std::string layer_name = lname + "." + std::to_string(i);
        if (i != num_layers - 1) {
            auto fc = network->addFullyConnected(
                *out,
                hidden_dim,
                weightMap[layer_name + ".weight"],
                weightMap[layer_name + ".bias"]);
            assert(fc);
            auto relu = network->addActivation(*fc->getOutput(0), ActivationType::kRELU);
            assert(relu);
            out = relu->getOutput(0);
        } else {
            auto fc = network->addFullyConnected(
                *out,
                output_dim,
                weightMap[layer_name + ".weight"],
                weightMap[layer_name + ".bias"]);
            assert(fc);
            out = fc->getOutput(0);
        }
    }
    return out;
}

std::vector<ITensor*> Predict(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
ITensor& src
) {
    auto class_embed = network->addFullyConnected(
        src,
        NUM_CLASS,
        weightMap["class_embed.weight"],
        weightMap["class_embed.bias"]);
    assert(class_embed);
    auto class_softmax = network->addSoftMax(*class_embed->getOutput(0));
    assert(class_softmax);
    class_softmax->setAxes(2);
    ITensor* bbox = MLP(network, weightMap, "bbox_embed.layers", src);
    auto bbox_sig = network->addActivation(*bbox, ActivationType::kSIGMOID);
    assert(bbox_sig);
    std::vector<ITensor*> output = { class_softmax->getOutput(0), bbox_sig->getOutput(0) };
    return output;
}

ICudaEngine* createEngine_r50detr(
unsigned int maxBatchSize,
const std::string& wtsfile,
IBuilder* builder,
IBuilderConfig* config,
DataType dt,
const std::string& modelType = "fp16"
) {
    /*
    description: after fuse bn
    */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput("data", dt, Dims3{ 3, INPUT_H, INPUT_W });

    // preprocess
    std::unordered_map<std::string, Weights> weightMap;
    loadWeights(wtsfile, weightMap);

    // backbone
    auto features = BuildResNet(network, weightMap, *data, R50, 64, 64, 256);
    ITensor* pos_embed = PositionEmbeddingSine(network, weightMap, *features, 128);
    auto input_proj = network->addConvolutionNd(
        *features,
        D_MODEL,
        DimsHW{ 1, 1 },
        weightMap["input_proj.weight"],
        weightMap["input_proj.bias"]);
    assert(input_proj);
    input_proj->setStrideNd(DimsHW{ 1, 1 });
    auto flatten = network->addShuffle(*input_proj->getOutput(0));
    assert(flatten);
    flatten->setReshapeDimensions(Dims4{ input_proj->getOutput(0)->getDimensions().d[0], -1, 1, 1 });
    flatten->setSecondTranspose(Permutation{ 1, 0, 2, 3 });

    auto out1 = Transformer(
        network,
        weightMap,
        "transformer",
        *flatten->getOutput(0),
        *pos_embed,
        NUM_QUERIES,
        NUM_ENCODE_LAYERS,
        NUM_DECODE_LAYERS,
        D_MODEL,
        NHEAD,
        DIM_FEEDFORWARD);
    std::vector<ITensor*> results = Predict(network, weightMap, *out1);

    // build output
    for (int i = 0; i < results.size(); i++) {
        network->markOutput(*results[i]);
        results[i]->setName(OUTPUT_NAMES[i].c_str());
    }

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 30);

    if (modelType == "fp32") {
    } else if (modelType == "fp16") {
        config->setFlag(BuilderFlag::kFP16);
    } else if (modelType == "int8") {
        // TODO: test with int8 quantization
    } else {
        throw("does not support model type");
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // destroy network
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return engine;
}

void BuildDETRModel(unsigned int maxBatchSize, IHostMemory** modelStream,
const std::string& wtsfile, std::string modelType = "fp32") {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine_r50detr(maxBatchSize,
        wtsfile, builder, config, DataType::kFLOAT, modelType);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, std::vector<void*>& buffers,
std::vector<float>& input, std::vector<float*>& output) {
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input.data(), input.size() * sizeof(float),
    cudaMemcpyHostToDevice, stream));

    context.enqueue(BATCH_SIZE, buffers.data(), stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output[0], buffers[1], BATCH_SIZE * NUM_QUERIES * NUM_CLASS * sizeof(float),
    cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output[1], buffers[2], BATCH_SIZE * NUM_QUERIES * 4 * sizeof(float),
    cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wtsFile, std::string& engineFile, std::string& imgDir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s") {
        wtsFile = std::string(argv[2]);
        engineFile = std::string(argv[3]);
    } else if (std::string(argv[1]) == "-d") {
        engineFile = std::string(argv[2]);
        imgDir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wtsFile = "";
    std::string engineFile = "";

    std::string imgDir;
    if (!parse_args(argc, argv, wtsFile, engineFile, imgDir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./detr -s [.wts] [.engine] // serialize model to plan file" << std::endl;
        std::cerr << "./detr -d [.engine] ../samples // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    if (!wtsFile.empty()) {
        IHostMemory* modelStream{ nullptr };
        BuildDETRModel(BATCH_SIZE, &modelStream, wtsFile, "fp32");
        assert(modelStream != nullptr);
        std::ofstream p(engineFile, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engineFile << " error!" << std::endl;
        return -1;
    }

    std::string trtModelStream;
    size_t modelSize{ 0 };
    file.seekg(0, file.end);
    modelSize = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream.resize(modelSize);
    assert(!trtModelStream.empty());
    file.read(const_cast<char*>(trtModelStream.c_str()), modelSize);
    file.close();

    // build engine
    std::cout << "build engine" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.c_str(), modelSize);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    runtime->destroy();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // prepare input file
    std::vector<std::string> fileList;
    if (read_files_in_dir(imgDir.c_str(), fileList) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // calculate input size
    int input_size = CalculateSize(context->getBindingDimensions(0));

    // prepare input data
    std::vector<float> data(BATCH_SIZE * input_size, 0);
    void *data_d, *scores_d, *boxes_d;
    CUDA_CHECK(cudaMalloc(&data_d, BATCH_SIZE * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scores_d, BATCH_SIZE * NUM_QUERIES * NUM_CLASS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&boxes_d, BATCH_SIZE * NUM_QUERIES * 4 * sizeof(float)));

    std::vector<float> scores_h(BATCH_SIZE * NUM_QUERIES * NUM_CLASS);
    std::vector<float> boxes_h(BATCH_SIZE * NUM_QUERIES * 4);

    std::vector<void*> buffers = { data_d, scores_d, boxes_d };
    std::vector<float*> outputs = {scores_h.data(), boxes_h.data()};

    int fcount = 0;
    int fileLen = fileList.size();
    for (int f = 0; f < fileLen; f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != fileLen) continue;

        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(imgDir + "/" + fileList[f - fcount + 1 + b]);
            preprocessImg(img);
            assert(img.cols * img.rows * 3 == input_size);
            if (img.empty()) continue;
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < img.rows; h++) {
                    for (int w = 0; w < img.cols; w++) {
                        data[b * input_size +
                        c * img.rows * img.cols + h * img.cols + w] = img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();

        doInference(*context, stream, buffers, data, outputs);

        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(imgDir + "/" + fileList[f - fcount + 1 + b]);
            for (int i = 0; i < scores_h.size(); i += NUM_CLASS) {
                int label = -1;
                float score = -1;
                for (int j = i; j < i + NUM_CLASS; j++) {
                    if (score < scores_h[j]) {
                        label = j;
                        score = scores_h[j];
                    }
                }
                if (score > SCORE_THRESH && (label % NUM_CLASS != NUM_CLASS - 1)) {
                    int ind = label / NUM_CLASS;
                    label = label % NUM_CLASS;
                    float cx = boxes_h[ind * 4];
                    float cy = boxes_h[ind * 4 + 1];
                    float w = boxes_h[ind * 4 + 2];
                    float h = boxes_h[ind * 4 + 3];
                    float x1 = (cx - w / 2.0) * img.cols;
                    float y1 = (cy - h / 2.0) * img.rows;
                    float x2 = (cx + w / 2.0) * img.cols;
                    float y2 = (cy + h / 2.0) * img.rows;
                    cv::Rect r(x1, y1, x2 - x1, y2 - y1);
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, std::to_string(label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
            }
            cv::imwrite("_" + fileList[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(scores_d));
    CUDA_CHECK(cudaFree(boxes_d));
    context->destroy();
    engine->destroy();

    return 0;
}
