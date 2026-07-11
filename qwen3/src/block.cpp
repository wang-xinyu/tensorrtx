#include "block.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include "apply_rotary_pos_emb_plugin.h"
#include "config.h"
#include "repeat_kv_plugin.h"

namespace {
std::vector<std::unique_ptr<int32_t[]>> gGatherIndexStorage;
}

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file) {
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

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
    }
    return WeightMap;
}

void freeWeights(std::map<std::string, nvinfer1::Weights>& weightMap) {
    for (auto& mem : weightMap) {
        free(const_cast<void*>(mem.second.values));
    }
    weightMap.clear();
}

nvinfer1::IConstantLayer* createScalarConstant(nvinfer1::INetworkDefinition* network, int ndDims,
                                               nvinfer1::Weights& weight) {
    nvinfer1::Dims dims{};
    dims.nbDims = ndDims;
    for (int i = 0; i < ndDims - 1; ++i) {
        dims.d[i] = 1;
    }
    dims.d[ndDims - 1] = weight.count;
    nvinfer1::IConstantLayer* const_layer = network->addConstant(dims, weight);
    return const_layer;
}

nvinfer1::IElementWiseLayer* Qwen3RMSNorm(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                          int ndDims, const std::string& lname) {
    auto input_dims = input.getDimensions();
    int nb_dims = input_dims.nbDims;
    int last_axis = nb_dims - 1;

    const std::string weight_name = lname.empty() ? "weight" : lname + ".weight";
    nvinfer1::IConstantLayer* weight_const = createScalarConstant(network, ndDims, weightMap[weight_name]);
    assert(weight_const);

    // ---------------------------------------------------------
    // 1. x^2 : input * input
    // ---------------------------------------------------------
    nvinfer1::IElementWiseLayer* pow2 = network->addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD);
    assert(pow2);

    // ---------------------------------------------------------
    // 2. mean : mean(-1, keepdim=True)
    // ---------------------------------------------------------
    uint32_t reduce_axis_mask = 1U << last_axis;
    nvinfer1::IReduceLayer* mean =
            network->addReduce(*pow2->getOutput(0), nvinfer1::ReduceOperation::kAVG, reduce_axis_mask, true);
    assert(mean);

    // ---------------------------------------------------------
    // 3. + epsilon : variance + eps
    // ---------------------------------------------------------
    nvinfer1::Weights eps_wt{nvinfer1::DataType::kFLOAT, nullptr, 1};
    eps_wt.values = &kRmsNormEps;
    nvinfer1::IConstantLayer* eps_const = createScalarConstant(network, ndDims, eps_wt);

    nvinfer1::IElementWiseLayer* add_eps = network->addElementWise(*mean->getOutput(0), *eps_const->getOutput(0),
                                                                   nvinfer1::ElementWiseOperation::kSUM);
    assert(add_eps);

    // ---------------------------------------------------------
    // 4. rsqrt : 1 / sqrt(variance + eps)
    // ---------------------------------------------------------
    // sqrt(variance + eps)
    nvinfer1::IUnaryLayer* sqrt = network->addUnary(*add_eps->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    assert(sqrt);

    // rsqrt : 1 / sqrt(...)
    nvinfer1::IUnaryLayer* rsqrt = network->addUnary(*sqrt->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
    assert(rsqrt);

    // ---------------------------------------------------------
    // 5. norm : hidden_states * rsqrt
    // ---------------------------------------------------------
    nvinfer1::IElementWiseLayer* norm =
            network->addElementWise(input, *rsqrt->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(norm);

    // ---------------------------------------------------------
    // 6. scale : weight * norm_hidden_states
    // ---------------------------------------------------------
    nvinfer1::IElementWiseLayer* scale = network->addElementWise(*weight_const->getOutput(0), *norm->getOutput(0),
                                                                 nvinfer1::ElementWiseOperation::kPROD);
    assert(scale);

    return scale;
}

nvinfer1::IShuffleLayer* transpose(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   const std::vector<int>& perm) {
    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(input);
    nvinfer1::Permutation p{};
    for (size_t i = 0; i < perm.size(); ++i) {
        p.order[i] = perm[i];
    }
    shuffle->setFirstTranspose(p);
    return shuffle;
}

nvinfer1::IShuffleLayer* reshape(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                 const std::vector<int>& dims) {
    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(input);
    nvinfer1::Dims d;
    d.nbDims = dims.size();
    for (size_t i = 0; i < dims.size(); ++i) {
        d.d[i] = dims[i];
    }
    shuffle->setReshapeDimensions(d);
    return shuffle;
}

nvinfer1::ILayer* addLinear(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                            nvinfer1::ITensor& input, int out_features, const std::string& lname) {
    // 1. PyTorch [out_features, in_features]
    const std::string weight_name = lname.empty() ? "weight" : lname + ".weight";
    const std::string bias_name = lname.empty() ? "bias" : lname + ".bias";
    nvinfer1::Weights weight_wt = weightMap[weight_name];
    int in_features = weight_wt.count / out_features;

    // 2. 2D [out_features, in_features]
    nvinfer1::Dims weight_dims{};
    weight_dims.nbDims = 3;
    weight_dims.d[0] = kBatchSize;
    weight_dims.d[1] = out_features;
    weight_dims.d[2] = in_features;

    nvinfer1::IConstantLayer* weight_const = network->addConstant(weight_dims, weight_wt);
    assert(weight_const);

    // 4. y = x * W^T
    // input: [B, S, in_features]
    // W (reshape): [1, out_features, in_features]
    // W^T (transpose): [1, in_features, out_features]
    // [B, S, out_features]
    nvinfer1::IMatrixMultiplyLayer* matmul =
            network->addMatrixMultiply(input, nvinfer1::MatrixOperation::kNONE, *weight_const->getOutput(0),
                                       nvinfer1::MatrixOperation::kTRANSPOSE);
    assert(matmul);

    // 5. bais
    if (weightMap.count(bias_name) && weightMap[bias_name].count > 0) {
        nvinfer1::Weights bias_wt = weightMap[bias_name];

        // 1D [out_features]
        nvinfer1::Dims bias_dims{};
        bias_dims.nbDims = 3;
        bias_dims.d[0] = kBatchSize;
        bias_dims.d[1] = 1;
        bias_dims.d[2] = out_features;

        nvinfer1::IConstantLayer* bias_const = network->addConstant(bias_dims, bias_wt);
        assert(bias_const);

        // y = y + b
        nvinfer1::IElementWiseLayer* add_bias = network->addElementWise(
                *matmul->getOutput(0), *bias_const->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        assert(add_bias);

        return add_bias;
    }

    return matmul;
}

nvinfer1::ILayer* Qwen3Embedding(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input_ids) {
    nvinfer1::Dims2 embedding_dims{kVocabSize, kHiddenSize};
    auto* embedding_const = network->addConstant(embedding_dims, weightMap["weight"]);
    assert(embedding_const != nullptr);
    auto* gather_layer = network->addGather(*embedding_const->getOutput(0), input_ids, 0);
    assert(gather_layer != nullptr);
    return gather_layer;
}

nvinfer1::ILayer* GatherLastToken(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& hidden_states) {
    auto* shape_layer = network->addShape(hidden_states);
    assert(shape_layer != nullptr);

    static const int64_t kSeqDimIndexData[] = {1};
    nvinfer1::Weights seq_dim_index{nvinfer1::DataType::kINT64, kSeqDimIndexData, 1};
    auto* seq_dim_index_const = network->addConstant(nvinfer1::Dims{1, {1}}, seq_dim_index);
    assert(seq_dim_index_const != nullptr);

    auto* seq_len = network->addGather(*shape_layer->getOutput(0), *seq_dim_index_const->getOutput(0), 0);
    assert(seq_len != nullptr);

    static const int64_t kOneData[] = {1};
    nvinfer1::Weights one_weight{nvinfer1::DataType::kINT64, kOneData, 1};
    auto* one_const = network->addConstant(nvinfer1::Dims{1, {1}}, one_weight);
    assert(one_const != nullptr);

    auto* last_index = network->addElementWise(*seq_len->getOutput(0), *one_const->getOutput(0),
                                               nvinfer1::ElementWiseOperation::kSUB);
    assert(last_index != nullptr);

    auto* last_token = network->addGather(hidden_states, *last_index->getOutput(0), 1);
    assert(last_token != nullptr);
    return last_token;
}

nvinfer1::ILayer* Qwen3MLP(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                           nvinfer1::ITensor& input, const std::string& lname) {
    auto* gate_proj = addLinear(network, weightMap, input, kIntermediateSize, lname + ".gate_proj");
    assert(gate_proj != nullptr);
    auto* up_proj = addLinear(network, weightMap, input, kIntermediateSize, lname + ".up_proj");
    assert(up_proj != nullptr);

    auto* gate_sigmoid = network->addActivation(*gate_proj->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    assert(gate_sigmoid != nullptr);
    auto* gate_silu = network->addElementWise(*gate_proj->getOutput(0), *gate_sigmoid->getOutput(0),
                                              nvinfer1::ElementWiseOperation::kPROD);
    assert(gate_silu != nullptr);

    auto* gated = network->addElementWise(*gate_silu->getOutput(0), *up_proj->getOutput(0),
                                          nvinfer1::ElementWiseOperation::kPROD);
    assert(gated != nullptr);

    auto* down_proj = addLinear(network, weightMap, *gated->getOutput(0), kHiddenSize, lname + ".down_proj");
    assert(down_proj != nullptr);
    return down_proj;
}

std::tuple<nvinfer1::ILayer*, nvinfer1::ILayer*, nvinfer1::ILayer*> Qwen3DecoderLayer(
        nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
        nvinfer1::ITensor& hidden_states, nvinfer1::ITensor& cos, nvinfer1::ITensor& sin,
        nvinfer1::ITensor& k_cache_input, nvinfer1::ITensor& v_cache_input, nvinfer1::ITensor& mask_input,
        int layer_idx) {
    auto* input_layernorm = Qwen3RMSNorm(network, weightMap, hidden_states, 3, "input_layernorm");
    assert(input_layernorm != nullptr);

    auto attention = Qwen3Attention(network, weightMap, *input_layernorm->getOutput(0), cos, sin, k_cache_input,
                                    v_cache_input, mask_input, layer_idx, "self_attn");

    auto* attention_output = std::get<0>(attention);
    auto* attention_residual = network->addElementWise(hidden_states, *attention_output->getOutput(0),
                                                       nvinfer1::ElementWiseOperation::kSUM);
    assert(attention_residual != nullptr);

    auto* post_attention_layernorm =
            Qwen3RMSNorm(network, weightMap, *attention_residual->getOutput(0), 3, "post_attention_layernorm");
    assert(post_attention_layernorm != nullptr);

    auto* mlp_output = Qwen3MLP(network, weightMap, *post_attention_layernorm->getOutput(0), "mlp");
    assert(mlp_output != nullptr);

    auto* final_hidden_states = network->addElementWise(*attention_residual->getOutput(0), *mlp_output->getOutput(0),
                                                        nvinfer1::ElementWiseOperation::kSUM);
    assert(final_hidden_states != nullptr);

    return {final_hidden_states, std::get<1>(attention), std::get<2>(attention)};
}

nvinfer1::ILayer* rotateHalf(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input) {
    nvinfer1::Dims inputDims = input.getDimensions();
    int headDim = inputDims.d[3];
    int halfHeadDim = headDim / 2;

    auto x2Indices = std::make_unique<int32_t[]>(halfHeadDim);
    for (int i = 0; i < halfHeadDim; ++i)
        x2Indices[i] = halfHeadDim + i;
    nvinfer1::Weights x2Weight{nvinfer1::DataType::kINT32, x2Indices.get(), halfHeadDim};
    auto* x2Const = network->addConstant(nvinfer1::Dims{1, halfHeadDim}, x2Weight);
    gGatherIndexStorage.push_back(std::move(x2Indices));
    auto* gatherX2 = network->addGather(input, *x2Const->getOutput(0), 3);
    nvinfer1::ITensor* x2 = gatherX2->getOutput(0);  // [B, H, S, halfHeadDim]

    static const float kNegOne = -1.0f;
    static const float kZero = 0.0f;
    nvinfer1::Weights negWeight{nvinfer1::DataType::kFLOAT, &kNegOne, 1};
    nvinfer1::Weights zeroWeight{nvinfer1::DataType::kFLOAT, &kZero, 1};
    auto* negScale = network->addScale(*x2, nvinfer1::ScaleMode::kUNIFORM, zeroWeight, negWeight,
                                       nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0});
    nvinfer1::ITensor* negX2Tensor = negScale->getOutput(0);  // [B, H, S, halfHeadDim]

    auto x1Indices = std::make_unique<int32_t[]>(halfHeadDim);
    for (int i = 0; i < halfHeadDim; ++i)
        x1Indices[i] = i;
    nvinfer1::Weights x1Weight{nvinfer1::DataType::kINT32, x1Indices.get(), halfHeadDim};
    auto* x1Const = network->addConstant(nvinfer1::Dims{1, halfHeadDim}, x1Weight);
    gGatherIndexStorage.push_back(std::move(x1Indices));
    auto* gatherX1 = network->addGather(input, *x1Const->getOutput(0), 3);
    nvinfer1::ITensor* x1 = gatherX1->getOutput(0);  // [B, H, S, halfHeadDim]

    nvinfer1::ITensor* concatInputs[] = {negX2Tensor, x1};
    auto* concat = network->addConcatenation(concatInputs, 2);
    concat->setAxis(3);

    return concat;
}

// q, k: [B, num_heads, S, head_dim]
// cos, sin: [B, S, head_dim]
// q_embed, k_embed : [B, num_heads, S, head_dim]
std::pair<nvinfer1::ILayer*, nvinfer1::ILayer*> applyRotaryPosEmb(nvinfer1::INetworkDefinition* network,
                                                                  nvinfer1::ITensor& q, nvinfer1::ITensor& k,
                                                                  nvinfer1::ITensor& cos, nvinfer1::ITensor& sin) {
    // cos old shape: [B, S, head_dim] (nb_dims=3)
    // cos new shape: [B, 1, S, head_dim] (nb_dims=4)
    auto cos_reshape = reshape(network, cos, {kBatchSize, 1, -1, kHeadDim});
    auto sin_reshape = reshape(network, sin, {kBatchSize, 1, -1, kHeadDim});

    // 2. q_embed = (q * cos) + (rotate_half(q) * sin)
    nvinfer1::IElementWiseLayer* q_mul_cos =
            network->addElementWise(q, *cos_reshape->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    // [B, H, S, head_dim]
    nvinfer1::ILayer* q_rotated = rotateHalf(network, q);
    nvinfer1::IElementWiseLayer* q_rot_mul_sin = network->addElementWise(
            *q_rotated->getOutput(0), *sin_reshape->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::IElementWiseLayer* q_embed = network->addElementWise(
            *q_mul_cos->getOutput(0), *q_rot_mul_sin->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    // 3. k_embed = (k * cos) + (rotate_half(k) * sin)
    nvinfer1::IElementWiseLayer* k_mul_cos =
            network->addElementWise(k, *cos_reshape->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::ILayer* k_rotated = rotateHalf(network, k);
    nvinfer1::IElementWiseLayer* k_rot_mul_sin = network->addElementWise(
            *k_rotated->getOutput(0), *sin_reshape->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::IElementWiseLayer* k_embed = network->addElementWise(
            *k_mul_cos->getOutput(0), *k_rot_mul_sin->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    return {q_embed, k_embed};
}

std::pair<nvinfer1::ILayer*, nvinfer1::ILayer*> applyRotaryPosEmbPlugin(nvinfer1::INetworkDefinition* network,
                                                                        nvinfer1::ITensor& q, nvinfer1::ITensor& k,
                                                                        nvinfer1::ITensor& cos,
                                                                        nvinfer1::ITensor& sin) {
    nvinfer1::ITensor* inputs[] = {&q, &k, &cos, &sin};
    auto* plugin = createApplyRotaryPosEmbPlugin();
    assert(plugin != nullptr);
    auto* layer = network->addPluginV2(inputs, 4, *plugin);
    assert(layer != nullptr);
    auto* qIdentity = network->addIdentity(*layer->getOutput(0));
    auto* kIdentity = network->addIdentity(*layer->getOutput(1));
    assert(qIdentity != nullptr);
    assert(kIdentity != nullptr);
    return {qIdentity, kIdentity};
}

// Along dim=1 (Heads), repeat KV tensors with repeat_interleave semantics (GQA)
// input_kv: [B, num_kv_heads, S, head_dim] -> output: [B, num_q_heads, S, head_dim]
nvinfer1::ILayer* repeatKVByConcat(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, int groups) {
    if (groups == 1) {
        auto* identity = network->addIdentity(input);
        assert(identity != nullptr);
        return identity;
    }

    const nvinfer1::Dims input_dims = input.getDimensions();
    assert(input_dims.nbDims == 4);
    const int num_key_value_heads = input_dims.d[1];
    assert(num_key_value_heads > 0);

    auto* plugin = createRepeatKVPlugin(groups);
    assert(plugin != nullptr);

    nvinfer1::ITensor* pluginInputs[] = {&input};
    auto* layer = network->addPluginV2(pluginInputs, 1, *plugin);
    assert(layer != nullptr);
    return layer;
}

// Scaled Dot-Product Attention: Softmax(Q * K^T * mask) * V
// query: [B, H_q, S_q, D]
// repeated_k: [B, H_q, S_kv, D]
// repeated_v: [B, H_q, S_kv, D]
// mask_input: [B, 1, S_q, S_kv] (Mask 张量，有效位置为 0，无效位置为 1。会自动在 H_q 维度广播)
// attention_output: [B, H_q, S_q, D]
nvinfer1::ILayer* buildAttention(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& query,
                                 nvinfer1::ITensor& repeated_k, nvinfer1::ITensor& repeated_v,
                                 nvinfer1::ITensor& mask_input) {
    // 1. Q * K^T
    auto* kt_shuffle = network->addShuffle(repeated_k);
    kt_shuffle->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});  // [B, H_q, D, S_kv]

    auto* qk_matmul = network->addMatrixMultiply(query, nvinfer1::MatrixOperation::kNONE, *kt_shuffle->getOutput(0),
                                                 nvinfer1::MatrixOperation::kNONE);
    // qk_score: [B, H_q, S_q, S_kv]
    auto* scale_const = network->addConstant(nvinfer1::Dims4{1, 1, 1, 1},
                                             nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &kQKScale, 1});
    auto* qk_scaled = network->addElementWise(*qk_matmul->getOutput(0), *scale_const->getOutput(0),
                                              nvinfer1::ElementWiseOperation::kPROD);

    // 2. Mask (score + (mask * -10000))
    // mask_input: [B, 1, S_q, S_kv]
    static float neg_val = -10000;
    auto* neg_const = network->addConstant(nvinfer1::Dims4{1, 1, 1, 1},
                                           nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &neg_val, 1});

    auto* masked_neg =
            network->addElementWise(mask_input, *neg_const->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    // masked_neg: [B, 1, S_q, S_kv]
    // return masked_neg; // [1 1 32 32]

    auto* qk_masked = network->addElementWise(*qk_scaled->getOutput(0), *masked_neg->getOutput(0),
                                              nvinfer1::ElementWiseOperation::kSUM);
    // qk_masked: [B, H_q, S_q, S_kv]

    // 3. Softmax
    uint32_t reduce_axis_mask = 1U << 3;
    auto* softmax_layer = network->addSoftMax(*qk_masked->getOutput(0));
    assert(softmax_layer != nullptr);
    softmax_layer->setAxes(reduce_axis_mask);

    // 4. Attn * V
    auto* attn_matmul = network->addMatrixMultiply(*softmax_layer->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                                   repeated_v, nvinfer1::MatrixOperation::kNONE);
    // attn_out: [B, H_q, S_q, D]

    // 5. attn_output.transpose(1, 2).contiguous()
    // [B, H_q, S_q, D] -> [B, S_q, H_q, D]
    auto* transpose_shuffle = network->addShuffle(*attn_matmul->getOutput(0));
    assert(transpose_shuffle != nullptr);
    transpose_shuffle->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});

    // [B, S_q, H_q, D]
    return transpose_shuffle;
}

std::tuple<nvinfer1::ILayer*, nvinfer1::ILayer*, nvinfer1::ILayer*> Qwen3Attention(
        nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
        nvinfer1::ITensor& hidden_states, nvinfer1::ITensor& cos, nvinfer1::ITensor& sin,
        nvinfer1::ITensor& k_cache_input, nvinfer1::ITensor& v_cache_input, nvinfer1::ITensor& mask_input,
        int layer_idx, const std::string& lname) {
    int head_dim = kHeadDim;
    int num_attention_heads = kNumAttentionHeads;
    int num_key_value_heads = kNumKeyValueHeads;
    int num_key_value_groups = num_attention_heads / num_key_value_heads;
    float scaling = 1.0f / sqrt(static_cast<float>(head_dim));
    int qkv_dim = num_attention_heads * head_dim;
    int kv_dim = num_key_value_heads * head_dim;

    // 1. QKV
    // hidden_states: [B, S, H]
    auto q_proj = addLinear(network, weightMap, hidden_states, qkv_dim, lname + ".q_proj");
    auto q_reshape = reshape(network, *q_proj->getOutput(0), {kBatchSize, -1, num_attention_heads, head_dim});
    auto q_norm = Qwen3RMSNorm(network, weightMap, *q_reshape->getOutput(0), 4, lname + ".q_norm");
    auto query_states = transpose(network, *q_norm->getOutput(0), {0, 2, 1, 3});

    auto k_proj = addLinear(network, weightMap, hidden_states, kv_dim, lname + ".k_proj");
    auto k_reshape = reshape(network, *k_proj->getOutput(0), {kBatchSize, -1, num_key_value_heads, head_dim});
    auto k_norm = Qwen3RMSNorm(network, weightMap, *k_reshape->getOutput(0), 4, lname + ".k_norm");
    auto key_states = transpose(network, *k_norm->getOutput(0), {0, 2, 1, 3});

    auto v_proj = addLinear(network, weightMap, hidden_states, kv_dim, lname + ".v_proj");
    auto v_reshape = reshape(network, *v_proj->getOutput(0), {kBatchSize, -1, num_key_value_heads, head_dim});
    auto value_states = transpose(network, *v_reshape->getOutput(0), {0, 2, 1, 3});

    auto [q_embed, k_embed] =
            applyRotaryPosEmbPlugin(network, *query_states->getOutput(0), *key_states->getOutput(0), cos, sin);

    // cat kv cache
    nvinfer1::ITensor* k_cache_concat[] = {&k_cache_input, k_embed->getOutput(0)};
    nvinfer1::ITensor* v_cache_concat[] = {&v_cache_input, value_states->getOutput(0)};
    auto* k_cache = network->addConcatenation(k_cache_concat, 2);
    k_cache->setAxis(2);  // seq dim
    auto* v_cache = network->addConcatenation(v_cache_concat, 2);
    v_cache->setAxis(2);  // seq dim

    // =========================================================
    // KV Repeat (GQA)
    // =========================================================
    // k_embed: [B, num_kv_heads, S, head_dim] -> [B, num_q_heads, S, head_dim]
    auto repeated_k = repeatKVByConcat(network, *k_cache->getOutput(0), num_key_value_groups);
    // value_states: [B, num_kv_heads, S, head_dim] -> [B, num_q_heads, S, head_dim]
    auto repeated_v = repeatKVByConcat(network, *v_cache->getOutput(0), num_key_value_groups);
    // =========================================================

    auto attention_layer = buildAttention(network, *q_embed->getOutput(0), *repeated_k->getOutput(0),
                                          *repeated_v->getOutput(0), mask_input);

    auto* attn_output =
            reshape(network, *attention_layer->getOutput(0), {kBatchSize, -1, num_attention_heads * head_dim});
    assert(attn_output != nullptr);

    auto* o_proj = addLinear(network, weightMap, *attn_output->getOutput(0), kHiddenSize, lname + ".o_proj");
    assert(o_proj != nullptr);

    return {o_proj, k_cache, v_cache};
}
