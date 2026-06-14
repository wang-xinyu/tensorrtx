#include <NvInfer.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cuda_allocator.h"
#include "logging.h"
#include "macros.h"
#include "profiler.h"
#include "utils.h"

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;
using M = nvinfer1::MatrixOperation;
using E = nvinfer1::ElementWiseOperation;
using NDCF = nvinfer1::NetworkDefinitionCreationFlag;

// DataType::kHALF  -> FP16
// DataType::kFLOAT -> FP32
static constexpr DataType BUILD_PRECISION = DataType::kHALF;
static constexpr int64_t BUILD_MIN_BATCH = 1;
static constexpr int64_t BUILD_OPT_BATCH = 1;
static constexpr int64_t BUILD_MAX_BATCH = 2;

template <std::integral T>
requires(sizeof(T) > sizeof(int32_t)) static auto toI32(T value) -> int32_t {
    assert(std::in_range<int32_t>(value));
    return static_cast<int32_t>(value);
}

// ViT model variant table.
// All variants share the same architecture; only sizes differ.
// `model_type` is the canonical name (e.g. "ViT-B/16"); aliases without slash
// (e.g. "b16") are accepted via normalizeModelType().
struct ViTConfig {
    int64_t hidden = 768;
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t ff_dim = 3072;
    int64_t num_classes = 1000;
    int64_t patch = 16;
    int64_t img_size = 224;
    // Optimization profile used by the generated engine.
    int64_t min_batch = BUILD_MIN_BATCH;
    int64_t opt_batch = BUILD_OPT_BATCH;
    int64_t max_batch = BUILD_MAX_BATCH;
    float lnorm_eps = 1e-12f;
    std::string model_type = "ViT-B/16";
    std::string wts_path;
    std::string engine_path;
    [[nodiscard]] int64_t num_patches() const { return (img_size / patch) * (img_size / patch); }
    [[nodiscard]] int64_t seq_len() const { return num_patches() + 1; }
};

static std::string normalizeModelType(const std::string& s) {
    // Accept "ViT-B/16", "vit-b/16", "b16", "B/16" -> "ViT-B/16".
    std::string t;
    for (char c : s) {
        if (c != '-' && c != '/' && c != ' ') {
            t += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
    }
    // Strip leading "VIT"
    if (t.starts_with("VIT"))
        t = t.substr(3);
    if (t == "B16")
        return "ViT-B/16";
    if (t == "B32")
        return "ViT-B/32";
    if (t == "L16")
        return "ViT-L/16";
    if (t == "L32")
        return "ViT-L/32";
    if (t == "H14")
        return "ViT-H/14";
    return s;  // unknown, return as-is for error reporting
}

static ViTConfig getVariantConfig(const std::string& raw_name) {
    const std::string name = normalizeModelType(raw_name);
    ViTConfig c;
    c.model_type = name;
    if (name == "ViT-B/16") {
        c.hidden = 768;
        c.num_layers = 12;
        c.num_heads = 12;
        c.ff_dim = 3072;
        c.patch = 16;
        c.img_size = 224;
        c.num_classes = 1000;
    } else if (name == "ViT-B/32") {
        c.hidden = 768;
        c.num_layers = 12;
        c.num_heads = 12;
        c.ff_dim = 3072;
        c.patch = 32;
        c.img_size = 384;
        c.num_classes = 1000;
    } else if (name == "ViT-L/16") {
        c.hidden = 1024;
        c.num_layers = 24;
        c.num_heads = 16;
        c.ff_dim = 4096;
        c.patch = 16;
        c.img_size = 224;
        c.num_classes = 1000;
    } else if (name == "ViT-L/32") {
        c.hidden = 1024;
        c.num_layers = 24;
        c.num_heads = 16;
        c.ff_dim = 4096;
        c.patch = 32;
        c.img_size = 384;
        c.num_classes = 1000;
    } else if (name == "ViT-H/14") {
        // HF only ships an ImageNet-21k checkpoint for huge: 21843 classes.
        c.hidden = 1280;
        c.num_layers = 32;
        c.num_heads = 16;
        c.ff_dim = 5120;
        c.patch = 14;
        c.img_size = 224;
        c.num_classes = 21843;
    } else {
        std::cerr << "Unknown model_type: " << raw_name
                  << " (expected ViT-B/16 | ViT-B/32 | ViT-L/16 | ViT-L/32 | ViT-H/14)\n";
        std::abort();
    }
    return c;
}

static constexpr const std::array<const char*, 2> NAMES = {"input", "logits"};
static constexpr const std::array<const float, 3> mean = {0.5f, 0.5f, 0.5f};
static constexpr const std::array<const float, 3> stdv = {0.5f, 0.5f, 0.5f};
static constexpr const char* LABELS_PATH = "assets/imagenet1000_clsidx_to_labels.txt";

static Logger gLogger;

static auto bytesPerElement(DataType t) -> std::size_t {
    switch (t) {
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT64:
            return 8;
        case DataType::kINT32:
            return 4;
#if TRT_VERSION_GE(8, 0, 0)
        case DataType::kBOOL:
#endif
#if TRT_VERSION_GE(8, 5, 0)
        case DataType::kUINT8:
#endif
        case DataType::kINT8:
            return 1;
        default:
            std::cerr << "Unsupported TensorRT DataType\n";
            std::abort();
    }
}

static void convertWeightMapToHalf(WeightMap& w) {
    for (auto& kv : w) {
        auto& wt = kv.second;
        if (wt.type != DataType::kFLOAT || wt.values == nullptr || wt.count <= 0) {
            continue;
        }

        auto* half_vals = new half[wt.count];
        const auto* raw = reinterpret_cast<const uint32_t*>(wt.values);
        for (int64_t i = 0; i < wt.count; ++i) {
            float f;
            std::memcpy(&f, &raw[i], sizeof(float));
            half_vals[i] = __float2half(f);
        }

        delete[] raw;
        wt.type = DataType::kHALF;
        wt.values = half_vals;
    }
}

struct ViTParam {
    uint32_t index;
    uint32_t head_num;
    int64_t hidden;
    int64_t ff_dim;
    float lnorm_eps = 1e-12f;
};

struct RepeatedWeightsStorage {
    std::vector<std::vector<uint32_t>> f32;
    std::vector<std::vector<half>> f16;
    std::vector<std::vector<int64_t>> i64;
};

static auto makeRepeatedBatchWeights(const Weights& src, int64_t repeat, RepeatedWeightsStorage& storage) -> Weights {
    if (repeat <= 0 || src.values == nullptr || src.count <= 0) {
        std::cerr << "invalid repeated weight\n";
        std::abort();
    }

    if (src.type == DataType::kHALF) {
        const auto* values = reinterpret_cast<const half*>(src.values);
        storage.f16.emplace_back(static_cast<std::size_t>(src.count * repeat));
        auto& dst = storage.f16.back();
        for (int64_t i = 0; i < repeat; ++i) {
            std::copy(values, values + src.count, dst.begin() + i * src.count);
        }
        return Weights{src.type, dst.data(), static_cast<int64_t>(dst.size())};
    }

    if (src.type == DataType::kFLOAT) {
        const auto* values = reinterpret_cast<const uint32_t*>(src.values);
        storage.f32.emplace_back(static_cast<std::size_t>(src.count * repeat));
        auto& dst = storage.f32.back();
        for (int64_t i = 0; i < repeat; ++i) {
            std::copy(values, values + src.count, dst.begin() + i * src.count);
        }
        return Weights{src.type, dst.data(), static_cast<int64_t>(dst.size())};
    }

    std::cerr << "unsupported repeated weight dtype\n";
    std::abort();
}

static auto addShapeConstant(INetworkDefinition* net, std::initializer_list<int64_t> values,
                             RepeatedWeightsStorage& storage) -> ITensor* {
    storage.i64.emplace_back(values);
    auto& data = storage.i64.back();
    Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = static_cast<int64_t>(data.size());
    auto* c = net->addConstant(
            dims, Weights{.type = DataType::kINT64, .values = data.data(), .count = static_cast<int64_t>(data.size())});
    return c->getOutput(0);
}

static auto addBatchedConstantSlice(INetworkDefinition* net, ITensor& input, ITensor& full, int64_t dim1, int64_t dim2,
                                    RepeatedWeightsStorage& storage) -> ITensor* {
    auto* input_shape = net->addShape(input);
    Dims batch_start{};
    batch_start.nbDims = 1;
    batch_start.d[0] = 0;
    Dims batch_size{};
    batch_size.nbDims = 1;
    batch_size.d[0] = 1;
    Dims batch_stride{};
    batch_stride.nbDims = 1;
    batch_stride.d[0] = 1;
    auto* batch = net->addSlice(*input_shape->getOutput(0), batch_start, batch_size, batch_stride);
    auto* size_tail = addShapeConstant(net, {dim1, dim2}, storage);
    const std::array<ITensor*, 2> size_inputs = {batch->getOutput(0), size_tail};
    auto* size = net->addConcatenation(size_inputs.data(), toI32(size_inputs.size()));
    size->setAxis(0);

    auto* start = addShapeConstant(net, {0, 0, 0}, storage);
    auto* stride = addShapeConstant(net, {1, 1, 1}, storage);
    const auto full_dims = full.getDimensions();
    auto* slice =
            net->addSlice(full, Dims3{0, 0, 0}, Dims3{full_dims.d[0], full_dims.d[1], full_dims.d[2]}, Dims3{1, 1, 1});
    slice->setInput(1, *start);
    slice->setInput(2, *size->getOutput(0));
    slice->setInput(3, *stride);
    return slice->getOutput(0);
}

static auto addGeLU(INetworkDefinition* net, ITensor& input) -> ILayer* {
#if TRT_VERSION_LT(10, 0, 0)
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const auto inputDims = input.getDimensions();

    Dims scalarDims{};
    scalarDims.nbDims = inputDims.nbDims;
    for (int i = 0; i < scalarDims.nbDims; ++i) {
        scalarDims.d[i] = 1;
    }

    static float _half = 0.5f;
    static float _one = 1.0f;
    static float _sqrt_2_div_pi = std::sqrt(2.0f / M_PI);
    static float _coeff = 0.044715f;
    auto* _w_half = net->addConstant(scalarDims, Weights{.type = DataType::kFLOAT, .values = &_half, .count = 1});
    auto* _w_one = net->addConstant(scalarDims, Weights{.type = DataType::kFLOAT, .values = &_one, .count = 1});
    auto* _w_sqrt_2_div_pi =
            net->addConstant(scalarDims, Weights{.type = DataType::kFLOAT, .values = &_sqrt_2_div_pi, .count = 1});
    auto* _w_coeff = net->addConstant(scalarDims, Weights{.type = DataType::kFLOAT, .values = &_coeff, .count = 1});

    auto* _x2 = net->addElementWise(input, input, E::kPROD);
    auto* x3_0 = net->addElementWise(*_x2->getOutput(0), input, E::kPROD);
    auto* x3_1 = net->addElementWise(*x3_0->getOutput(0), *_w_coeff->getOutput(0), E::kPROD);
    auto* x3_2 = net->addElementWise(input, *x3_1->getOutput(0), E::kSUM);
    auto* scaled = net->addElementWise(*x3_2->getOutput(0), *_w_sqrt_2_div_pi->getOutput(0), E::kPROD);

    auto* t = net->addActivation(*scaled->getOutput(0), ActivationType::kTANH);
    auto* one_plus = net->addElementWise(*t->getOutput(0), *_w_one->getOutput(0), E::kSUM);
    auto* half_x = net->addElementWise(input, *_w_half->getOutput(0), E::kPROD);
    return net->addElementWise(*half_x->getOutput(0), *one_plus->getOutput(0), E::kPROD);
#else
    // erf approximation
    return net->addActivation(input, ActivationType::kGELU_ERF);
#endif
}

static auto addLinearNorm(INetworkDefinition* net, ITensor& input, ITensor& scale, ITensor& bias,
                          uint32_t axesMask) noexcept -> ILayer* {
#if TRT_VERSION_GE(10, 15, 0)
    auto* ln = net->addNormalizationV2(input, scale, bias, axesMask);
#else
    auto* ln = net->addNormalization(input, scale, bias, axesMask);
#endif
    ln->setEpsilon(1e-12f);
    return ln;
}

auto ViTLayer(INetworkDefinition* net, WeightMap& w, ITensor& input, const ViTParam& param) -> ITensor* {
    std::string name = "vit.encoder.layer." + std::to_string(param.index);
    auto attn_name = name + ".attention";
    int64_t H = param.hidden;
    int64_t F = param.ff_dim;
    int64_t attn_head_size = H / param.head_num;

    auto* qw = net->addConstant(Dims3{1, H, H}, w.at(attn_name + ".attention.query.weight"));
    auto* kw = net->addConstant(Dims3{1, H, H}, w.at(attn_name + ".attention.key.weight"));
    auto* vw = net->addConstant(Dims3{1, H, H}, w.at(attn_name + ".attention.value.weight"));
    /* 1. layer norm before attention */
    auto pre_ln_name = name + ".layernorm_before";
    auto dims = input.getDimensions();
    uint32_t axes = 1U << static_cast<uint32_t>(dims.nbDims - 1);
    auto* ln_scale = net->addConstant(Dims3{1, 1, dims.d[dims.nbDims - 1]}, w[pre_ln_name + ".weight"]);
    auto* ln_bias = net->addConstant(Dims3{1, 1, dims.d[dims.nbDims - 1]}, w[pre_ln_name + ".bias"]);
    auto* pre_lnorm = addLinearNorm(net, input, *ln_scale->getOutput(0), *ln_bias->getOutput(0), axes);

    /** 2. multi-head self-attention */
    auto* qb = net->addConstant(Dims3{1, 1, H}, w.at(attn_name + ".attention.query.bias"));
    auto* kb = net->addConstant(Dims3{1, 1, H}, w.at(attn_name + ".attention.key.bias"));
    auto* vb = net->addConstant(Dims3{1, 1, H}, w.at(attn_name + ".attention.value.bias"));
    auto* _lno = pre_lnorm->getOutput(0);
    // 2.1 Q, K attention matmul
    auto* _q_attn = net->addMatrixMultiply(*_lno, M::kNONE, *qw->getOutput(0), M::kTRANSPOSE);
    auto* _k_attn = net->addMatrixMultiply(*_lno, M::kNONE, *kw->getOutput(0), M::kTRANSPOSE);
    auto* _v_attn = net->addMatrixMultiply(*_lno, M::kNONE, *vw->getOutput(0), M::kTRANSPOSE);
    _q_attn->setName((attn_name + "query").c_str());
    _k_attn->setName((attn_name + "key").c_str());
    _v_attn->setName((attn_name + "value").c_str());
    auto* q_attn = net->addElementWise(*_q_attn->getOutput(0), *qb->getOutput(0), E::kSUM);
    auto* k_attn = net->addElementWise(*_k_attn->getOutput(0), *kb->getOutput(0), E::kSUM);
    auto* v_attn = net->addElementWise(*_v_attn->getOutput(0), *vb->getOutput(0), E::kSUM);
    auto* q_s = net->addShuffle(*q_attn->getOutput(0));
    auto* k_s = net->addShuffle(*k_attn->getOutput(0));
    auto* v_s = net->addShuffle(*v_attn->getOutput(0));
    q_s->setReshapeDimensions(Dims4{0, 0, param.head_num, attn_head_size});
    q_s->setSecondTranspose({0, 2, 1, 3});
    k_s->setReshapeDimensions(Dims4{0, 0, param.head_num, attn_head_size});
    k_s->setSecondTranspose({0, 2, 1, 3});
    v_s->setReshapeDimensions(Dims4{0, 0, param.head_num, attn_head_size});
    v_s->setSecondTranspose({0, 2, 1, 3});

    // 2.2 Q, K scaling (and softmax / fused attention)
    const float scale_f = 1.0f / std::sqrt(static_cast<float>(attn_head_size));
    if (input.getType() == DataType::kHALF) {
        auto* scale_val = new half[1];
        scale_val[0] = __float2half(scale_f);
        w[attn_name + ".scale"] = Weights{.type = DataType::kHALF, .values = scale_val, .count = 1};
    } else {
        auto* scale_val = new uint32_t[1];
        std::memcpy(scale_val, &scale_f, sizeof(float));
        w[attn_name + ".scale"] = Weights{.type = DataType::kFLOAT, .values = scale_val, .count = 1};
    }
    auto* qk_scale_w = net->addConstant(Dims4{1, 1, 1, 1}, w.at(attn_name + ".scale"));

    // 2.3 QKV attention output and reshape
#if TRT_VERSION_GE(10, 14, 0) && TRT_VERSION_LT(10, 15, 0)
    gLogger.log(Severity::kWARNING,
                "IAttention is available in TensorRT 10.14.1 SDK but have bugs, use 10.15.1+ to enable native fused "
                "kernel");
#endif
#if TRT_VERSION_GE(10, 15, 0)
    using ANO = AttentionNormalizationOp;
    auto* q_scaled = net->addElementWise(*q_s->getOutput(0), *qk_scale_w->getOutput(0), E::kPROD)->getOutput(0);
    auto* attn = net->addAttention(*q_scaled, *k_s->getOutput(0), *v_s->getOutput(0), ANO::kSOFTMAX, false);
    assert(attn != nullptr);
    if (!attn->setDecomposable(false)) {
        std::cerr << "setDecomposable failed\n";
        std::abort();
    }
    auto* attn_out = net->addShuffle(*attn->getOutput(0));
#else
    auto* qk = net->addMatrixMultiply(*q_s->getOutput(0), M::kNONE, *k_s->getOutput(0), M::kTRANSPOSE);
    auto* attn_qk = net->addElementWise(*qk->getOutput(0), *qk_scale_w->getOutput(0), E::kPROD);
    auto* qk_softmax = net->addSoftMax(*attn_qk->getOutput(0));
    qk_softmax->setAxes(1U << static_cast<uint32_t>(attn_qk->getOutput(0)->getDimensions().nbDims - 1));
    auto* attn_qkv = net->addMatrixMultiply(*qk_softmax->getOutput(0), M::kNONE, *v_s->getOutput(0), M::kNONE);
    attn_qkv->setName((attn_name + ".attn_qkv").c_str());
    auto* attn_out = net->addShuffle(*attn_qkv->getOutput(0));
#endif
    attn_out->setFirstTranspose({0, 2, 1, 3});
    attn_out->setReshapeDimensions(Dims3{0, 0, H});
    // 2.4 attention output projection
    auto* out_proj_w = net->addConstant(Dims3{1, H, H}, w.at(attn_name + ".output.dense.weight"))->getOutput(0);
    auto* out_proj_b = net->addConstant(Dims3{1, 1, H}, w.at(attn_name + ".output.dense.bias"))->getOutput(0);
    auto* attn_fcw = net->addMatrixMultiply(*attn_out->getOutput(0), M::kNONE, *out_proj_w, M::kTRANSPOSE);
    auto* attn_fcb = net->addElementWise(*attn_fcw->getOutput(0), *out_proj_b, E::kSUM);
    attn_fcb->setName((attn_name + ".out_proj").c_str());

    /** 3. attention and hidden state residual connection */
    auto* attn_residual = net->addElementWise(input, *attn_fcb->getOutput(0), E::kSUM);
    attn_residual->setName((name + "attn_residual").c_str());

    /**  4. layer norm after attention */
    auto post_ln_name = name + ".layernorm_after";
    ln_scale = net->addConstant(Dims3{1, 1, dims.d[dims.nbDims - 1]}, w[post_ln_name + ".weight"]);
    ln_bias = net->addConstant(Dims3{1, 1, dims.d[dims.nbDims - 1]}, w[post_ln_name + ".bias"]);
    auto* _res = attn_residual->getOutput(0);
    axes = 1U << static_cast<uint32_t>(_res->getDimensions().nbDims - 1);
    auto* post_lnorm = addLinearNorm(net, *_res, *ln_scale->getOutput(0), *ln_bias->getOutput(0), axes);

    /** 6. intermediate (feed-forward) layer and activation */
    auto intermediate_name = name + ".intermediate.dense";
    std::cout << "Building: " << intermediate_name << "\n";
    auto* iw = net->addConstant(Dims3{1, F, H}, w[intermediate_name + ".weight"]);
    auto* ib = net->addConstant(Dims3{1, 1, F}, w[intermediate_name + ".bias"]);
    ib->setName((intermediate_name + ".bias").c_str());
    auto* inter0 = net->addMatrixMultiply(*post_lnorm->getOutput(0), M::kNONE, *iw->getOutput(0), M::kTRANSPOSE);
    auto* inter1 = net->addElementWise(*inter0->getOutput(0), *ib->getOutput(0), E::kSUM);
    auto* inter_act = addGeLU(net, *inter1->getOutput(0));

    /** 7. output projection */
    auto output_name = name + ".output.dense";
    std::cout << "Building: " << output_name << "\n";
    auto* ow = net->addConstant(Dims3{1, H, F}, w[output_name + ".weight"]);
    auto* ob = net->addConstant(Dims3{1, 1, H}, w[output_name + ".bias"]);
    ob->setName((output_name + ".bias").c_str());
    auto* out0 = net->addMatrixMultiply(*inter_act->getOutput(0), M::kNONE, *ow->getOutput(0), M::kTRANSPOSE);
    auto* out1 = net->addElementWise(*out0->getOutput(0), *ob->getOutput(0), E::kSUM);

    /** 8. residual */
    auto* output_residual = net->addElementWise(*out1->getOutput(0), *attn_residual->getOutput(0), E::kSUM);
    output_residual->setName((name + ".output_residual").c_str());
    return output_residual->getOutput(0);
}

// Creat the engine using only the API without any parser.
auto createEngine(const ViTConfig& cfg, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config,
                  DataType dt) -> ICudaEngine* {
    WeightMap w = loadWeights(cfg.wts_path);
    if (dt == DataType::kHALF) {
        convertWeightMapToHalf(w);
    }

    RepeatedWeightsStorage repeated_storage;

#if TRT_VERSION_GE(10, 0, 0)
    auto* net = builder->createNetworkV2(1U << static_cast<uint32_t>(NDCF::kSTRONGLY_TYPED));
#else
    auto* net = builder->createNetworkV2(1U << static_cast<int>(NDCF::kEXPLICIT_BATCH));
#endif

    // 1. patch embedding
    Dims input_dims{.nbDims = 4, .d = {-1, 3, cfg.img_size, cfg.img_size}};
    ITensor* data = net->addInput(NAMES[0], dt, input_dims);
    std::string name = "vit.embeddings.patch_embeddings.projection.";
    auto* embed = net->addConvolutionNd(*data, cfg.hidden, DimsHW{cfg.patch, cfg.patch}, w[name + "weight"],
                                        w[name + "bias"]);
    embed->setName("patch embedding");
    embed->setStrideNd(DimsHW{cfg.patch, cfg.patch});
    auto* s = net->addShuffle(*embed->getOutput(0));
    s->setReshapeDimensions(Dims3{0, cfg.hidden, cfg.num_patches()});
    s->setSecondTranspose({0, 2, 1});

    // 2. add cls token and position embedding
    auto cls_weights = makeRepeatedBatchWeights(w["vit.embeddings.cls_token"], cfg.max_batch, repeated_storage);
    auto pos_weights =
            makeRepeatedBatchWeights(w["vit.embeddings.position_embeddings"], cfg.max_batch, repeated_storage);
    auto* cls_full = net->addConstant(Dims3{cfg.max_batch, 1, cfg.hidden}, cls_weights);
    auto* pos_full = net->addConstant(Dims3{cfg.max_batch, cfg.seq_len(), cfg.hidden}, pos_weights);
    auto* cls_token = addBatchedConstantSlice(net, *data, *cls_full->getOutput(0), 1, cfg.hidden, repeated_storage);
    auto* pos_embed =
            addBatchedConstantSlice(net, *data, *pos_full->getOutput(0), cfg.seq_len(), cfg.hidden, repeated_storage);
    const std::array<ITensor*, 2> _cat = {cls_token, s->getOutput(0)};
    auto* cat = net->addConcatenation(_cat.data(), 2);
    cat->setAxis(1);
    cat->setName("cat_clstoken_embed");
    auto* pos_added = net->addElementWise(*cat->getOutput(0), *pos_embed, ElementWiseOperation::kSUM);
    pos_added->setName("position_embed");

    // 3. transformer encoder layers
    ITensor* input = pos_added->getOutput(0);
    for (int64_t i = 0; i < cfg.num_layers; i++) {
        auto* vit = ViTLayer(net, w, *input,
                             {.index = static_cast<uint32_t>(i),
                              .head_num = static_cast<uint32_t>(cfg.num_heads),
                              .hidden = cfg.hidden,
                              .ff_dim = cfg.ff_dim,
                              .lnorm_eps = cfg.lnorm_eps});
        input = vit;
    }

    // 4. layer norm after transformer encoder
    auto* ln_scale = net->addConstant(Dims3{1, 1, cfg.hidden}, w["vit.layernorm.weight"]);
    auto* ln_bias = net->addConstant(Dims3{1, 1, cfg.hidden}, w["vit.layernorm.bias"]);
    uint32_t axes = 1U << static_cast<uint32_t>(input->getDimensions().nbDims - 1);
    auto* post_lnorm = addLinearNorm(net, *input, *ln_scale->getOutput(0), *ln_bias->getOutput(0), axes);
    // 6. classifier head -- take CLS token (index 0 along seq axis) via Gather
    static int32_t cls_idx_data = 0;
    Weights cls_idx_w{DataType::kINT32, &cls_idx_data, 1};
    Dims idx_dims;
    idx_dims.nbDims = 1;
    idx_dims.d[0] = 1;
    auto* idx_const = net->addConstant(idx_dims, cls_idx_w)->getOutput(0);
    auto* gather = net->addGather(*post_lnorm->getOutput(0), *idx_const, 1);
    auto* shuffle = net->addShuffle(*gather->getOutput(0));
    shuffle->setReshapeDimensions(Dims2{-1, cfg.hidden});
    auto* cls_w = net->addConstant(DimsHW{cfg.num_classes, cfg.hidden}, w["classifier.weight"]);
    auto* cls_b = net->addConstant(DimsHW{1, cfg.num_classes}, w["classifier.bias"]);
    auto* cls_0 = net->addMatrixMultiply(*shuffle->getOutput(0), M::kNONE, *cls_w->getOutput(0), M::kTRANSPOSE);
    auto* cls_1 = net->addElementWise(*cls_b->getOutput(0), *cls_0->getOutput(0), E::kSUM);
    cls_1->getOutput(0)->setName(NAMES[1]);
    net->markOutput(*cls_1->getOutput(0));

#if TRT_VERSION_GE(8, 0, 0)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    config->setBuilderOptimizationLevel(5);
#if TRT_VERSION_LT(10, 0, 0)
    // Strongly-typed networks (TRT 10+) take their precision from the
    // tensor types declared on inputs/weights and reject BuilderFlag::kFP16.
    // Pre-TRT-10 networks are weakly typed, so explicitly request FP16
    // tactics when BUILD_PRECISION == kHALF; otherwise TRT may fall back
    // to FP32 kernels even though the inputs/weights are half.
    if (BUILD_PRECISION == DataType::kHALF) {
        config->setFlag(BuilderFlag::kFP16);
    }
#endif
    Dims _min{.nbDims = 4, .d = {cfg.min_batch, 3, cfg.img_size, cfg.img_size}};
    Dims _opt{.nbDims = 4, .d = {cfg.opt_batch, 3, cfg.img_size, cfg.img_size}};
    Dims _max{.nbDims = 4, .d = {cfg.max_batch, 3, cfg.img_size, cfg.img_size}};
    auto* profile = builder->createOptimizationProfile();
    profile->setDimensions(NAMES[0], OptProfileSelector::kMIN, _min);
    profile->setDimensions(NAMES[0], OptProfileSelector::kOPT, _opt);
    profile->setDimensions(NAMES[0], OptProfileSelector::kMAX, _max);
    config->addOptimizationProfile(profile);
    IHostMemory* mem = builder->buildSerializedNetwork(*net, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete net;
#else
    builder->setMaxBatchSize(cfg.max_batch);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*net, *config);
    net->destroy();
#endif

    std::cout << "build finished\n";
    // Release host memory
    for (auto& mem : w) {
        if (mem.second.values == nullptr) {
            continue;
        }
        if (mem.second.type == DataType::kHALF) {
            delete[] reinterpret_cast<const half*>(mem.second.values);
        } else {
            // loadWeights() allocates with new uint32_t[]
            delete[] reinterpret_cast<const uint32_t*>(mem.second.values);
        }
    }

    return engine;
}

std::vector<std::vector<float>> doInference(IExecutionContext& context, const void* input, std::size_t batchSize,
                                            const ViTConfig& cfg) {
    const ICudaEngine& engine = context.getEngine();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<void*> buffers;
#if TRT_VERSION_GE(10, 0, 0)
    auto allocator = CudaOutputAllocator::Create(stream);
#endif

#if TRT_VERSION_GE(8, 0, 0)
    const int32_t nIO = engine.getNbIOTensors();
#else
    const int32_t nIO = engine.getNbBindings();
#endif

    // SIZES per IO: input is C*H*W per sample; output is num_classes per sample.
    const int64_t in_per_sample = 3LL * cfg.img_size * cfg.img_size;
    const int64_t out_per_sample = cfg.num_classes;
    auto sizeOf = [&](int i) -> int64_t {
        return i == 0 ? in_per_sample : out_per_sample;
    };

#if TRT_VERSION_GE(8, 0, 0)
    if (engine.getTensorShape(NAMES[0]).d[0] == -1) {
        Dims in_dims{.nbDims = 4, .d = {static_cast<int64_t>(batchSize), 3, cfg.img_size, cfg.img_size}};
        if (!context.setInputShape(NAMES[0], in_dims)) {
            std::cerr << "setInputShape failed batch=" << batchSize << "\n";
            std::abort();
        }
    }
#endif

    buffers.resize(nIO, nullptr);
    for (auto i = 0; i < nIO; ++i) {

#if TRT_VERSION_GE(8, 0, 0)
        auto* tensor_name = engine.getIOTensorName(i);
        const auto dtype = engine.getTensorDataType(tensor_name);
        std::size_t size = batchSize * sizeOf(i) * bytesPerElement(dtype);
#if TRT_VERSION_GE(10, 0, 0)
        if (engine.getTensorIOMode(tensor_name) == TensorIOMode::kINPUT) {
            CHECK(cudaMalloc(&buffers[i], size));
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
            if (!context.setTensorAddress(tensor_name, buffers[i])) {
                std::cerr << "setTensorAddress failed\n";
                std::abort();
            }
        } else {
            context.setOutputAllocator(tensor_name, allocator.get());
        }
#else
        if (engine.getTensorIOMode(tensor_name) == TensorIOMode::kINPUT) {
            CHECK(cudaMalloc(&buffers[i], size));
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        } else {
            CHECK(cudaMalloc(&buffers[i], size));
        }
        if (!context.setTensorAddress(tensor_name, buffers[i])) {
            std::cerr << "setTensorAddress failed\n";
            std::abort();
        }
#endif
#else
        std::size_t size = batchSize * sizeOf(i) * sizeof(float);
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        assert(idx == i);
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
#endif
    }

#if TRT_VERSION_GE(8, 0, 0)
    if (!context.enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed\n";
        std::abort();
    }
#else
    if (!context.enqueueV2(buffers.data(), stream, nullptr)) {
        std::cerr << "enqueueV2 failed\n";
        std::abort();
    }
#endif

    std::vector<std::vector<float>> prob;
    for (int i = 0; i < nIO; ++i) {
#if TRT_VERSION_GE(8, 0, 0)
        auto* tensor_name = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(tensor_name) != TensorIOMode::kOUTPUT)
            continue;
        const auto dtype = engine.getTensorDataType(tensor_name);
        std::size_t count = batchSize * out_per_sample;
        std::size_t size = count * bytesPerElement(dtype);
#if TRT_VERSION_GE(10, 0, 0)
        void* out_ptr = allocator->getBuffer(tensor_name);
#else
        void* out_ptr = buffers[i];
#endif
        if (dtype == DataType::kHALF) {
            std::vector<__half> tmp_h(count);
            CHECK(cudaMemcpyAsync(tmp_h.data(), out_ptr, size, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            std::vector<float> tmp(count);
            for (std::size_t j = 0; j < tmp.size(); ++j)
                tmp[j] = __half2float(tmp_h[j]);
            prob.emplace_back(std::move(tmp));
        } else {
            std::vector<float> tmp(count, std::nanf(""));
            CHECK(cudaMemcpyAsync(tmp.data(), out_ptr, size, cudaMemcpyDeviceToHost, stream));
            prob.emplace_back(std::move(tmp));
        }
#else
        if (i == 0)
            continue;
        std::vector<float> tmp(batchSize * sizeOf(i), std::nanf(""));
        std::size_t size = batchSize * sizeOf(i) * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(std::move(tmp));
#endif
    }
    CHECK(cudaStreamSynchronize(stream));

    for (auto& buffer : buffers) {
        if (buffer != nullptr) {
            CHECK(cudaFree(buffer));
        }
    }
#if TRT_VERSION_GE(10, 0, 0)
    allocator.reset();
#endif
    CHECK(cudaStreamDestroy(stream));
    return prob;
}

void APIToModel(const ViTConfig& cfg, IRuntime* runtime, IHostMemory** modelStream) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    ICudaEngine* engine = createEngine(cfg, runtime, builder, config, BUILD_PRECISION);
    assert(engine != nullptr);

    (*modelStream) = engine->serialize();

#if TRT_VERSION_GE(8, 0, 0)
    delete engine;
    delete config;
    delete builder;
#else
    engine->destroy();
    config->destroy();
    builder->destroy();
#endif
}

// Recover ViTConfig fields needed by inference (img_size, num_classes) from the
// deserialized engine. The other fields (hidden/layers/heads/ff/patch) are baked
// into the engine and not needed at runtime.
static ViTConfig configFromEngine(const ICudaEngine& engine) {
    ViTConfig cfg;
    auto in_dims = engine.getTensorShape(NAMES[0]);   // [N or -1, 3, H, W]
    auto out_dims = engine.getTensorShape(NAMES[1]);  // [N or -1, num_classes]
    if (in_dims.nbDims != 4 || out_dims.nbDims != 2) {
        std::cerr << "Unexpected engine IO shapes\n";
        std::abort();
    }
    cfg.img_size = in_dims.d[2];
    cfg.num_classes = out_dims.d[1];
    if (in_dims.d[0] == -1) {
        auto pmin = engine.getProfileShape(NAMES[0], 0, OptProfileSelector::kMIN);
        auto popt = engine.getProfileShape(NAMES[0], 0, OptProfileSelector::kOPT);
        auto pmax = engine.getProfileShape(NAMES[0], 0, OptProfileSelector::kMAX);
        cfg.min_batch = pmin.d[0];
        cfg.opt_batch = popt.d[0];
        cfg.max_batch = pmax.d[0];
    } else {
        cfg.min_batch = in_dims.d[0];
        cfg.opt_batch = in_dims.d[0];
        cfg.max_batch = in_dims.d[0];
    }
    return cfg;
}

static auto loadLabels() -> std::map<int, std::string> {
    auto labels = loadImagenetLabelMap(LABELS_PATH);
    if (labels.empty()) {
        std::cerr << "failed to load labels: " << LABELS_PATH << "\n";
        std::abort();
    }
    return labels;
}

// Collect image files from a path: if a directory, return all *.jpg/*.jpeg/*.png
// files; if a single file, return just that file.
static std::vector<std::string> collectImages(const std::string& path) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    fs::path p(path);
    if (!fs::exists(p)) {
        std::cerr << "image path does not exist: " << path << "\n";
        std::abort();
    }
    if (fs::is_regular_file(p)) {
        out.push_back(path);
    } else if (fs::is_directory(p)) {
        for (auto& e : fs::directory_iterator(p)) {
            if (!e.is_regular_file())
                continue;
            auto ext = e.path().extension().string();
            for (auto& c : ext) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                out.push_back(e.path().string());
            }
        }
        std::sort(out.begin(), out.end());
    }
    return out;
}

auto main(int argc, char** argv) -> int {
    std::cout << "TensorRT version: " << TRT_VERSION << "\n";
    if (argc < 2 || (std::string(argv[1]) == "-s" && argc != 5) ||
        (std::string(argv[1]) == "-d" && argc != 4 && argc != 5)) {
        std::cerr << "usage:\n"
                  << "  ./vit -s <wts_path> <engine_path> <model_type>\n"
                  << "  ./vit -d <engine_path> <image_dir> [--profile]\n"
                  << "  model_type: ViT-B/16 | ViT-B/32 | ViT-L/16 | ViT-L/32 | ViT-H/14\n"
                  << "              (aliases: b16, b32, l16, l32, h14 also accepted)\n"
                  << "  build precision is configured in code via BUILD_PRECISION (see top of vit.cc).\n";
        return 1;
    }
    std::string mode = argv[1];

#ifndef NDEBUG
    gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kINFO);
#endif
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    if (mode == "-s") {
        std::string wts_path = argv[2];
        std::string engine_path = argv[3];
        std::string model_type = argv[4];
        ViTConfig cfg = getVariantConfig(model_type);
        cfg.wts_path = wts_path;
        cfg.engine_path = engine_path;
        const char* prec_str = (BUILD_PRECISION == DataType::kHALF) ? "fp16" : "fp32";
        std::cout << "[cfg] " << cfg.model_type << " hidden=" << cfg.hidden << " layers=" << cfg.num_layers
                  << " heads=" << cfg.num_heads << " ff=" << cfg.ff_dim << " patch=" << cfg.patch
                  << " img=" << cfg.img_size << " classes=" << cfg.num_classes
                  << " batch[min/opt/max]=" << cfg.min_batch << "/" << cfg.opt_batch << "/" << cfg.max_batch
                  << " precision=" << prec_str << "\n[wts]    " << wts_path << "\n[engine] " << engine_path << "\n";

        IHostMemory* modelStream{nullptr};
        APIToModel(cfg, runtime, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_path, std::ios::binary | std::ios::trunc);
        if (!p) {
            std::cerr << "could not open plan output file: " << engine_path << "\n";
            return -1;
        }
        const auto* data_ptr = reinterpret_cast<const char*>(modelStream->data());
        auto data_size = static_cast<std::streamsize>(modelStream->size());
        p.write(data_ptr, data_size);
#if TRT_VERSION_GE(8, 0, 0)
        delete modelStream;
#else
        modelStream->destroy();
#endif
        return 0;
    } else if (mode == "-d") {
        std::string engine_path = argv[2];
        std::string image_dir = argv[3];
        bool enable_profile = false;
        if (argc == 5) {
            if (std::string(argv[4]) != "--profile") {
                std::cerr << "unknown option: " << argv[4] << "\n";
                return 1;
            }
            enable_profile = true;
        }

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read engine file error: " << engine_path << "\n";
            return -1;
        }
        file.seekg(0, file.end);
        std::streamsize size = file.tellg();
        file.seekg(0, file.beg);
        if (size <= 0) {
            std::cerr << "empty engine file: " << engine_path << "\n";
            return -1;
        }
        std::vector<char> trtModelStream(static_cast<std::size_t>(size));
        file.read(trtModelStream.data(), size);
        file.close();

#if TRT_VERSION_GE(8, 0, 0)
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
#else
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
#endif
        assert(engine != nullptr);
        trtModelStream.clear();
        trtModelStream.shrink_to_fit();
        auto* context = engine->createExecutionContext();
        assert(context != nullptr);
        Profiler profiler("VisionTransformerProfiler");
        if (enable_profile) {
            context->setProfiler(&profiler);
        }

        ViTConfig cfg = configFromEngine(*engine);
        std::cout << "[engine] img=" << cfg.img_size << " classes=" << cfg.num_classes
                  << " batch[min/opt/max]=" << cfg.min_batch << "/" << cfg.opt_batch << "/" << cfg.max_batch << "\n";

        auto images = collectImages(image_dir);
        if (images.empty()) {
            std::cerr << "no images found under: " << image_dir << "\n";
            return -1;
        }
        // Cap batch to engine's max profile.
        int64_t infer_batch = std::min<int64_t>(static_cast<int64_t>(images.size()), cfg.max_batch);
        std::cout << "[infer] found " << images.size() << " image(s), using batch=" << infer_batch << "\n";

        std::vector<__half> input_buf;
        input_buf.reserve(infer_batch * 3 * cfg.img_size * cfg.img_size);
        for (int64_t i = 0; i < infer_batch; ++i) {
            auto img = cv::imread(images[i], cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "cannot read image: " << images[i] << "\n";
                return -1;
            }
            auto one = preprocess_img(img, false, mean, stdv, 1, toI32(cfg.img_size), toI32(cfg.img_size));
            input_buf.insert(input_buf.end(), one.begin(), one.end());
        }

        // Match the engine's declared input dtype. The engine input is FP16 when
        // built with `-s ... fp16` (default) and FP32 when built with `... fp32`.
        const auto in_dtype = engine->getTensorDataType(NAMES[0]);
        std::vector<float> input_buf_f32;
        const void* input_ptr = nullptr;
        if (in_dtype == DataType::kHALF) {
            input_ptr = input_buf.data();
        } else if (in_dtype == DataType::kFLOAT) {
            input_buf_f32.resize(input_buf.size());
            for (std::size_t k = 0; k < input_buf.size(); ++k)
                input_buf_f32[k] = __half2float(input_buf[k]);
            input_ptr = input_buf_f32.data();
        } else {
            std::cerr << "unsupported engine input dtype\n";
            return -1;
        }

        for (int i = 0; i < 5; ++i) {
            (void)doInference(*context, input_ptr, infer_batch, cfg);
        }

        auto firstProb = doInference(*context, input_ptr, infer_batch, cfg);
        auto labels = (cfg.num_classes == 1000) ? loadLabels() : std::map<int, std::string>{};
        for (int64_t b = 0; b < infer_batch; ++b) {
            std::cout << "[sample " << b << "] " << images[b] << "\n";
            std::vector<float> sample(firstProb[0].begin() + b * cfg.num_classes,
                                      firstProb[0].begin() + (b + 1) * cfg.num_classes);
            printFirstOutputs("vit", sample.data(), sample.size());
            int _top = 0;
            for (auto& [idx, logits] : topk(sample, 3)) {
                std::cout << "  Top: " << _top++ << " idx: " << idx << ", logits: " << logits;
                if (!labels.empty()) {
                    std::cout << ", label: " << labels[idx];
                }
                std::cout << "\n";
            }
        }

        std::vector<double> latencies;
        latencies.reserve(kBenchmarkRuns);
        for (int i = 0; i < kBenchmarkRuns; ++i) {
            auto start = std::chrono::steady_clock::now();
            (void)doInference(*context, input_ptr, infer_batch, cfg);
            auto end = std::chrono::steady_clock::now();
            auto period = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            latencies.push_back(static_cast<double>(period.count()) / 1000.0);
        }
        printBenchmark("vit", latencies, infer_batch);
        if (enable_profile) {
            std::cout << profiler;
        }
        return 0;
    }
    return 0;
}
