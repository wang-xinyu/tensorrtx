#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
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

static constexpr const int64_t N = 1;
static constexpr const int64_t INPUT_H = 224;
static constexpr const int64_t INPUT_W = 224;

static constexpr const char* WTS_PATH = "../models/vit.wts";
static constexpr const char* ENGINE_PATH = "../models/vit.engine";
static constexpr const char* LABELS_PATH = "../assets/imagenet1000_clsidx_to_labels.txt";
static constexpr const std::array<const char*, 2> NAMES = {"input", "logits"};
static constexpr const std::array<int64_t, 2> SIZES = {3 * INPUT_H * INPUT_W, 1000};
static constexpr const std::array<const float, 3> mean = {0.5f, 0.5f, 0.5f};
static constexpr const std::array<const float, 3> stdv = {0.5f, 0.5f, 0.5f};

static Logger gLogger;

static auto bytesPerElement(DataType t) -> std::size_t {
    switch (t) {
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT32:
            return 4;
#if TRT_VERSION >= 8000
        case DataType::kBOOL:
#endif
#if TRT_VERSION >= 8500
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
    float lnorm_eps = 1e-12f;
};

static auto addGeLU(INetworkDefinition* net, ITensor& input) -> ILayer* {
#if TRT_VERSION < 10000
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
    auto* _w_half = net->addConstant(scalarDims, Weights{DataType::kFLOAT, &_half, 1});
    auto* _w_one = net->addConstant(scalarDims, Weights{DataType::kFLOAT, &_one, 1});
    auto* _w_sqrt_2_div_pi = net->addConstant(scalarDims, Weights{DataType::kFLOAT, &_sqrt_2_div_pi, 1});
    auto* _w_coeff = net->addConstant(scalarDims, Weights{DataType::kFLOAT, &_coeff, 1});

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
#if TRT_VERSION >= 11500
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
    int64_t attn_head_size = 768LL / param.head_num;

    auto* qw = net->addConstant(Dims3{1, 768, 768}, w.at(attn_name + ".attention.query.weight"));
    auto* kw = net->addConstant(Dims3{1, 768, 768}, w.at(attn_name + ".attention.key.weight"));
    auto* vw = net->addConstant(Dims3{1, 768, 768}, w.at(attn_name + ".attention.value.weight"));
    /* 1. layer norm before attention */
    auto pre_ln_name = name + ".layernorm_before";
    auto dims = input.getDimensions();
    uint32_t axes = 1U << static_cast<uint32_t>(dims.nbDims - 1);
    auto* ln_scale = net->addConstant(Dims3{1, 1, dims.d[dims.nbDims - 1]}, w[pre_ln_name + ".weight"]);
    auto* ln_bias = net->addConstant(Dims3{1, 1, dims.d[dims.nbDims - 1]}, w[pre_ln_name + ".bias"]);
    auto* pre_lnorm = addLinearNorm(net, input, *ln_scale->getOutput(0), *ln_bias->getOutput(0), axes);

    /** 2. multi-head self-attention */
    auto* qb = net->addConstant(Dims3{1, 1, 768}, w.at(attn_name + ".attention.query.bias"));
    auto* kb = net->addConstant(Dims3{1, 1, 768}, w.at(attn_name + ".attention.key.bias"));
    auto* vb = net->addConstant(Dims3{1, 1, 768}, w.at(attn_name + ".attention.value.bias"));
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
#if TRT_VERSION >= 11400 && TRT_VERSION < 11500
    gLogger.log(Severity::kWARNING,
                "IAttention is available in TensorRT 10.14.1 SDK but have bugs, use 10.15.1+ to enable native fused "
                "kernel");
#endif
#if TRT_VERSION >= 11500
    using ANO = AttentionNormalizationOp;
    auto* q_scaled = net->addElementWise(*q_s->getOutput(0), *qk_scale_w->getOutput(0), E::kPROD)->getOutput(0);
    auto* attn = net->addAttention(*q_scaled, *k_s->getOutput(0), *v_s->getOutput(0), ANO::kSOFTMAX, false);
    assert(attn != nullptr);
    auto status = attn->setDecomposable(false);
    assert(status);
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
    attn_out->setReshapeDimensions(Dims3{0, 0, 768});
    // 2.4 attention output projection
    auto* out_proj_w = net->addConstant(Dims3{1, 768, 768}, w.at(attn_name + ".output.dense.weight"))->getOutput(0);
    auto* out_proj_b = net->addConstant(Dims3{1, 1, 768}, w.at(attn_name + ".output.dense.bias"))->getOutput(0);
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
    auto* iw = net->addConstant(Dims3{1, 3072, 768}, w[intermediate_name + ".weight"]);
    auto* ib = net->addConstant(Dims3{1, 1, 3072}, w[intermediate_name + ".bias"]);
    ib->setName((intermediate_name + ".bias").c_str());
    auto* inter0 = net->addMatrixMultiply(*post_lnorm->getOutput(0), M::kNONE, *iw->getOutput(0), M::kTRANSPOSE);
    auto* inter1 = net->addElementWise(*inter0->getOutput(0), *ib->getOutput(0), E::kSUM);
    auto* inter_act = addGeLU(net, *inter1->getOutput(0));

    /** 7. output projection */
    auto output_name = name + ".output.dense";
    std::cout << "Building: " << output_name << "\n";
    auto* ow = net->addConstant(Dims3{1, 768, 3072}, w[output_name + ".weight"]);
    auto* ob = net->addConstant(Dims3{1, 1, 768}, w[output_name + ".bias"]);
    ob->setName((output_name + ".bias").c_str());
    auto* out0 = net->addMatrixMultiply(*inter_act->getOutput(0), M::kNONE, *ow->getOutput(0), M::kTRANSPOSE);
    auto* out1 = net->addElementWise(*out0->getOutput(0), *ob->getOutput(0), E::kSUM);

    /** 8. residual */
    auto* output_residual = net->addElementWise(*out1->getOutput(0), *attn_residual->getOutput(0), E::kSUM);
    output_residual->setName((name + ".output_residual").c_str());
    return output_residual->getOutput(0);
}

// Creat the engine using only the API without any parser.
auto createEngine(int64_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config,
                  DataType dt) -> ICudaEngine* {
    WeightMap w = loadWeights(WTS_PATH);
    if (dt == DataType::kHALF) {
        convertWeightMapToHalf(w);
    }

#if TRT_VERSION >= 10000
    auto* net = builder->createNetworkV2(1U << static_cast<uint32_t>(NDCF::kSTRONGLY_TYPED));
#else
    auto* net = builder->createNetworkV2(1U << static_cast<int>(NDCF::kEXPLICIT_BATCH));
#endif

    // 1. patch embedding
    ITensor* data = net->addInput(NAMES[0], dt, Dims4{-1, 3, INPUT_H, INPUT_W});
    std::string name = "vit.embeddings.patch_embeddings.projection.";
    auto* embed = net->addConvolutionNd(*data, 768, DimsHW{16, 16}, w[name + "weight"], w[name + "bias"]);
    embed->setName("patch embedding");
    embed->setStrideNd(DimsHW{16, 16});
    auto* s = net->addShuffle(*embed->getOutput(0));
    s->setReshapeDimensions(Dims3{0, 768, 14LL * 14});
    s->setSecondTranspose({0, 2, 1});

    // 2. add cls token and position embedding
    auto* cls_token = net->addConstant(Dims3{1, 1, 768}, w["vit.embeddings.cls_token"]);
    auto* pos_embed = net->addConstant(Dims3{1, 197, 768}, w["vit.embeddings.position_embeddings"]);
    const std::array<ITensor*, 2> _cat = {cls_token->getOutput(0), s->getOutput(0)};
    auto* cat = net->addConcatenation(_cat.data(), 2);
    cat->setAxis(1);
    cat->setName("cat_clstoken_embed");
    auto* pos_added = net->addElementWise(*cat->getOutput(0), *pos_embed->getOutput(0), ElementWiseOperation::kSUM);
    pos_added->setName("position_embed");

    // 3. transformer encoder layers
    ITensor* input = pos_added->getOutput(0);
    for (auto i = 0u; i < 12; i++) {
        auto* vit = ViTLayer(net, w, *input, {.index = i, .head_num = 12, .lnorm_eps = 1e-12f});
        input = vit;
    }

    // 4. layer norm after transformer encoder
    auto* ln_scale = net->addConstant(Dims3{1, 1, 768}, w["vit.layernorm.weight"]);
    auto* ln_bias = net->addConstant(Dims3{1, 1, 768}, w["vit.layernorm.bias"]);
    uint32_t axes = 1U << static_cast<uint32_t>(input->getDimensions().nbDims - 1);
    auto* post_lnorm = addLinearNorm(net, *input, *ln_scale->getOutput(0), *ln_bias->getOutput(0), axes);
    // 6. classifier head
    auto* slice = net->addSlice(*post_lnorm->getOutput(0), Dims3{0, 0, 0}, Dims3{N, 1, 768}, Dims3{1, 1, 1});
    auto* shuffle = net->addShuffle(*slice->getOutput(0));
    shuffle->setReshapeDimensions(Dims2{N, 768});
    auto* cls_w = net->addConstant(DimsHW{1000, 768}, w["classifier.weight"]);
    auto* cls_b = net->addConstant(DimsHW{1, 1000}, w["classifier.bias"]);
    auto* cls_0 = net->addMatrixMultiply(*shuffle->getOutput(0), M::kNONE, *cls_w->getOutput(0), M::kTRANSPOSE);
    auto* cls_1 = net->addElementWise(*cls_b->getOutput(0), *cls_0->getOutput(0), E::kSUM);
    net->markOutput(*cls_1->getOutput(0));

    Dims4 _min{1, 3, INPUT_H, INPUT_W}, _opt{N, 3, INPUT_H, INPUT_W}, _max{2 * N, 3, INPUT_H, INPUT_W};
#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    config->setHardwareCompatibilityLevel(HardwareCompatibilityLevel::kAMPERE_PLUS);
    auto* profile = builder->createOptimizationProfile();
    profile->setDimensions(NAMES[0], OptProfileSelector::kMIN, _min);
    profile->setDimensions(NAMES[0], OptProfileSelector::kOPT, _opt);
    profile->setDimensions(NAMES[0], OptProfileSelector::kMAX, _max);
    config->addOptimizationProfile(profile);
    IHostMemory* mem = builder->buildSerializedNetwork(*net, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete net;
#else
    builder->setMaxBatchSize(N);
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

std::vector<std::vector<float>> doInference(IExecutionContext& context, __half* input, std::size_t batchSize) {
    const ICudaEngine& engine = context.getEngine();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<void*> buffers;
#if TRT_VERSION >= 10000
    auto allocator = CudaOutputAllocator::Create(stream);
#endif

#if TRT_VERSION >= 8000
    const int32_t nIO = engine.getNbIOTensors();
#else
    const int32_t nIO = engine.getNbBindings();
#endif

    buffers.resize(nIO, nullptr);
    for (auto i = 0; i < nIO; ++i) {

#if TRT_VERSION >= 8000
        // TensorRT 8+ use name based SDK
        auto* tensor_name = engine.getIOTensorName(i);
        const auto dtype = engine.getTensorDataType(tensor_name);
        std::size_t size = batchSize * SIZES[i] * bytesPerElement(dtype);
#if TRT_VERSION >= 10000
        // TensorRT 10+ use outuput allocator
        if (i == 0) {
            CHECK(cudaMalloc(&buffers[i], size));
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
            context.setTensorAddress(tensor_name, buffers[i]);
        } else {
            context.setOutputAllocator(tensor_name, allocator.get());
        }
#else
        if (i != 0) {
            CHECK(cudaMalloc(&buffers[i], size));
        } else {
            CHECK(cudaMalloc(&buffers[i], size));
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
        context.setTensorAddress(tensor_name, buffers[i]);
#endif
#else
        std::size_t size = batchSize * SIZES[i] * sizeof(float);
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        assert(idx == i);
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
#endif
    }

#if TRT_VERSION >= 8000
    assert(context.enqueueV3(stream));
#else
    assert(context.enqueueV2(buffers.data(), stream, nullptr));
#endif

    std::vector<std::vector<float>> prob;
    for (int i = 1; i < nIO; ++i) {
#if TRT_VERSION >= 10000
        auto* tensor_name = engine.getIOTensorName(i);
        const auto dtype = engine.getTensorDataType(tensor_name);
        std::size_t size = batchSize * SIZES[i] * bytesPerElement(dtype);
        void* out_ptr = allocator->getBuffer(tensor_name);
        // D2H data transfer
        if (dtype == DataType::kHALF) {
            std::vector<__half> tmp_h(batchSize * SIZES[i]);
            CHECK(cudaMemcpyAsync(tmp_h.data(), out_ptr, size, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            std::vector<float> tmp(batchSize * SIZES[i]);
            for (std::size_t j = 0; j < tmp.size(); ++j) {
                tmp[j] = __half2float(tmp_h[j]);
            }
            prob.emplace_back(std::move(tmp));
        } else {
            std::vector<float> tmp(batchSize * SIZES[i], std::nanf(""));
            CHECK(cudaMemcpyAsync(tmp.data(), out_ptr, size, cudaMemcpyDeviceToHost, stream));
            prob.emplace_back(std::move(tmp));
        }
#else
        std::vector<float> tmp(batchSize * SIZES[i], std::nanf(""));
        std::size_t size = batchSize * SIZES[i] * sizeof(float);
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
#if TRT_VERSION >= 10000
    allocator.reset();
#endif
    CHECK(cudaStreamDestroy(stream));
    return prob;
}

void APIToModel(int32_t N, IRuntime* runtime, IHostMemory** modelStream) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    ICudaEngine* engine = createEngine(N, runtime, builder, config, DataType::kHALF);
    assert(engine != nullptr);

    (*modelStream) = engine->serialize();

#if TRT_VERSION >= 8000
    delete engine;
    delete config;
    delete builder;
#else
    engine->destroy();
    config->destroy();
    builder->destroy();
#endif
}

auto main(int argc, char** argv) -> int {
    std::cout << "TensorRT version: " << TRT_VERSION << "\n";
    if (argc != 2) {
        std::cerr << "arguments not right!\n";
        std::cerr << "./vit -s  // serialize model to plan file\n";
        std::cerr << "./vit -d  // deserialize plan file and run inference\n";

        return 1;
    }
#ifndef NDEBUG
    gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);
#endif
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trtModelStream{nullptr};
    std::streamsize size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(N, runtime, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(ENGINE_PATH, std::ios::binary | std::ios::trunc);
        if (!p) {
            std::cerr << "could not open plan output file\n";
            return -1;
        }
        if (modelStream->size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::cerr << "this model is too large to serialize\n";
            return -1;
        }
        const auto* data_ptr = reinterpret_cast<const char*>(modelStream->data());
        auto data_size = static_cast<std::streamsize>(modelStream->size());
        p.write(data_ptr, data_size);
#if TRT_VERSION >= 8000
        delete modelStream;
#else
        modelStream->destroy();
#endif
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file(ENGINE_PATH, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        } else {
            std::cerr << "read engine file error!\n";
            return -1;
        }

#if TRT_VERSION >= 8000
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
        assert(engine != nullptr);
        auto* context = engine->createExecutionContext();
        assert(context != nullptr);

        // VIT use default BGR order
        auto img = cv::imread("../assets/cats.jpg", cv::IMREAD_COLOR);
        auto input = preprocess_img(img, false, mean, stdv, N, INPUT_H, INPUT_W);

        Profiler profiler("VisionTransformerProfiler");

        // Warmup: run a few iterations without profiling.
        for (int i = 0; i < 5; ++i) {
            (void)doInference(*context, input.data(), N);
        }

        // Profiled runs
        context->setProfiler(&profiler);
        for (int i = 0; i < 20; ++i) {
            auto start = std::chrono::system_clock::now();
            auto prob = doInference(*context, input.data(), N);
            auto end = std::chrono::system_clock::now();
            auto period = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << period.count() << "us\n";

            for (const auto& vector : prob) {
                int idx = 0;
                for (auto v : vector) {
                    std::cout << std::setprecision(4) << v << ", " << std::flush;
                    if (++idx > 20) {
                        std::cout << "\n====\n";
                        break;
                    }
                }
            }

            if (i == 19) {
                std::cout << "prediction result: \n";
                auto labels = loadImagenetLabelMap(LABELS_PATH);
                int _top = 0;
                for (auto& [idx, logits] : topk(prob[0], 3)) {
                    std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits
                              << ", label: " << labels[idx] << "\n";
                }
                std::cout << profiler << "\n";
            }
        }
        return 0;
    }
    return 0;
}
