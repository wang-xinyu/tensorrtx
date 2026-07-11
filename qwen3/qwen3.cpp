#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "NvInfer.h"
#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "tokenizer.hpp"
#include "utils.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

using namespace nvinfer1;
// static Logger gLogger(nvinfer1::ILogger::Severity::kVERBOSE);
static Logger gLogger;

[[noreturn]] void fail_trt(const std::string& msg) {
    std::cerr << "[ERROR] " << msg << std::endl;
    std::abort();
}

std::string strip_assistant_reply(std::string text) {
    const std::string end_tag = "<|im_end|>";
    const size_t end_pos = text.find(end_tag);
    if (end_pos != std::string::npos) {
        text.erase(end_pos);
    }
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

std::string remove_think_block(std::string text) {
    const std::string think_begin = "<think>";
    const std::string think_end = "</think>";
    while (true) {
        const size_t begin = text.find(think_begin);
        if (begin == std::string::npos) {
            break;
        }
        const size_t end = text.find(think_end, begin);
        if (end == std::string::npos) {
            text.erase(begin);
            break;
        }
        text.erase(begin, end + think_end.size() - begin);
    }
    while (!text.empty() && (text.front() == '\n' || text.front() == '\r' || text.front() == ' ')) {
        text.erase(text.begin());
    }
    return text;
}

void strip_utf8_bom_inplace(std::string& text) {
    static const std::string kUtf8Bom = "\xEF\xBB\xBF";
    if (text.rfind(kUtf8Bom, 0) == 0) {
        text.erase(0, kUtf8Bom.size());
    }
}

void append_utf8_text(const std::string& path, const std::string& text) {
    const bool need_bom = !std::ifstream(path, std::ios::binary).good();
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[ERROR] failed to open file for utf-8 append: " << path << std::endl;
        return;
    }
    if (need_bom) {
        static const unsigned char kUtf8Bom[] = {0xEF, 0xBB, 0xBF};
        out.write(reinterpret_cast<const char*>(kUtf8Bom), sizeof(kUtf8Bom));
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

#ifdef _WIN32
std::string wide_to_utf8(const std::wstring& wide) {
    if (wide.empty()) {
        return {};
    }
    const int size =
            WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string utf8(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(), size, nullptr, nullptr);
    return utf8;
}

bool read_console_utf8_line(std::string& out) {
    if (!_isatty(_fileno(stdin))) {
        if (!std::getline(std::cin, out)) {
            return false;
        }
        strip_utf8_bom_inplace(out);
        return true;
    }
    std::wstring wide;
    if (!std::getline(std::wcin, wide)) {
        return false;
    }
    out = wide_to_utf8(wide);
    strip_utf8_bom_inplace(out);
    return true;
}
#else
bool read_console_utf8_line(std::string& out) {
    return static_cast<bool>(std::getline(std::cin, out));
}
#endif

void serialize_engine(const std::string& wts_name, const std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    IHostMemory* serialized_engine = nullptr;
    serialized_engine = buildEngineQwen3_0_6B(builder, config, DataType::kFLOAT, wts_name);
    assert(serialized_engine);
    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    delete serialized_engine;
    delete config;
    delete builder;
}

void deserialize_engine(const std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

template <typename T>
std::vector<T> read_input(const std::string& input_data_path) {
    std::ifstream infile(input_data_path);
    if (!infile.is_open()) {
        std::cerr << "error: unable to open file input_data.txt" << std::endl;
        return {};
    }
    std::vector<T> data;
    T value;

    while (infile >> value) {
        data.push_back(value);
    }
    infile.close();
    return data;
}

template <typename T>
void save_output(const std::string& output_data_path, const std::vector<T>& data) {
    std::ofstream outfile(output_data_path);
    if (!outfile.is_open()) {
        std::cerr << "error: unable to open file output_data.txt" << std::endl;
        return;
    }
    for (const auto& value : data) {
        outfile << value << "\n";
    }
    outfile.close();
}

std::string get_format_shape(const Dims& shape) {
    std::stringstream output;
    char buffer[64];
    const char* fmts[] = {"%d", "x%d"};
    for (int i = 0; i < shape.nbDims; ++i) {
        snprintf(buffer, sizeof(buffer), fmts[i != 0], shape.d[i]);
        output << buffer;
    }
    return output.str();
}

std::string dtype2string(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT:
            return "FLOAT";
        case DataType::kHALF:
            return "HALF";
        case DataType::kINT8:
            return "INT8";
        case DataType::kINT32:
            return "INT32";
        case DataType::kBOOL:
            return "BOOL";
        case DataType::kUINT8:
            return "UINT8";
        case DataType::kFP8:
            return "FP8";
        case DataType::kBF16:
            return "BF16";
        case DataType::kINT64:
            return "INT64";
        case DataType::kINT4:
            return "INT4";
        case DataType::kFP4:
            return "FP4";
        default:
            return "UNKNOWN";
    }
}

std::vector<int32_t> build_position_ids(int seq_len) {
    std::vector<int32_t> position_ids(seq_len);
    std::iota(position_ids.begin(), position_ids.end(), 0);
    return position_ids;
}

std::vector<float> compute_default_inv_freq() {
    const int rotary_dim = static_cast<int>(static_cast<float>(kHeadDim) * kPartialRotaryFactor);
    assert(rotary_dim > 0);
    assert(rotary_dim <= kHeadDim);
    assert(rotary_dim % 2 == 0);

    std::vector<float> inv_freq(rotary_dim / 2);
    for (int i = 0; i < rotary_dim / 2; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
        inv_freq[i] = 1.0f / std::pow(kRopeTheta, exponent);
    }
    return inv_freq;
}

std::pair<std::vector<float>, std::vector<float>> build_default_rope(const std::vector<int32_t>& position_ids) {
    const int seq_len = static_cast<int>(position_ids.size());
    const int rotary_dim = static_cast<int>(static_cast<float>(kHeadDim) * kPartialRotaryFactor);
    const int half_dim = rotary_dim / 2;
    const auto inv_freq = compute_default_inv_freq();

    std::vector<float> cos(seq_len * kHeadDim, 1.0f);
    std::vector<float> sin(seq_len * kHeadDim, 0.0f);

    for (int pos = 0; pos < seq_len; ++pos) {
        const float position = static_cast<float>(position_ids[pos]);
        for (int i = 0; i < half_dim; ++i) {
            const float angle = position * inv_freq[i];
            const float cos_value = std::cos(angle) * kRopeAttentionScaling;
            const float sin_value = std::sin(angle) * kRopeAttentionScaling;
            cos[pos * kHeadDim + i] = cos_value;
            cos[pos * kHeadDim + half_dim + i] = cos_value;
            sin[pos * kHeadDim + i] = sin_value;
            sin[pos * kHeadDim + half_dim + i] = sin_value;
        }
    }

    return {cos, sin};
}

void print(const ICudaEngine* engine) {
    std::cout << "[INFO] Engine has " << engine->getNbIOTensors() << " bindings" << std::endl;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        auto name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            auto dtype = engine->getTensorDataType(name);
            auto dims = engine->getTensorShape(name);
            std::cout << "[INFO] Input " << i << " name: " << name << " dtype: " << dtype2string(dtype)
                      << " dims: " << get_format_shape(dims) << std::endl;
        } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            auto dtype = engine->getTensorDataType(name);
            auto dims = engine->getTensorShape(name);
            std::cout << "[INFO] Output " << i << " name: " << name << " dtype: " << dtype2string(dtype)
                      << " dims: " << get_format_shape(dims) << std::endl;
        } else {
            std::cerr << "[INFO] Unknown tensor I/O mode" << std::endl;
            assert(false);
        }
    }
}

unsigned int cumprod(const Dims& dims) {
    unsigned int sum = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        sum *= dims.d[i];
    }
    return sum;
}

void print_topk_logits(const std::vector<float>& logits, int topk) {
    std::vector<std::pair<int, float>> id_logit_pairs;
    id_logit_pairs.reserve(logits.size());
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        id_logit_pairs.emplace_back(i, logits[i]);
    }

    if (topk > static_cast<int>(id_logit_pairs.size())) {
        topk = static_cast<int>(id_logit_pairs.size());
    }

    std::partial_sort(
            id_logit_pairs.begin(), id_logit_pairs.begin() + topk, id_logit_pairs.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });
}

std::pair<int32_t, float> select_top1(const std::vector<float>& logits) {
    assert(!logits.empty());
    auto iter = std::max_element(logits.begin(), logits.end());
    const int32_t token_id = static_cast<int32_t>(std::distance(logits.begin(), iter));
    return {token_id, *iter};
}

bool is_eos_token(int32_t token_id) {
    return std::find(kEosTokenIds.begin(), kEosTokenIds.end(), token_id) != kEosTokenIds.end();
}

std::pair<int32_t, float> sample_token(const std::vector<float>& logits, std::mt19937& rng) {
    assert(!logits.empty());

    std::vector<std::pair<int32_t, float>> candidates;
    candidates.reserve(logits.size());
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i) {
        candidates.emplace_back(i, logits[i] / kTemperature);
    }

    if (kTopK > 0 && kTopK < static_cast<int>(candidates.size())) {
        std::partial_sort(candidates.begin(), candidates.begin() + kTopK, candidates.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        candidates.resize(kTopK);
    } else {
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    float max_logit = candidates.front().second;
    std::vector<float> probs(candidates.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp(candidates[i].second - max_logit);
        sum += probs[i];
    }
    for (float& prob : probs) {
        prob /= sum;
    }

    if (kTopP < 1.0f) {
        float cumulative = 0.0f;
        size_t keep_count = 0;
        for (; keep_count < probs.size(); ++keep_count) {
            cumulative += probs[keep_count];
            if (cumulative >= kTopP) {
                ++keep_count;
                break;
            }
        }
        keep_count = std::max<size_t>(1, std::min(keep_count, probs.size()));
        candidates.resize(keep_count);
        probs.resize(keep_count);
        float renorm = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float& prob : probs) {
            prob /= renorm;
        }
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    const int selected_index = dist(rng);
    const int32_t token_id = candidates[selected_index].first;
    const float selected_prob = probs[selected_index];
    return {token_id, selected_prob};
}

// (Mask tensor, valid position is 0, invalid position is 1. Will automatically broadcast in the H_q dimension)
std::vector<float> make_causal_mask(int seq) {
    std::vector<float> mask(seq * seq, 0.0f);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            if (j > i) {
                mask[i * seq + j] = 1.0f;
            }
        }
    }
    return mask;
}

class Qwen3PrefillRunner {
   public:
    Qwen3PrefillRunner(const std::string& engine_path, int max_seq_len) : mMaxSeqLen(max_seq_len), mRng(42) {
        deserialize_engine(const_cast<std::string&>(engine_path), &mRuntime, &mEngine, &mContext);
        assert(mRuntime != nullptr);
        assert(mEngine != nullptr);
        assert(mContext != nullptr);
        print(mEngine);
        CUDA_CHECK(cudaStreamCreate(&mStream));
        allocate_buffers();
    }

    ~Qwen3PrefillRunner() {
        release_buffers();
        if (mStream != nullptr) {
            cudaStreamDestroy(mStream);
        }
        delete mContext;
        delete mEngine;
        delete mRuntime;
    }

    std::vector<float> run(const std::vector<int32_t>& input_ids) {
        assert(!input_ids.empty());
        assert(static_cast<int>(input_ids.size()) <= mMaxSeqLen);

        const int seq = static_cast<int>(input_ids.size());
        const auto position_ids = build_position_ids(seq);
        auto rope = build_default_rope(position_ids);
        auto mask = make_causal_mask(seq);

        set_shapes(seq);
        upload_inputs(input_ids, rope.first, rope.second, mask);
        bind_tensors();

        if (!mContext->enqueueV3(mStream)) {
            fail_trt("enqueueV3 failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));

        const auto output_shape = mContext->getTensorShape(kOutputTensorName);
        std::vector<float> output_data(cumprod(output_shape));
        CUDA_CHECK(cudaMemcpy(output_data.data(), mOutputDevice, output_data.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return output_data;
    }

    std::vector<int32_t> generate(std::vector<int32_t> input_ids) {
        assert(static_cast<int>(input_ids.size()) <= mMaxSeqLen);
        std::vector<int32_t> generated_ids;
        generated_ids.reserve(std::max(0, mMaxSeqLen - static_cast<int>(input_ids.size())));

        int step = 0;
        while (static_cast<int>(input_ids.size()) < mMaxSeqLen) {
            const auto logits = run(input_ids);
            const auto [token_id, score] = kDoSample ? sample_token(logits, mRng) : select_top1(logits);
            generated_ids.push_back(token_id);
            input_ids.push_back(token_id);
            if (is_eos_token(token_id)) {
                break;
            }
            ++step;
        }

        return generated_ids;
    }

   private:
    void allocate_buffers() {
        const size_t max_input_ids_bytes = static_cast<size_t>(mMaxSeqLen) * sizeof(int32_t);
        const size_t max_rope_bytes = static_cast<size_t>(mMaxSeqLen) * kHeadDim * sizeof(float);
        const size_t max_mask_bytes = static_cast<size_t>(mMaxSeqLen) * mMaxSeqLen * sizeof(float);
        const size_t max_output_bytes = static_cast<size_t>(kVocabSize) * sizeof(float);
        const size_t cache_elems_per_layer =
                static_cast<size_t>(kBatchSize) * kNumKeyValueHeads * mMaxSeqLen * kHeadDim;
        mCacheElemsPerLayer = cache_elems_per_layer;
        const size_t total_cache_bytes = static_cast<size_t>(kNumHiddenLayers) * cache_elems_per_layer * sizeof(float);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mInputIdsDevice), max_input_ids_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mCosDevice), max_rope_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mSinDevice), max_rope_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mMaskDevice), max_mask_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mOutputDevice), max_output_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mKeyCacheBuffer), total_cache_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mValueCacheBuffer), total_cache_bytes));
        CUDA_CHECK(cudaMemset(mKeyCacheBuffer, 0, total_cache_bytes));
        CUDA_CHECK(cudaMemset(mValueCacheBuffer, 0, total_cache_bytes));
    }

    void release_buffers() {
        if (mInputIdsDevice)
            CUDA_CHECK(cudaFree(mInputIdsDevice));
        if (mCosDevice)
            CUDA_CHECK(cudaFree(mCosDevice));
        if (mSinDevice)
            CUDA_CHECK(cudaFree(mSinDevice));
        if (mMaskDevice)
            CUDA_CHECK(cudaFree(mMaskDevice));
        if (mOutputDevice)
            CUDA_CHECK(cudaFree(mOutputDevice));
        if (mKeyCacheBuffer)
            CUDA_CHECK(cudaFree(mKeyCacheBuffer));
        if (mValueCacheBuffer)
            CUDA_CHECK(cudaFree(mValueCacheBuffer));
    }

    void set_shapes(int seq) {
        if (!mContext->setInputShape(kInputIdsName, Dims2{kBatchSize, seq})) {
            fail_trt("setInputShape failed for input_ids");
        }
        if (!mContext->setInputShape(kInputCosName, Dims3{kBatchSize, seq, kHeadDim})) {
            fail_trt("setInputShape failed for input_cos");
        }
        if (!mContext->setInputShape(kInputSinName, Dims3{kBatchSize, seq, kHeadDim})) {
            fail_trt("setInputShape failed for input_sin");
        }
        if (!mContext->setInputShape(kInputMaskName, Dims4{kBatchSize, 1, seq, seq})) {
            fail_trt("setInputShape failed for input_mask");
        }

        for (int layer_idx = 0; layer_idx < kNumHiddenLayers; ++layer_idx) {
            const std::string key_cache_input_name = getKeyCacheInputName(layer_idx);
            const std::string value_cache_input_name = getValueCacheInputName(layer_idx);
            if (!mContext->setInputShape(key_cache_input_name.c_str(),
                                         Dims4{kBatchSize, kNumKeyValueHeads, 0, kHeadDim})) {
                fail_trt("setInputShape failed for " + key_cache_input_name);
            }
            if (!mContext->setInputShape(value_cache_input_name.c_str(),
                                         Dims4{kBatchSize, kNumKeyValueHeads, 0, kHeadDim})) {
                fail_trt("setInputShape failed for " + value_cache_input_name);
            }
        }

        if (!mContext->allInputDimensionsSpecified()) {
            fail_trt("not all input dimensions are specified");
        }

        const int infer_status = mContext->inferShapes(0, nullptr);
        if (infer_status < 0) {
            fail_trt("inferShapes failed");
        }

        const auto output_shape = mContext->getTensorShape(kOutputTensorName);
        // std::cout << "[INFO] output shape: ";
        for (int i = 0; i < output_shape.nbDims; ++i) {
            // std::cout << output_shape.d[i] << " ";
            if (output_shape.d[i] < 0) {
                fail_trt("output shape still contains dynamic dim after inferShapes");
            }
        }
        // std::cout << std::endl;
    }

    void upload_inputs(const std::vector<int32_t>& input_ids, const std::vector<float>& cos,
                       const std::vector<float>& sin, const std::vector<float>& mask) {
        CUDA_CHECK(cudaMemcpy(mInputIdsDevice, input_ids.data(), input_ids.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mCosDevice, cos.data(), cos.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mSinDevice, sin.data(), sin.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mMaskDevice, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void bind_tensors() {
        mContext->setInputTensorAddress(kInputIdsName, mInputIdsDevice);
        mContext->setInputTensorAddress(kInputCosName, mCosDevice);
        mContext->setInputTensorAddress(kInputSinName, mSinDevice);
        mContext->setInputTensorAddress(kInputMaskName, mMaskDevice);
        mContext->setOutputTensorAddress(kOutputTensorName, mOutputDevice);

        for (int layer_idx = 0; layer_idx < kNumHiddenLayers; ++layer_idx) {
            const size_t layer_offset = static_cast<size_t>(layer_idx) * mCacheElemsPerLayer;
            float* key_cache_ptr = mKeyCacheBuffer + layer_offset;
            float* value_cache_ptr = mValueCacheBuffer + layer_offset;

            const std::string key_cache_input_name = getKeyCacheInputName(layer_idx);
            const std::string value_cache_input_name = getValueCacheInputName(layer_idx);
            const std::string key_cache_output_name = getKeyCacheOutputName(layer_idx);
            const std::string value_cache_output_name = getValueCacheOutputName(layer_idx);

            mContext->setInputTensorAddress(key_cache_input_name.c_str(), key_cache_ptr);
            mContext->setInputTensorAddress(value_cache_input_name.c_str(), value_cache_ptr);
            mContext->setOutputTensorAddress(key_cache_output_name.c_str(), key_cache_ptr);
            mContext->setOutputTensorAddress(value_cache_output_name.c_str(), value_cache_ptr);
        }
    }

   private:
    int mMaxSeqLen{0};
    size_t mCacheElemsPerLayer{0};

    IRuntime* mRuntime{nullptr};
    ICudaEngine* mEngine{nullptr};
    IExecutionContext* mContext{nullptr};
    cudaStream_t mStream{nullptr};

    int32_t* mInputIdsDevice{nullptr};
    float* mCosDevice{nullptr};
    float* mSinDevice{nullptr};
    float* mMaskDevice{nullptr};
    float* mOutputDevice{nullptr};
    float* mKeyCacheBuffer{nullptr};
    float* mValueCacheBuffer{nullptr};
    std::mt19937 mRng;
};

#if 0
int run_dialog_mode() {
    const std::string engine_model = "../models/test.engine";
    const std::string dialog_log_path = "../models/dialog_utf8.txt";

    auto tokenizer = tokenizer::AutoTokenizer::from_pretrained(R"(F:\LLM\modelscope\hub\models\Qwen\Qwen3-0.6B)");
    assert(tokenizer != nullptr);

    const std::string system_prompt = "You are a professional AI assistant, please answer the user's questions in Chinese.";
    std::string conversation = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    append_utf8_text(dialog_log_path, "===== New Session =====\n");
    append_utf8_text(dialog_log_path, "[system]\n" + system_prompt + "\n\n");

    Qwen3PrefillRunner runner(engine_model, kMaxSeqLen);
    std::cout << "[INFO] dialog mode ready, type `exit` to quit." << std::endl;

    while (true) {
        std::cout << "\nuser> " << std::flush;
        std::string user_input;
        if (!read_console_utf8_line(user_input)) {
            break;
        }
        if (user_input == "exit" || user_input == "quit") {
            break;
        }
        if (user_input.empty()) {
            continue;
        }

        const std::string prompt = conversation
            + "<|im_start|>user\n" + user_input + "<|im_end|>\n"
            + "<|im_start|>assistant\n";
        std::vector<int> encoded_ids = tokenizer->encode(prompt);
        std::vector<int32_t> input_ids(encoded_ids.begin(), encoded_ids.end());
        if (input_ids.empty()) {
            std::cerr << "[ERROR] empty prompt ids" << std::endl;
            continue;
        }
        if (static_cast<int>(input_ids.size()) >= kMaxSeqLen) {
            std::cerr << "[ERROR] prompt too long: " << input_ids.size()
                      << " >= " << kMaxSeqLen << std::endl;
            continue;
        }

        auto generated_ids = runner.generate(input_ids);
        std::vector<int> generated_ids_int(generated_ids.begin(), generated_ids.end());
        // const std::string assistant_reply = remove_think_block(
        // strip_assistant_reply(tokenizer->decode(generated_ids_int, false)));
        const std::string assistant_reply = strip_assistant_reply(tokenizer->decode(generated_ids_int, false));

        std::cout << "assistant> " << assistant_reply << std::endl;
        append_utf8_text(dialog_log_path, "[user]\n" + user_input + "\n");
        append_utf8_text(dialog_log_path, "[assistant]\n" + assistant_reply + "\n\n");

        conversation += "<|im_start|>user\n" + user_input + "<|im_end|>\n";
        conversation += "<|im_start|>assistant\n" + assistant_reply + "<|im_end|>\n";
    }

    return 0;
}
#endif

void print_usage(const char* exe_name) {
    std::cerr << "Arguments not right!" << std::endl;
    std::cerr << exe_name << " -s [qwen3_wts_dir] [.trt]" << std::endl;
    std::cerr << exe_name << " -d [.trt] [tokenizer_dir]" << std::endl;
}

bool parse_args(int argc, char** argv, std::string& wts_name, std::string& engine_name, std::string& tokenizer_dir,
                bool& is_serialize) {
    if (argc < 2) {
        return false;
    }

    const std::string mode = argv[1];
    if (mode == "-s") {
        if (argc != 4) {
            return false;
        }
        is_serialize = true;
        wts_name = argv[2];
        engine_name = argv[3];
        return true;
    }
    if (mode == "-d") {
        if (argc != 4) {
            return false;
        }
        is_serialize = false;
        wts_name.clear();
        engine_name = argv[2];
        tokenizer_dir = argv[3];
        return true;
    }
    return false;
}

int run_dialog_mode_cli(const std::string& engine_model, const std::string& tokenizer_dir) {
    const std::string dialog_log_path = "../models/dialog_utf8.txt";

    auto tokenizer = tokenizer::AutoTokenizer::from_pretrained(tokenizer_dir);
    assert(tokenizer != nullptr);

    const std::string system_prompt =
            "You are a professional AI assistant, please answer the user's questions in Chinese.";
    std::string conversation = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    append_utf8_text(dialog_log_path, "===== New Session =====\n");
    append_utf8_text(dialog_log_path, "[system]\n" + system_prompt + "\n\n");

    Qwen3PrefillRunner runner(engine_model, kMaxSeqLen);
    std::cout << "[INFO] dialog mode ready, type `exit` to quit." << std::endl;

    while (true) {
        std::cout << "\nuser> " << std::flush;
        std::string user_input;
        if (!read_console_utf8_line(user_input)) {
            break;
        }
        if (user_input == "exit" || user_input == "quit") {
            break;
        }
        if (user_input.empty()) {
            continue;
        }

        const std::string prompt =
                conversation + "<|im_start|>user\n" + user_input + "<|im_end|>\n" + "<|im_start|>assistant\n";
        std::vector<int> encoded_ids = tokenizer->encode(prompt);
        std::vector<int32_t> input_ids(encoded_ids.begin(), encoded_ids.end());
        if (input_ids.empty()) {
            std::cerr << "[ERROR] empty prompt ids" << std::endl;
            continue;
        }
        if (static_cast<int>(input_ids.size()) >= kMaxSeqLen) {
            std::cerr << "[ERROR] prompt too long: " << input_ids.size() << " >= " << kMaxSeqLen << std::endl;
            continue;
        }

        auto generated_ids = runner.generate(input_ids);
        std::vector<int> generated_ids_int(generated_ids.begin(), generated_ids.end());
        // const std::string assistant_reply = remove_think_block(strip_assistant_reply(tokenizer->decode(generated_ids_int, false));
        const std::string assistant_reply = strip_assistant_reply(tokenizer->decode(generated_ids_int, false));

        std::cout << "assistant> " << assistant_reply << std::endl;
        append_utf8_text(dialog_log_path, "[user]\n" + user_input + "\n");
        append_utf8_text(dialog_log_path, "[assistant]\n" + assistant_reply + "\n\n");

        conversation += "<|im_start|>user\n" + user_input + "<|im_end|>\n";
        conversation += "<|im_start|>assistant\n" + assistant_reply + "<|im_end|>\n";
    }

    return 0;
}

#if 0
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    if (_isatty(_fileno(stdin))) {
        _setmode(_fileno(stdin), _O_U16TEXT);
    }
#endif

    return run_dialog_mode();

    const std::string wts_path = "F:\\LLM\\transformers-4.57.6\\wts";
    const std::string engine_model = "../models/test.engine";

    serialize_engine(wts_path, engine_model);

    auto tokenizer = tokenizer::AutoTokenizer::from_pretrained(R"(F:\LLM\modelscope\hub\models\Qwen\Qwen3-0.6B)");
    assert(tokenizer != nullptr);

    const std::string prompt = "<|im_start|>system\n"
        "You are a professional AI assistant, please answer the user's questions in Chinese.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello! Can you introduce yourself?<|im_end|>\n"
        "<|im_start|>assistant\n";

    std::vector<int> encoded_ids = tokenizer->encode(prompt);
    std::vector<int32_t> input_ids(encoded_ids.begin(), encoded_ids.end());
    assert(!input_ids.empty());

    // std::cout << "[INFO] prompt ids: ";
    // for (int32_t id : input_ids) {
    //     std::cout << id << " ";
    // }
    // std::cout << std::endl;

    Qwen3PrefillRunner runner(engine_model, kMaxSeqLen);
    auto prefill_output = runner.run(input_ids);
    print_topk_logits(prefill_output, 10);
    save_output("../models/output.txt", prefill_output);

    auto generated_ids = runner.generate(input_ids);
    save_output("../models/generated_ids.txt", generated_ids);

    std::vector<int32_t> full_sequence = input_ids;
    full_sequence.insert(full_sequence.end(), generated_ids.begin(), generated_ids.end());
    save_output("../models/full_sequence.txt", full_sequence);

    std::vector<int> generated_ids_int(generated_ids.begin(), generated_ids.end());
    std::vector<int> full_sequence_int(full_sequence.begin(), full_sequence.end());

    std::cout << "[INFO] prompt text:\n" << tokenizer->decode(encoded_ids, false) << std::endl;
    // std::cout << "[INFO] generated ids: ";
    // for (int32_t id : generated_ids) {
        // std::cout << id << " ";
    // }
    std::cout << std::endl;
    // std::cout << "[INFO] generated text:\n" << tokenizer->decode(generated_ids_int, false) << std::endl;
    std::cout << "[INFO] full decoded text:\n" << tokenizer->decode(full_sequence_int, false) << std::endl;

    return 0;
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    if (_isatty(_fileno(stdin))) {
        _setmode(_fileno(stdin), _O_U16TEXT);
    }
#endif

    std::string wts_name;
    std::string engine_name;
    std::string tokenizer_dir;
    bool is_serialize = false;
    if (!parse_args(argc, argv, wts_name, engine_name, tokenizer_dir, is_serialize)) {
        print_usage(argv[0]);
        return -1;
    }

    if (is_serialize) {
        serialize_engine(wts_name, engine_name);
        return 0;
    }

    return run_dialog_mode_cli(engine_name, tokenizer_dir);
}
