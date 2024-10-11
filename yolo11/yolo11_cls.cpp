#include "calibrator.h"
#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kClsNumClass;

void batch_preprocess(std::vector<cv::Mat>& imgs, float* output, int dst_width = 224, int dst_height = 224) {
    for (size_t b = 0; b < imgs.size(); b++) {
        int h = imgs[b].rows;
        int w = imgs[b].cols;
        int m = std::min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::Mat img = imgs[b](cv::Rect(left, top, m, m));
        cv::resize(img, img, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        // CHW format
        for (int c = 0; c < 3; ++c) {
            int i = 0;
            for (int row = 0; row < dst_height; ++row) {
                for (int col = 0; col < dst_width; ++col) {
                    output[b * 3 * dst_height * dst_width + c * dst_height * dst_width + i] =
                            channels[c].at<float>(row, col);
                    ++i;
                }
            }
        }
    }
}

std::vector<float> softmax(float* prob, int n) {
    std::vector<float> res;
    float sum = 0.0f;
    float t;
    for (int i = 0; i < n; i++) {
        t = expf(prob[i]);
        res.push_back(t);
        sum += t;
    }
    for (int i = 0; i < n; i++) {
        res[i] /= sum;
    }
    return res;
}

std::vector<int> topk(const std::vector<float>& vec, int k) {
    std::vector<int> topk_index;
    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(),
              [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), k);

    for (int i = 0; i < k_num; ++i) {
        topk_index.push_back(vec_index[i]);
    }

    return topk_index;
}

std::vector<std::string> read_classes(std::string file_name) {
    std::vector<std::string> classes;
    std::ifstream ifs(file_name, std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << file_name << " is not found, pls refer to README and download it." << std::endl;
        assert(0);
    }
    std::string s;
    while (std::getline(ifs, s)) {
        classes.push_back(s);
    }
    ifs.close();
    return classes;
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw,
                std::string& img_dir, std::string& type, int& max_channels) {
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        } else if (net[0] == 's') {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        } else if (net[0] == 'm') {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        } else if (net[0] == 'x') {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_input_buffer,
                     float** output_buffer_host) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kClsInputH * kClsInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    *cpu_input_buffer = new float[kBatchSize * 3 * kClsInputH * kClsInputW];
    *output_buffer_host = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output,
           int batchSize) {
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * kClsInputH * kClsInputW * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);
}

void serialize_engine(float& gd, float& gw, std::string& wts_name, std::string& engine_name, std::string& type,
                      int max_channels) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    IHostMemory* serialized_engine = nullptr;
    //engine = buildEngineYolo11Cls(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    serialized_engine = buildEngineYolo11Cls(builder, config, DataType::kFLOAT, wts_name, gd, gw, type, max_channels);
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

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
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

int main(int argc, char** argv) {
    // yolo11_cls -s ../models/yolo11n-cls.wts ../models/yolo11n-cls.fp32.trt n
    // yolo11_cls -d ../models/yolo11n-cls.fp32.trt ../images
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    std::string type;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir, type, max_channels)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolo11_cls -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolo11_cls -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(gd, gw, wts_name, engine_name, type, max_channels);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Prepare cpu and gpu buffers
    float* device_buffers[2];
    float* cpu_input_buffer = nullptr;
    float* output_buffer_host = nullptr;
    prepare_buffers(engine, &device_buffers[0], &device_buffers[1], &cpu_input_buffer, &output_buffer_host);

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // Read imagenet labels
    auto classes = read_classes("imagenet_classes.txt");

    // batch predict
    for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
        // Get a batch of images
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;
        for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }

        // Preprocess
        batch_preprocess(img_batch, cpu_input_buffer);

        // Run inference
        auto start = std::chrono::system_clock::now();
        infer(*context, stream, (void**)device_buffers, cpu_input_buffer, output_buffer_host, kBatchSize);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        // Postprocess and get top-k result
        for (size_t b = 0; b < img_name_batch.size(); b++) {
            float* p = &output_buffer_host[b * kOutputSize];
            auto res = softmax(p, kOutputSize);
            auto topk_idx = topk(res, 3);
            std::cout << img_name_batch[b] << std::endl;
            for (auto idx : topk_idx) {
                std::cout << "  " << classes[idx] << " " << res[idx] << std::endl;
            }
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    delete[] cpu_input_buffer;
    delete[] output_buffer_host;
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
    return 0;
}
