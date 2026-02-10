#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "types.h"
#include "utils.h"

#include "yololayer.h"

Logger gLogger;
using namespace nvinfer1;
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

void serialize_engine(const std::string& wts_name, std::string& engine_name, float& gd, float& gw, int& max_channels,
                      std::string& type) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine =
            buildEngineYolo26Cls(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
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

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** input_buffer_host, float** output_buffer_host) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kClsInputH * kClsInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    *input_buffer_host = new float[kBatchSize * 3 * kClsInputH * kClsInputW];
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

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    std::string img_dir;
    std::string type;
    int model_bboxes = 0;
    float gd = 0, gw = 0;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, gd, gw, max_channels)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo26_cls -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo26_cls -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, gd, gw, max_channels, type);
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
    float* input_buffer_host = nullptr;
    float* output_buffer_host = nullptr;
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &input_buffer_host, &output_buffer_host);

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
        batch_preprocess(img_batch, input_buffer_host, kClsInputW, kClsInputH);

        std::ofstream p("engine_input.txt");
        if (!p) {
            std::cout << "could not open input file" << std::endl;
            assert(false);
        }
        for (int i = 0; i < kBatchSize * 3 * kClsInputH * kClsInputW; i++) {
            p << input_buffer_host[i] << "\n";
        }
        p.close();

        // Run inference
        auto start = std::chrono::system_clock::now();
        infer(*context, stream, (void**)device_buffers, input_buffer_host, output_buffer_host, kBatchSize);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        // Postprocess and get top-k result
        for (size_t b = 0; b < img_name_batch.size(); b++) {
            float* p = &output_buffer_host[b * kOutputSize];
            std::vector<float> prob(p, p + kOutputSize);
            auto topk_idx = topk(prob, 3);
            std::cout << img_name_batch[b] << std::endl;
            for (auto idx : topk_idx) {
                std::cout << "  " << classes[idx] << " " << p[idx] << std::endl;
            }
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    delete[] input_buffer_host;
    delete[] output_buffer_host;
    delete context;
    delete engine;
    delete runtime;
    return 0;
}