#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "utils.h"

using namespace nvinfer1;

static Logger gLogger;
// const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kInputSize = 3 * 256 * 256;
const static int kOutputSize = 1 * 256 * 256;

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw,
                std::string& img_dir) {
    if (argc != 4)
        return false;
    if (std::string(argv[1]) == "-s") {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
    } else if (std::string(argv[1]) == "-d") {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

void prepare_infer_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                           float** cpu_output_buffer) {
    // assert(engine->getNbIOTensors() == 2);
    assert(engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    // nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create GPU in/output buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * 1 * kOutputSize * sizeof(float)));  // 3 or 1 ??
    // Create CPU output buffers on host
    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void preprocessImg(cv::Mat& img, int newh, int neww) {
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(neww, newh));
    img.convertTo(img, CV_32FC3);
    // ImageNet normalize
    img /= 255.0f;
    img -= cv::Scalar(0.485, 0.456, 0.406);
    img /= cv::Scalar(0.229, 0.224, 0.225);
}

void infer(IExecutionContext& context, cudaStream_t& stream, std::vector<void*>& gpu_buffers,
           std::vector<float>& cpu_input_data, std::vector<float>& cpu_output_data, int batchsize) {
    // copy input data from host (CPU) to device (GPU)
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[0], cpu_input_data.data(), cpu_input_data.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    // execute inference using context provided by engine
    context.enqueue(batchsize, gpu_buffers.data(), stream, nullptr);
    // copy output back from device (GPU) to host (CPU)
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_data.data(), gpu_buffers[1], batchsize * kOutputSize * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    // synchronize the stream to prevent issues (block CUDA and wait for CUDA operations to be completed)
    cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name,
                      std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = nullptr;
    engine = build_efficientAD_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    config->destroy();
    serialized_engine->destroy();
    builder->destroy();
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
    assert(*engine != nullptr);
    *context = (*engine)->createExecutionContext();
    assert(*context);

    delete[] serialized_engine;
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 1.0f, gw = 1.0f;
    std::string img_dir;

    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./efficientad_det -s [.wts] [.engine]  // serialize model to plan file" << std::endl;
        std::cerr
                << "./efficientad_det -d [.engine] [../../datas/images/...]  // deserialize plan file and run inference"
                << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(kBatchSize, gd, gw, wts_name, engine_name);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);

    // create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // prepare cpu and gpu buffers
    void *gpu_input_buffer, *gpu_output_buffer;
    CUDA_CHECK(cudaMalloc(&gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_output_buffer, kBatchSize * 1 * kOutputSize * sizeof(float)));  // 3 or 1 ??
    std::vector<void*> gpu_buffers = {gpu_input_buffer, gpu_output_buffer};
    std::vector<float> cpu_input_data(kBatchSize * kInputSize, 0);
    std::vector<float> cpu_output_data(kBatchSize * kOutputSize, 0);

    // read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    std::vector<cv::Mat> originImg_batch;
    for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
        // get a batch of images
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;

        for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            originImg_batch.push_back(img.clone());
            preprocessImg(img, kInputW, kInputH);
            assert(img.cols * img.rows * 3 == 3 * 256 * 256);
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < img.rows; h++) {
                    for (int w = 0; w < img.cols; w++) {
                        cpu_input_data[c * img.rows * img.cols + h * img.cols + w] = img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        // infer(*context, stream, (void**)gpu_buffers, cpu_input_data, cpu_output_buffer, kBatchSize);
        infer(*context, stream, gpu_buffers, cpu_input_data, cpu_output_data,
              kBatchSize);  // change to save into vec `cpu_output_data`
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        // postProcess
        cv::Mat img_1(256, 256, CV_8UC1);
        for (int row = 0; row < 256; row++) {
            for (int col = 0; col < 256; col++) {
                float value = cpu_output_data[row * 256 + col];
                if (value < 0)  // clip(0,1)
                    value = 0;
                else if (value > 1)
                    value = 1;
                img_1.at<uchar>(row, col) = static_cast<uchar>(value * 255);
            }
        }

        cv::Mat HeatMap, colorMap;
        // genHeatMap(img_batch[0], img_1, HeatMap);
        cv::applyColorMap(img_1, colorMap, cv::COLORMAP_JET);
        cv::resize(originImg_batch[i], originImg_batch[i], cv::Size(256, 256));
        cv::cvtColor(originImg_batch[i], originImg_batch[i], cv::COLOR_RGB2BGR);
        cv::addWeighted(originImg_batch[i], 0.5, colorMap, 0.5, 0, HeatMap);

        // Save images
        for (size_t j = 0; j < img_batch.size(); j++) {
            cv::imwrite("_output" + img_name_batch[j], img_1);
            cv::imwrite("_heatmap" + img_name_batch[j], HeatMap);
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
