#include <fstream>
#include <iostream>
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
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(const std::string& wts_name, std::string& engine_name, float& gd, float& gw, int& max_channels,
                      std::string& type) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine =
            buildEngineYolo26Obb(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

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
                    float** output_buffer_host) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);

    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kObbInputH * kObbInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    *output_buffer_host = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           int model_bboxes) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueueV2(buffers, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    std::string img_dir;
    std::string type;
    int model_bboxes;
    float gd = 0.0f, gw = 0.0f;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, gd, gw, max_channels)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo26_obb -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolo26_obb -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
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
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    // Prepare cpu and gpu buffers
    float* device_buffers[2];
    float* output_buffer_host = nullptr;

    setPluginDeviceParams(kConfThresh);

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host);

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
        cuda_batch_preprocess(img_batch, device_buffers[0], kObbInputW, kObbInputH, stream);

        infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize, model_bboxes);

        std::vector<std::vector<Detection>> res_batch;
        batch_decode_obb(res_batch, output_buffer_host, img_batch.size(), kOutputSize);
        draw_bbox_obb(img_batch, res_batch);
        // Save images
        for (size_t j = 0; j < img_batch.size(); j++) {
            cv::imwrite("_" + img_name_batch[j], img_batch[j]);
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;

    return 0;
}