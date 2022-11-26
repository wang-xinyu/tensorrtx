#include "config.h"
#include "model.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000  // max input image buffer size

using namespace nvinfer1;

const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
static Logger gLogger;

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string& wts_name, std::string& sub_type) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = nullptr;

    if (sub_type == "t") {
        engine = build_engine_yolov7_tiny(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "v7") {
        engine = build_engine_yolov7(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "x") {
        engine = build_engine_yolov7x(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "w6") {
        engine = build_engine_yolov7w6(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "e6") {
        engine = build_engine_yolov7e6(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "d6") {
        engine = build_engine_yolov7d6(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "e6e") {
        engine = build_engine_yolov7e6e(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir, std::string& sub_type) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && argc == 5) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        sub_type = std::string(argv[4]);
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir;
    std::string sub_type = "";

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolov7 -s [.wts] [.engine] [t/v7/x/w6/e6/d6/e6e]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov7 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream = nullptr;

        APIToModel(BATCH_SIZE, &modelStream, wts_name, sub_type);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    static float prob[BATCH_SIZE * kOutputSize];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * kOutputSize * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t *img_host = nullptr;
    uint8_t *img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    int fcount = 0;
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        //auto start = std::chrono::system_clock::now();
        float* buffer_idx = (float*)buffers[inputIndex];
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            imgs_buffer[b] = img;
            size_t size_image = img.cols * img.rows * 3;
            size_t size_image_dst = kInputH * kInputW * 3;
            //copy data to pinned memory
            memcpy(img_host, img.data, size_image);
            //copy data to device memory
            CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, kInputW, kInputH, stream);
            buffer_idx += size_image_dst;
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * kOutputSize], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            cv::Mat img = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < kOutputSize; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
