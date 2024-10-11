
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(std::string& wts_name, std::string& engine_name, float& gd, float& gw, int& max_channels,
                      std::string& type) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine = nullptr;

    serialized_engine = buildEngineYolo11Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

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
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device,
                    std::string cuda_post_process) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueueV2(buffers, stream, nullptr);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(
                cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);  //cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                   stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir, std::string& type,
                std::string& cuda_post_process, float& gd, float& gw, int& max_channels) {
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto sub_type = std::string(argv[4]);

        if (sub_type[0] == 'n') {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        } else if (sub_type[0] == 's') {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        } else if (sub_type[0] == 'm') {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        } else if (sub_type[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        } else if (sub_type[0] == 'x') {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // yolo11_det -s ../models/yolo11n.wts ../models/yolo11n.fp32.trt n
    // yolo11_det -d ../models/yolo11n.fp32.trt ../images c
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    std::string img_dir;
    std::string cuda_post_process;
    std::string type;
    int model_bboxes;
    float gd = 0, gw = 0;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, cuda_post_process, gd, gw, max_channels)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo11_det -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo11_det -d [.engine] ../images  [c/g]// deserialize plan file and run inference"
                  << std::endl;
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
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device = nullptr;

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host,
                   &decode_ptr_device, cuda_post_process);

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
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        // Run inference
        infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
              decode_ptr_device, model_bboxes, cuda_post_process);
        // 保存output_buffer_host的前100个值，一行一个
        //        std::ofstream out("../models/output.txt");
        //        for (int j = 0; j < 100; j++) {
        //            out << output_buffer_host[j] << std::endl;
        //        }
        //        out.close();
        std::vector<std::vector<Detection>> res_batch;
        if (cuda_post_process == "c") {
            // NMS
            batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        } else if (cuda_post_process == "g") {
            //Process gpu decode and nms results
            batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
        }
        // Draw bounding boxes
        draw_bbox(img_batch, res_batch);
        // Save images
        for (size_t j = 0; j < img_batch.size(); j++) {
            cv::imwrite("_" + img_name_batch[j], img_batch[j]);
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;

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
