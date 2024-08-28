#include <chrono>
#include <fstream>
#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

using namespace nvinfer1;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
static Logger gLogger;
void serialize_engine(unsigned int max_batchsize, std::string& wts_name, std::string& sub_type,
                      std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    IHostMemory* serialized_engine = nullptr;
    if (sub_type == "t") {
        serialized_engine = build_engine_yolov9_t(max_batchsize, builder, config, DataType::kFLOAT, wts_name, false);
    } else if (sub_type == "s") {
        serialized_engine = build_engine_yolov9_s(max_batchsize, builder, config, DataType::kFLOAT, wts_name, false);
    } else if (sub_type == "m") {
        serialized_engine = build_engine_yolov9_m(max_batchsize, builder, config, DataType::kFLOAT, wts_name, false);
    } else if (sub_type == "c") {
        serialized_engine = build_engine_yolov9_c(max_batchsize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "e") {
        serialized_engine = build_engine_yolov9_e(max_batchsize, builder, config, DataType::kFLOAT, wts_name);
    }

    else if (sub_type == "gt") {
        serialized_engine = build_engine_yolov9_t(max_batchsize, builder, config, DataType::kFLOAT, wts_name, true);
    } else if (sub_type == "gs") {
        serialized_engine = build_engine_yolov9_s(max_batchsize, builder, config, DataType::kFLOAT, wts_name, true);
    } else if (sub_type == "gm") {
        serialized_engine = build_engine_yolov9_m(max_batchsize, builder, config, DataType::kFLOAT, wts_name, true);
    } else if (sub_type == "gc") {
        serialized_engine = build_engine_gelan_c(max_batchsize, builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "ge") {
        serialized_engine = build_engine_gelan_e(max_batchsize, builder, config, DataType::kFLOAT, wts_name);
    } else {
        return;
    }

    assert(serialized_engine != nullptr);

    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    delete config;
    delete serialized_engine;
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
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    *output_buffer_host = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir,
                std::string& sub_type) {
    if (argc < 4)
        return false;
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
    cudaSetDevice(kGpuId);

    std::string wts_name = "";
    std::string engine_name = "../yolov9-m-converted.engine";
    std::string img_dir = "../images";
    std::string sub_type = "m";
    // speed test or inference
    const int speed_test_iter = 1000;
    // const int speed_test_iter = 1;

    // if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type)) {
    //     std::cerr << "Arguments not right!" << std::endl;
    //     std::cerr << "./yolov9 -s [.wts] [.engine] [s/m/c/e/gt/gs/gm/gc/ge]  // serialize model to plan file" << std::endl;
    //     std::cerr << "./yolov9 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
    //     return -1;
    // }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(kBatchSize, wts_name, sub_type, engine_name);
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

    // Prepare cpu and gpu buffers
    float* device_buffers[2];
    float* output_buffer_host = nullptr;
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host);

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

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
        auto start = std::chrono::system_clock::now();
        for (int j = 0; j < speed_test_iter; j++) {
            infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize);
        }
        // infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 /
                             speed_test_iter
                  << "ms" << std::endl;

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

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
