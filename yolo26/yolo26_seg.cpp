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
const static int kOutputProtoSize = 32 * (kInputH / 4) * (kInputW / 4);

static cv::Rect get_downscale_rect(float bbox[4], float scale) {
    float left = bbox[0];
    float top = bbox[1];
    float right = bbox[2];
    float bottom = bbox[3];

    left = std::max(0.0f, left);
    top = std::max(0.0f, top);
    right = std::min((float)kInputW, right);
    bottom = std::min((float)kInputH, bottom);

    left /= scale;
    top /= scale;
    right /= scale;
    bottom /= scale;
    return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {
    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, 4);

        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                if (x < 0 || x >= mask_mat.cols || y < 0 || y >= mask_mat.rows) {
                    continue;
                }
                float e = 0.0f;
                for (int j = 0; j < 32; j++) {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}

void serialize_engine(const std::string& wts_name, std::string& engine_name, float& gd, float& gw, int& max_channels,
                      std::string& type) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine =
            buildEngineYolo26Seg(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

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
                    float** output_proto_buffer_device, float** output_buffer_host,
                    float** output_proto_buffer_host) {
    assert(engine->getNbBindings() == 3);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    const int outputProtoIndex = engine->getBindingIndex(kProtoTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    assert(outputProtoIndex == 2);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_proto_buffer_device, kBatchSize * kOutputProtoSize * sizeof(float)));

    *output_buffer_host = new float[kBatchSize * kOutputSize];
    *output_proto_buffer_host = new float[kBatchSize * kOutputProtoSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, float* output_proto,
           int batchsize, int model_bboxes) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueueV2(buffers, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(output_proto, buffers[2], batchsize * kOutputProtoSize * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

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
    int model_bboxes = 0;
    float gd = 0, gw = 0;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, gd, gw, max_channels)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo26_seg -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo26_seg -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
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
    float* device_buffers[3];
    float* output_buffer_host = nullptr;
    float* output_proto_buffer_host = nullptr;

    // WARN: If you change kMaxNumOutputBbox, it must be smaller than the value kMaxNumOutputBbox in config.h,
    // otherwise there will be memory overflow!
    // Or you should modify the config.h and recompile.
    setPluginDeviceParams(kConfThresh);

    // Read class labels, used by draw_mask_bbox to print the class name next to each box.
    std::unordered_map<int, std::string> labels_map;
    read_labels("coco.txt", labels_map);

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &device_buffers[2], &output_buffer_host,
                   &output_proto_buffer_host);

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
        infer(*context, stream, (void**)device_buffers, output_buffer_host, output_proto_buffer_host, kBatchSize,
              model_bboxes);

        std::vector<std::vector<Detection>> res_batch;
        batch_decode(res_batch, output_buffer_host, kBatchSize, kOutputSize);

        // Decode masks and draw them together with the boxes
        for (size_t b = 0; b < img_batch.size(); b++) {
            auto& res = res_batch[b];
            cv::Mat img = img_batch[b];
            auto masks = process_mask(&output_proto_buffer_host[b * kOutputProtoSize], kOutputProtoSize, res);
            draw_mask_bbox(img, res, masks, labels_map);
            cv::imwrite("_" + img_name_batch[b], img);
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(device_buffers[2]));
    delete[] output_buffer_host;
    delete[] output_proto_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;

    return 0;
}