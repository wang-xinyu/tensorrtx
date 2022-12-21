#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"

#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE1 = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const int OUTPUT_SIZE2 = 32 * (INPUT_H / 4) * (INPUT_W / 4);
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // Backbone
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");

    // Head
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // Segmentation
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (32 + Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (32 + Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (32 + Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2}, true);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    auto proto = Proto(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 32, "model.24.proto");
    proto->getOutput(0)->setName("proto");
    network->markOutput(*proto->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
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

cv::Rect get_downscale_rect(float bbox[4], float scale) {
  float left = bbox[0] - bbox[2] / 2;
  float top = bbox[1] - bbox[3] / 2;
  float right = bbox[0] + bbox[2] / 2;
  float bottom = bbox[1] + bbox[3] / 2;
  left /= scale;
  top /= scale;
  right /= scale;
  bottom /= scale;
  return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
}

std::vector<cv::Mat> process_mask(const float* proto, std::vector<Yolo::Detection>& dets) {
  std::vector<cv::Mat> masks;
  for (size_t i = 0; i < dets.size(); i++) {
    cv::Mat mask_mat = cv::Mat::zeros(INPUT_H / 4, INPUT_W / 4, CV_32FC1);
    auto r = get_downscale_rect(dets[i].bbox, 4);
    for (int x = r.x; x < r.x + r.width; x++) {
      for (int y = r.y; y < r.y + r.height; y++) {
        float e = 0.0f;
        for (int j = 0; j < 32; j++) {
          e += dets[i].mask[j] * proto[j * OUTPUT_SIZE2 / 32 + y * mask_mat.cols + x];
        }
        e = 1.0f / (1.0f + expf(-e));
        mask_mat.at<float>(y, x) = e;
        // if (e > 0.5) {
        //   // TODO(Call for PR): Use different colors for different class ids
        //   mask_mat.at<cv::Vec3b>(y, x)[2] = 0xFF;
        //   mask_mat.at<cv::Vec3b>(y, x)[1] = 0x38;
        //   mask_mat.at<cv::Vec3b>(y, x)[0] = 0x38;
        // }
      }
    }
    cv::resize(mask_mat, mask_mat, cv::Size(INPUT_W, INPUT_H));
    masks.push_back(mask_mat);
  }
  return masks;
}

cv::Mat scale_mask(cv::Mat mask, cv::Mat img) {
  int x, y, w, h;
  float r_w = INPUT_W / (img.cols * 1.0);
  float r_h = INPUT_H / (img.rows * 1.0);
  if (r_h > r_w) {
    w = INPUT_W;
    h = r_w * img.rows;
    x = 0;
    y = (INPUT_H - h) / 2;
  } else {
    w = r_h * img.cols;
    h = INPUT_H;
    x = (INPUT_W - w) / 2;
    y = 0;
  }
  cv::Rect r(x, y, w, h);
  cv::Mat res;
  cv::resize(mask(r), res, img.size());
  return res;
}

void draw_mask_bbox(cv::Mat& img, std::vector<Yolo::Detection>& dets, std::vector<cv::Mat>& masks) {
  static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A,
                                         0x92CC17, 0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF,
                                         0x344593, 0x6473FF, 0x0018EC, 0x8438FF, 0x520085, 0xCB38FF,
                                         0xFF95C8, 0xFF37C7};
  for (size_t i = 0; i < dets.size(); i++) {
    cv::Mat img_mask = scale_mask(masks[i], img);
    auto color = colors[(int)dets[i].class_id % colors.size()];
    auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);

    cv::Rect r = get_rect(img, dets[i].bbox);
    for (int x = r.x; x < r.x + r.width; x++) {
      for (int y = r.y; y < r.y + r.height; y++) {
        float val = img_mask.at<float>(y, x);
        if (val <= 0.5) continue;
        img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2 + bgr[0] / 2;
        img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2 + bgr[1] / 2;
        img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2 + bgr[2] / 2;
      }
    }

    cv::rectangle(img, r, bgr, 2);
    // TODO(Call for PR): convert class id to class name
    cv::putText(img, std::to_string((int)dets[i].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);
  }
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5_seg -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5_seg -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, gd, gw, wts_name);
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
    char *trtModelStream = nullptr;
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

    static float prob[BATCH_SIZE * OUTPUT_SIZE1];
    static float proto[BATCH_SIZE * OUTPUT_SIZE2];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 3);
    float* buffers[3];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    const int outputIndex2 = engine->getBindingIndex("proto");
    assert(inputIndex == 0);
    assert(outputIndex1 == 1);
    assert(outputIndex2 == 2);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex1], BATCH_SIZE * OUTPUT_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex2], BATCH_SIZE * OUTPUT_SIZE2 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
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
        float *buffer_idx = (float*)buffers[inputIndex];
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            imgs_buffer[b] = img;
            size_t size_image = img.cols * img.rows * 3;
            size_t size_image_dst = INPUT_H * INPUT_W * 3;
            //copy data to pinned memory
            memcpy(img_host, img.data, size_image);
            //copy data to device memory
            CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
            buffer_idx += size_image_dst;
            cudaStreamSynchronize(stream);
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, proto, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE1], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            cv::Mat img = imgs_buffer[b];

            auto masks = process_mask(&proto[b * OUTPUT_SIZE2], res);
            draw_mask_bbox(img, res, masks);
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex1]));
    CUDA_CHECK(cudaFree(buffers[outputIndex2]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}

