#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 448;
static const int INPUT_W = 448;
static const int S = 7;
static const int B = 2;
static const int C = 20;
static const int OUTPUT_SIZE = S * S * (B * 5 + C);  // 7*7*30 = 1470

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

std::vector<std::string> class_names = {"person", "bird",         "cat",          "cow",     "dog",
                                        "horse",  "sheep",        "aeroplane",    "bicycle", "boat",
                                        "bus",    "car",          "motorbike",    "train",   "bottle",
                                        "chair",  "dining table", "potted plant", "sofa",    "tvmonitor"};

using namespace nvinfer1;

static Logger gLogger;

/**
 * @brief Load model weights from a file into a map.
 * 
 * This function reads a weights file where each line contains a weight blob's
 * name and values in hexadecimal format. It stores each weight blob in a 
 * `Weights` structure and returns a map from the blob name to its `Weights`.
 * 
 * @param file The path to the weights file to be loaded.
 * @return std::map<std::string, Weights> A map where keys are weight names and 
 *         values are corresponding `Weights` structures containing the data, 
 *         data type, and number of elements.
 */
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

/**
 * @brief Add a 2D Batch Normalization layer to a TensorRT network.
 * 
 * This function converts PyTorch-style batch normalization parameters 
 * (weight, bias, running mean, running variance) into TensorRT scale, shift, 
 * and power weights, and then adds an IScaleLayer to the network.
 * 
 * The scale, shift, and power weights are also stored in the weightMap to 
 * prevent memory leaks, as they are allocated dynamically.
 * 
 * @param network Pointer to the TensorRT network definition to which the batch 
 *                normalization layer will be added.
 * @param weightMap Map of weight blobs loaded from a pretrained model, used to 
 *                  retrieve gamma, beta, running mean, and running variance.
 * @param midName The base name of the layer (used to index the weights in weightMap).
 * @param input The input ITensor to the batch normalization layer.
 * @param eps Small epsilon value to avoid division by zero in variance normalization.
 *            Default is 1e-5.
 * @return IScaleLayer* Pointer to the created IScaleLayer in the TensorRT network.
 */
IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                            const std::string& midName, ITensor& input, float eps = 1e-5) {
    float* gamma = (float*)weightMap["conv_layers." + midName + ".weight"].values;
    float* beta = (float*)weightMap["conv_layers." + midName + ".bias"].values;
    float* mean = (float*)weightMap["conv_layers." + midName + ".running_mean"].values;
    float* var = (float*)weightMap["conv_layers." + midName + ".running_var"].values;

    int len = weightMap["conv_layers." + midName + ".running_var"].count;

    // scale = gamma / sqrt(var + eps)
    float* scaleVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        scaleVal[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scaleVal, len};

    // shift = beta - mean * scale
    float* shiftVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        shiftVal[i] = beta[i] - mean[i] * scaleVal[i];
    }
    Weights shift{DataType::kFLOAT, shiftVal, len};

    // power = 1.0
    float* powerVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        powerVal[i] = 1.0f;
    }
    Weights power{DataType::kFLOAT, powerVal, len};

    // save as weight Map, otherwise cause memory leak.
    weightMap["conv_layers." + midName + ".scale"] = scale;
    weightMap["conv_layers." + midName + ".shift"] = shift;
    weightMap["conv_layers." + midName + ".power"] = power;

    IScaleLayer* bn = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn);

    return bn;
}

/**
 * @brief Add a Convolution -> BatchNorm -> Leaky ReLU sequence to a TensorRT network.
 * 
 * This function adds a convolution layer, followed by a batch normalization layer
 * and a Leaky ReLU activation layer to the specified TensorRT network.
 * It uses weights from a preloaded weightMap and constructs the layers in sequence.
 * 
 * @param network Pointer to the TensorRT network definition to which the layers will be added.
 * @param weightMap Map of weight blobs loaded from a pretrained model, used to retrieve convolution and batch norm weights.
 * @param convMidName The base name of the convolution layer (used to index weights in weightMap).
 * @param bnMidName The base name of the batch normalization layer (used to index weights in weightMap).
 * @param input The input ITensor to the convolution layer.
 * @param out_ch Number of output channels for the convolution layer.
 * @param ksize Kernel size for the convolution layer (default: 3).
 * @param stride Stride for the convolution layer (default: 1).
 * @param padding Padding for the convolution layer (default: 1).
 * @param groups Number of groups for grouped convolution (default: 1).
 * @return IActivationLayer* Pointer to the Leaky ReLU activation layer added to the network.
 */
IActivationLayer* convBnRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                             const std::string& convMidName, const std::string& bnMidName, ITensor& input, int out_ch,
                             int ksize = 3, int stride = 1, int padding = 1, int groups = 1) {
    // 1. conv
    std::string convWeightName = "conv_layers." + convMidName + ".weight";
    std::string convBiasName = "conv_layers." + convMidName + ".bias";
    IConvolutionLayer* conv = network->addConvolutionNd(input, out_ch, DimsHW{ksize, ksize}, weightMap[convWeightName],
                                                        weightMap[convBiasName]);
    assert(conv);

    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});

    // 2. BatchNorm
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, bnMidName, *conv->getOutput(0), 1e-5f);
    assert(bn);

    // 3. relu
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    relu->setAlpha(0.1);
    assert(relu);

    return relu;
}

/**
 * @brief Create a TensorRT engine for YOLOv1.
 * 
 * This function constructs a complete YOLOv1 network in TensorRT using 
 * pre-trained weights. It builds convolution, batch normalization, pooling,
 * fully connected, and activation layers, then returns a built ICudaEngine.
 * 
 * The function uses the provided builder and builder config to create the 
 * engine, and dynamically loads the weights from a .wts file.
 * 
 * @param maxBatchSize The maximum batch size the engine should support.
 * @param builder Pointer to the TensorRT IBuilder used to create the network.
 * @param config Pointer to the TensorRT IBuilderConfig for engine configuration.
 * @param dt The data type for network inputs (e.g., DataType::kFLOAT).
 * @return ICudaEngine* Pointer to the created TensorRT engine.
 */

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    // 1. Create Network
    INetworkDefinition* network =
            builder->createNetworkV2(1U << (unsigned int)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    assert(network);

    // 2. Input
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);
    std::cout << "Input shape = (1,3," << INPUT_H << "," << INPUT_W << ")" << std::endl;

    // 3. Load weights
    std::map<std::string, Weights> weightMap = loadWeights("../models/yolov1.wts");

    std::cout << "Loaded " << weightMap.size() << " weights." << std::endl;

    // stage1
    IActivationLayer* stage1 = convBnRelu(network, weightMap, "0", "1", *data, 192,
                                          7,  // ksize
                                          2,  // stride
                                          1   // padding
    );

    IPoolingLayer* pool1 = network->addPoolingNd(*stage1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    // stage2
    IActivationLayer* stage2 = convBnRelu(network, weightMap, "4", "5", *pool1->getOutput(0), 256);
    IPoolingLayer* pool2 = network->addPoolingNd(*stage2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool2->setStrideNd(DimsHW{2, 2});
    assert(pool2);

    IActivationLayer* stage3 = convBnRelu(network, weightMap, "8", "9", *pool2->getOutput(0), 512);
    IPoolingLayer* pool3 = network->addPoolingNd(*stage3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool3->setStrideNd(DimsHW{2, 2});
    assert(pool3);

    IActivationLayer* stage4 = convBnRelu(network, weightMap, "12", "13", *pool3->getOutput(0), 1024);
    IPoolingLayer* pool4 = network->addPoolingNd(*stage4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool4->setStrideNd(DimsHW{2, 2});
    assert(pool4);

    IActivationLayer* stage5 = convBnRelu(network, weightMap, "16", "17", *pool4->getOutput(0), 1024, 3, 2);

    IActivationLayer* stage6 = convBnRelu(network, weightMap, "19", "20", *stage5->getOutput(0), 1024);

    // flatten stage6
    IShuffleLayer* stage6FlattenLayer = network->addShuffle(*stage6->getOutput(0));
    stage6FlattenLayer->setReshapeDimensions(Dims2(1, 50176));

    // reshape fc1 weight shape.
    Dims fc1WeightDims = Dims2{4096, 50176};
    Weights fc1_w = weightMap["fc_layers.0.weight"];
    IConstantLayer* fc1WeightLayer = network->addConstant(fc1WeightDims, fc1_w);
    assert(fc1WeightLayer);

    // matrix multiply
    IMatrixMultiplyLayer* fc1MatrixMultiplyLayer =
            network->addMatrixMultiply(*stage6FlattenLayer->getOutput(0), MatrixOperation::kNONE,
                                       *fc1WeightLayer->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(fc1WeightLayer);

    // add fc1 bias
    Dims fc1BiasDims = Dims2{1, 4096};
    Weights fc1Bias = weightMap["fc_layers.0.bias"];
    IConstantLayer* fc1BiasLayer = network->addConstant(fc1BiasDims, fc1Bias);
    assert(fc1BiasLayer);

    IElementWiseLayer* fc1OutLayer = network->addElementWise(*fc1MatrixMultiplyLayer->getOutput(0),
                                                             *fc1BiasLayer->getOutput(0), ElementWiseOperation::kSUM);
    assert(fc1OutLayer);

    // relu after fc1
    IActivationLayer* fc1Relu = network->addActivation(*fc1OutLayer->getOutput(0), ActivationType::kLEAKY_RELU);
    fc1Relu->setAlpha(0.1);
    assert(fc1Relu);

    // reshape fc2 weight
    Dims fc2WeightDims = Dims2{1470, 4096};
    Weights fc2_w = weightMap["fc_layers.3.weight"];
    IConstantLayer* fc2WeightLayer = network->addConstant(fc2WeightDims, fc2_w);
    assert(fc2WeightLayer);

    // fc2 matrix multiply
    IMatrixMultiplyLayer* fc2MatrixMultiplyLayer = network->addMatrixMultiply(
            *fc1Relu->getOutput(0), MatrixOperation::kNONE, *fc2WeightLayer->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(fc2WeightLayer);

    // add fc2 bias
    Dims fc2BiasDims = Dims2{1, 1470};
    Weights fc2Bias = weightMap["fc_layers.3.bias"];
    IConstantLayer* fc2BiasLayer = network->addConstant(fc2BiasDims, fc2Bias);
    assert(fc2BiasLayer);

    IElementWiseLayer* fc2OutLayer = network->addElementWise(*fc2MatrixMultiplyLayer->getOutput(0),
                                                             *fc2BiasLayer->getOutput(0), ElementWiseOperation::kSUM);
    assert(fc2OutLayer);

    auto fc2Sig = network->addActivation(*fc2OutLayer->getOutput(0), ActivationType::kSIGMOID);

    // reshape to YOLOv1 output (7*7*30)
    IShuffleLayer* out = network->addShuffle(*fc2Sig->getOutput(0));

    // correct Dims3(CHW)
    out->setReshapeDimensions(Dims3(30, 7, 7));

    out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*out->getOutput(0));

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    std::cout << "Engine build completed!" << std::endl;

    return engine;
}

/**
 * @brief Build and serialize a TensorRT engine to a memory stream.
 * 
 * This function creates a TensorRT builder and configuration, constructs the 
 * network engine by calling `createEngine`, and then serializes the engine 
 * into a memory stream. The serialized engine can be later deserialized to 
 * create an ICudaEngine for inference.
 * 
 * @param maxBatchSize The maximum batch size that the engine should support.
 * @param modelStream Pointer to an IHostMemory* that will receive the serialized engine.
 */
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();
}

/**
 * @brief Preprocess an image for TensorRT inference.
 * 
 * This function converts an input OpenCV BGR image to the format required 
 * by TensorRT: RGB order, resized to the network input size, normalized 
 * to [0,1], and rearranged from HWC (height, width, channel) to CHW 
 * (channel, height, width) format.
 * 
 * @param img The input image in OpenCV BGR format.
 * @param data Pointer to a pre-allocated float array where the preprocessed 
 *             image data will be stored in CHW order.
 * @param input_h The target input height for the network.
 * @param input_w The target input width for the network.
 */
void preprocess_trt_cpp(const cv::Mat& img, float* data, int input_h, int input_w) {
    // 1. BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // 2. Resize to input size
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(input_w, input_h));

    // 3. Convert to float32
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 4. HWC -> CHW
    int channels = 3;
    int img_size = input_h * input_w;
    std::vector<cv::Mat> split_channels;
    cv::split(resized, split_channels);  // R, G, B channels

    // 5. RR.. GG.. BB..
    for (int c = 0; c < channels; ++c) {
        memcpy(data + c * img_size, split_channels[c].data, img_size * sizeof(float));
    }
}

/**
 * @brief Perform inference using a TensorRT execution context.
 * 
 * This function executes a forward pass of the network for a given batch of
 * input data. It allocates device memory for input and output tensors, copies
 * the input data to the GPU, performs asynchronous inference, and copies the
 * results back to host memory.
 * 
 * @param context The TensorRT execution context used to run inference.
 * @param input Pointer to the preprocessed input data in CHW float format.
 * @param output Pointer to pre-allocated memory where inference results will be stored.
 * @param pool3FlattenLayerOutput (Unused in current implementation, reserved for intermediate outputs if needed.)
 * @param batchSize Number of images in the batch to process.
 */
void doInference(IExecutionContext& context, float* input, float* output, float* pool3FlattenLayerOutput,
                 int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    std::cout << " engine.getNbIOTensors is :" << engine.getNbIOTensors() << std::endl;

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const char* inputName = INPUT_BLOB_NAME;
    const char* outputName = OUTPUT_BLOB_NAME;

    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};

    // Create GPU buffers on device
    CHECK(cudaMalloc(&deviceInput, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(deviceInput, input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice,
                          stream));

    context.setTensorAddress(inputName, deviceInput);
    context.setTensorAddress(outputName, deviceOutput);

    context.enqueueV3(stream);

    CHECK(cudaMemcpyAsync(output, deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
}

/**
 * @brief Compute the Intersection over Union (IoU) of two bounding boxes in [x, y, w, h] format.
 * 
 * The boxes are given in the format where (x, y) represents the center of the box,
 * and w, h represent the width and height. The function converts them to corner 
 * coordinates, computes the intersection area, and returns the IoU.
 * 
 * @param a Pointer to the first bounding box array [x, y, w, h].
 * @param b Pointer to the second bounding box array [x, y, w, h].
 * @return float The IoU value in the range [0, 1].
 */
float iou_xywh(const float* a, const float* b) {
    // a: [x,y,w,h], b: [x,y,w,h]
    float ax1 = a[0] - a[2] / 2.0f;
    float ay1 = a[1] - a[3] / 2.0f;
    float ax2 = a[0] + a[2] / 2.0f;
    float ay2 = a[1] + a[3] / 2.0f;

    float bx1 = b[0] - b[2] / 2.0f;
    float by1 = b[1] - b[3] / 2.0f;
    float bx2 = b[0] + b[2] / 2.0f;
    float by2 = b[1] + b[3] / 2.0f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float areaA = (ax2 - ax1) * (ay2 - ay1);
    float areaB = (bx2 - bx1) * (by2 - by1);

    return inter_area / (areaA + areaB - inter_area + 1e-6f);
}

/**
 * @brief Structure representing a single detection result.
 */
struct Detection {
    float x, y, w, h;
    float score;
    int cls;
};

/**
 * @brief Perform Non-Maximum Suppression (NMS) on bounding boxes.
 * 
 * This function applies class-specific NMS to filter overlapping bounding boxes.
 * It first computes class-specific confidence scores (conf * class probability),
 * then sorts detections per class by score, and suppresses boxes with high IoU.
 * 
 * @param bboxes Vector of bounding boxes, each in the format [x, y, w, h, conf, class_probs...].
 * @param num_classes Number of classes in the model.
 * @param conf_thresh Confidence threshold to filter low-confidence detections (default: 0.1).
 * @param iou_thresh IoU threshold for NMS to remove overlapping boxes (default: 0.3).
 * @return std::vector<Detection> Vector of final detection results after NMS.
 */
std::vector<Detection> nms_cpp(const std::vector<std::vector<float>>& bboxes, int num_classes, float conf_thresh = 0.1f,
                               float iou_thresh = 0.3f) {
    int N = bboxes.size();
    if (N == 0)
        return {};

    // class-specific confidence = conf * prob
    std::vector<std::vector<float>> cls_scores(N, std::vector<float>(num_classes));

    for (int i = 0; i < N; i++) {
        float conf = bboxes[i][4];
        for (int c = 0; c < num_classes; c++) {
            float score = conf * bboxes[i][5 + c];
            cls_scores[i][c] = (score > conf_thresh ? score : 0.0f);
        }
    }

    std::vector<bool> keep(N, false);
    for (int i = 0; i < N; i++)
        for (int c = 0; c < num_classes; c++)
            if (cls_scores[i][c] > 0)
                keep[i] = true;

    // NMS for every class
    for (int c = 0; c < num_classes; c++) {
        // collect index for each class.
        std::vector<int> idx;
        for (int i = 0; i < N; i++)
            if (cls_scores[i][c] > 0)
                idx.push_back(i);

        // order by scores.
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return cls_scores[a][c] > cls_scores[b][c]; });

        // NMS
        for (int i = 0; i < (int)idx.size(); i++) {
            if (cls_scores[idx[i]][c] == 0)
                continue;

            for (int j = i + 1; j < (int)idx.size(); j++) {
                if (cls_scores[idx[j]][c] == 0)
                    continue;

                float iou = iou_xywh(bboxes[idx[i]].data(), bboxes[idx[j]].data());

                if (iou > iou_thresh)
                    cls_scores[idx[j]][c] = 0;
            }
        }
    }

    // gene finally result.
    std::vector<Detection> out;
    for (int i = 0; i < N; i++) {
        float best_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes; c++) {
            if (cls_scores[i][c] > best_score) {
                best_score = cls_scores[i][c];
                best_class = c;
            }
        }

        if (best_class >= 0) {
            Detection det;
            det.x = bboxes[i][0];
            det.y = bboxes[i][1];
            det.w = bboxes[i][2];
            det.h = bboxes[i][3];
            det.score = best_score;
            det.cls = best_class;
            out.push_back(det);
        }
    }

    return out;
}

/**
 * @brief Convert YOLOv1 network predictions to bounding boxes and apply NMS.
 * 
 * This function takes the raw network output (predictions) in YOLOv1 format
 * and converts it into bounding boxes in [x, y, w, h, conf, class_probs...] format.
 * It selects the bounding box with higher confidence for each grid cell, 
 * then applies class-specific Non-Maximum Suppression (NMS) to produce final detections.
 * 
 * @param pred Pointer to the raw prediction array from the network. 
 *             Shape = S*S*30 (e.g., 7*7*30 = 1470 for YOLOv1).
 * @param S The grid size of the network output (e.g., 7 for YOLOv1).
 * @param B Number of predicted bounding boxes per grid cell (usually 2).
 * @param num_classes Number of object classes.
 * @param conf_thresh Confidence threshold to filter low-confidence boxes before NMS.
 * @param iou_thresh IoU threshold for NMS to remove overlapping boxes.
 * @return std::vector<Detection> Vector of final detection results after NMS.
 */
std::vector<Detection> pred2xywhcc_cpp(const float* pred,  // shape = S*S*30 = 1470
                                       int S, int B, int num_classes, float conf_thresh, float iou_thresh) {
    // reshape pred to pred[s][s][30]

    std::vector<std::vector<float>> bboxes(S * S, std::vector<float>(5 + num_classes));

    for (int x = 0; x < S; x++) {
        for (int y = 0; y < S; y++) {
            int idx = (x * S + y) * 30;

            // B1
            float b1_x = pred[idx + 0];
            float b1_y = pred[idx + 1];
            float b1_w = pred[idx + 2];
            float b1_h = pred[idx + 3];
            float b1_conf = pred[idx + 4];

            // B2
            float b2_x = pred[idx + 5];
            float b2_y = pred[idx + 6];
            float b2_w = pred[idx + 7];
            float b2_h = pred[idx + 8];
            float b2_conf = pred[idx + 9];

            int target = x * S + y;

            if (b1_conf > b2_conf) {
                bboxes[target][0] = b1_x;
                bboxes[target][1] = b1_y;
                bboxes[target][2] = b1_w;
                bboxes[target][3] = b1_h;
                bboxes[target][4] = b1_conf;
            } else {
                bboxes[target][0] = b2_x;
                bboxes[target][1] = b2_y;
                bboxes[target][2] = b2_w;
                bboxes[target][3] = b2_h;
                bboxes[target][4] = b2_conf;
            }

            for (int c = 0; c < num_classes; c++)
                bboxes[target][5 + c] = pred[idx + 10 + c];
        }
    }

    // NMS
    return nms_cpp(bboxes, num_classes, conf_thresh, iou_thresh);
}

/**
 * @brief Draw bounding boxes and class labels on an image.
 * 
 * This function takes a list of detection results and draws the corresponding 
 * bounding boxes and class labels onto the input OpenCV image. Each class is 
 * assigned a unique random color for visualization.
 * 
 * @param img The OpenCV image on which to draw bounding boxes (BGR format).
 * @param dets Vector of Detection objects representing the detected objects.
 * @param class_names Vector of class names corresponding to class IDs in Detection.
 */
void draw_bbox(cv::Mat& img, const std::vector<Detection>& dets, const std::vector<std::string>& class_names) {
    int h = img.rows;
    int w = img.cols;

    for (size_t i = 0; i < dets.size(); i++) {
        const Detection& d = dets[i];

        int x1 = static_cast<int>((d.x - d.w / 2.0f) * w);
        int y1 = static_cast<int>((d.y - d.h / 2.0f) * h);
        int x2 = static_cast<int>((d.x + d.w / 2.0f) * w);
        int y2 = static_cast<int>((d.y + d.h / 2.0f) * h);

        x1 = std::max(0, std::min(x1, w - 1));
        y1 = std::max(0, std::min(y1, h - 1));
        x2 = std::max(0, std::min(x2, w - 1));
        y2 = std::max(0, std::min(y2, h - 1));

        std::vector<std::array<int, 3>> COLORS(100);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (int i = 0; i < 100; ++i) {
            COLORS[i][0] = dis(gen);  // R
            COLORS[i][1] = dis(gen);  // G
            COLORS[i][2] = dis(gen);  // B
        }
        cv::Scalar color(COLORS[d.cls][2], COLORS[d.cls][1], COLORS[d.cls][0]);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string cls_name = class_names[d.cls];

        cv::putText(img, cls_name, cv::Point(x1, y1 - 3), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }
}

/**
 * @brief Main entry point for YOLOv1 TensorRT demo.
 * 
 * This program can either serialize a YOLOv1 model to a TensorRT engine
 * file or deserialize an existing engine and run inference on a test image.
 * The detection results are drawn on the image and saved to disk.
 * 
 * Usage:
 *  - ./regnet -s : serialize the model to a plan file (../models/yolov1.engine)
 *  - ./regnet -d : deserialize the plan file and run inference
 * 
 * @param argc Number of command line arguments.
 * @param argv Array of command line argument strings.
 * @return int Returns 1 if serialization succeeds, -1 for errors, 0 on successful inference.
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./regnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./regnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("../models/yolov1.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("../models/yolov1.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    float data[3 * INPUT_H * INPUT_W];

    // read the image.
    cv::Mat img = cv::imread("../test.jpg");
    if (img.empty()) {
        std::cerr << "Failed to read image!" << std::endl;
        return -1;
    }

    preprocess_trt_cpp(img, data, INPUT_H, INPUT_W);

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    float prob[OUTPUT_SIZE];
    float pool3FlattenLayerOutput[9216];
    doInference(*context, data, prob, pool3FlattenLayerOutput, 1);

    auto dets = pred2xywhcc_cpp(prob, 7, 2, 20, 0.1f, 0.3f);

    if (!dets.empty()) {
        draw_bbox(img, dets, class_names);

        std::string out_path = "../output.jpg";
        cv::imwrite(out_path, img);
        std::cout << "Saved to " << out_path << std::endl;
    }

    return 0;
}
