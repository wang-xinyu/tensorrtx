#include <iostream>
#include <opencv2/opencv.hpp>
#include "backbone.hpp"
#include "RpnDecodePlugin.h"
#include "RpnNmsPlugin.h"
#include "RoiAlignPlugin.h"
#include "PredictorDecodePlugin.h"
#include "BatchedNmsPlugin.h"
#include "calibrator.hpp"

#define DEVICE 0
#define BATCH_SIZE 1
#define BACKBONE_RESNETTYPE R50
// data
static const std::vector<float> PIXEL_MEAN = { 103.53, 116.28, 123.675 };
static const std::vector<float> PIXEL_STD = {1.0, 1.0, 1.0};
static constexpr float MIN_SIZE = 800.0;
static constexpr float MAX_SIZE = 1333.0;
static constexpr int NUM_CLASSES = 80;
static constexpr int INPUT_H = 480;
static constexpr int INPUT_W = 640;
static int IMAGE_HEIGHT = 800;
static int IMAGE_WIDTH = 1333;
// rpn
static const std::vector<float> ANCHOR_SIZES = { 32, 64, 128, 256, 512 };
static const std::vector<float> ASPECT_RATIOS = { 0.5, 1.0, 2.0 };
static constexpr int PRE_NMS_TOP_K_TEST = 6000;
static constexpr float RPN_NMS_THRESH = 0.7;
static constexpr int POST_NMS_TOPK = 1000;
// roialign
static constexpr int STRIDES = 16;
static constexpr int SAMPLING_RATIO = 0;
static constexpr int POOLER_RESOLUTION = 14;
// roihead
static constexpr float NMS_THRESH_TEST = 0.5;
static constexpr int DETECTIONS_PER_IMAGE = 100;
static constexpr float SCORE_THRESH = 0.6;
static const std::vector<float> BBOX_REG_WEIGHTS = { 10.0, 10.0, 5.0, 5.0 };

static const char* INPUT_NODE_NAME = "images";
static const std::vector<std::string> OUTPUT_NAMES = { "scores", "boxes",
"labels" };

std::vector<float> GenerateAnchors(const std::vector<float>& anchor_sizes,
const std::vector<float>& aspect_ratios) {
    std::vector<float> res;
    for (auto as : anchor_sizes) {
        float area = as * as;
        for (auto ar : aspect_ratios) {
            float w = sqrt(area / ar);
            float h = ar * w;
            res.push_back(-w / 2.0);
            res.push_back(-h / 2.0);
            res.push_back(w / 2.0);
            res.push_back(h / 2.0);
        }
    }
    return res;
}

// transpose && resize && normalization && padding
ITensor* DataPreprocess(INetworkDefinition *network, ITensor& input) {
    // get h and w
    auto input_hw = input.getDimensions();
    int c = input_hw.d[2];
    int height = input_hw.d[0];
    int width = input_hw.d[1];

    // resize
    float ratio = MIN_SIZE / static_cast<float>(std::min(height, width));
    float newh = 0, neww = 0;
    if (height < width) {
        newh = MIN_SIZE;
        neww = ratio * width;
    } else {
        newh = ratio * height;
        neww = MIN_SIZE;
    }
    if (std::max(newh, neww) > MAX_SIZE) {
        ratio = MAX_SIZE / static_cast<float>(std::max(newh, neww));
        newh = newh * ratio;
        neww = neww * ratio;
    }
    height = static_cast<int>(newh + 0.5);
    width = static_cast<int>(neww + 0.5);
    auto resize_layer = network->addResize(input);
    assert(resize_layer);
    resize_layer->setResizeMode(ResizeMode::kLINEAR);
    resize_layer->setOutputDimensions(Dims3{ height, width, c });
    IMAGE_HEIGHT = height;
    IMAGE_WIDTH = width;

    // HWC->CHW
    auto channel_permute = network->addShuffle(*resize_layer->getOutput(0));
    assert(channel_permute);
    channel_permute->setFirstTranspose(Permutation{ 2, 0, 1 });

    // sub pixel mean
    auto pixel_mean = network->addConstant(Dims3{ 3, 1, 1 },
    Weights{ DataType::kFLOAT, PIXEL_MEAN.data(), 3 });
    assert(pixel_mean);
    auto sub = network->addElementWise(*channel_permute->getOutput(0),
    *pixel_mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(sub);
    auto pixel_std = network->addConstant(Dims3{ 3, 1, 1 }, Weights{DataType::kFLOAT, PIXEL_STD.data(), 3});
    assert(pixel_std);
    auto div = network->addElementWise(*sub->getOutput(0), *pixel_std->getOutput(0), ElementWiseOperation::kDIV);
    assert(div);

    return div->getOutput(0);
}

void calculateRatio() {
    float ratio = MIN_SIZE / static_cast<float>(std::min(INPUT_H, INPUT_W));
    float newh = 0, neww = 0;
    if (INPUT_H < INPUT_W) {
        newh = MIN_SIZE;
        neww = ratio * INPUT_W;
    } else {
        newh = ratio * INPUT_H;
        neww = MIN_SIZE;
    }
    if (std::max(newh, neww) > MAX_SIZE) {
        ratio = MAX_SIZE / static_cast<float>(std::max(newh, neww));
        newh = newh * ratio;
        neww = neww * ratio;
    }
    IMAGE_HEIGHT = static_cast<int>(newh + 0.5);
    IMAGE_WIDTH = static_cast<int>(neww + 0.5);
}

ITensor* RPN(INetworkDefinition *network,
std::map<std::string, Weights>& weightMap, ITensor& features,
    int out_channels = 256) {
    int num_anchors = ANCHOR_SIZES.size() * ASPECT_RATIOS.size();
    int box_dim = 4;

    // rpn head conv
    auto rpn_head_conv = network->addConvolutionNd(features, out_channels,
    DimsHW{ 3, 3 }, weightMap["proposal_generator.rpn_head.conv.weight"],
    weightMap["proposal_generator.rpn_head.conv.bias"]);
    assert(rpn_head_conv);
    rpn_head_conv->setStrideNd(DimsHW{ 1, 1 });
    rpn_head_conv->setPaddingNd(DimsHW{ 1, 1 });
    auto rpn_head_relu = network->addActivation(*rpn_head_conv->getOutput(0), ActivationType::kRELU);
    assert(rpn_head_relu);

    // objectness logits
    auto rpn_head_logits = network->addConvolutionNd(*rpn_head_relu->getOutput(0), num_anchors, DimsHW{ 1, 1 },
    weightMap["proposal_generator.rpn_head.objectness_logits.weight"],
    weightMap["proposal_generator.rpn_head.objectness_logits.bias"]);
    assert(rpn_head_logits);
    rpn_head_logits->setStrideNd(DimsHW{ 1, 1 });

    // anchor deltas
    auto rpn_head_deltas = network->addConvolutionNd(*rpn_head_relu->getOutput(0), num_anchors * box_dim,
    DimsHW{ 1, 1 },
    weightMap["proposal_generator.rpn_head.anchor_deltas.weight"],
    weightMap["proposal_generator.rpn_head.anchor_deltas.bias"]);
    assert(rpn_head_deltas);
    auto rpn_head_deltas_dim = rpn_head_deltas->getOutput(0)->getDimensions();
    rpn_head_deltas->setStrideNd(DimsHW{ 1, 1 });

    auto anchors = GenerateAnchors(ANCHOR_SIZES, ASPECT_RATIOS);
    auto rpnDecodePlugin = RpnDecodePlugin(PRE_NMS_TOP_K_TEST, anchors, STRIDES, IMAGE_HEIGHT, IMAGE_WIDTH);
    std::vector<ITensor*> faster_decode_inputs = { rpn_head_logits->getOutput(0), rpn_head_deltas->getOutput(0) };
    auto rpnDecodeLayer = network->addPluginV2(faster_decode_inputs.data(), faster_decode_inputs.size(),
    rpnDecodePlugin);

    std::vector<ITensor*> nms_input = { rpnDecodeLayer->getOutput(0), rpnDecodeLayer->getOutput(1) };

    // nms
    auto nmsPlugin = RpnNmsPlugin(RPN_NMS_THRESH, POST_NMS_TOPK);
    auto nmsLayer = network->addPluginV2(nms_input.data(), nms_input.size(), nmsPlugin);
    return nmsLayer->getOutput(0);
}

std::vector<ITensor*> ROIHeads(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,
ITensor& proposals, ITensor& features) {
    std::vector<ITensor*> roi_inputs = { &proposals, &features };
    auto roiAlignPlugin = RoiAlignPlugin(POOLER_RESOLUTION, 1 / static_cast<float>(STRIDES), SAMPLING_RATIO,
    POST_NMS_TOPK, features.getDimensions().d[0]);
    auto roiAlignLayer = network->addPluginV2(roi_inputs.data(), roi_inputs.size(), roiAlignPlugin);

    // res5
    auto box_features = MakeStage(network, weightMap, "roi_heads.res5", *roiAlignLayer->getOutput(0),
    num_blocks_per_stage.at(BACKBONE_RESNETTYPE)[3], 1024, 512, 2048, 2);
    auto box_features_mean = network->addReduce(*box_features, ReduceOperation::kAVG, 12, true);

    // score
    auto scores = network->addFullyConnected(*box_features_mean->getOutput(0), NUM_CLASSES + 1,
    weightMap["roi_heads.box_predictor.cls_score.weight"],
    weightMap["roi_heads.box_predictor.cls_score.bias"]);
    auto probs = network->addSoftMax(*scores->getOutput(0));

    auto probs_dim = probs->getOutput(0)->getDimensions();
    auto score_slice = network->addSlice(*probs->getOutput(0), Dims4{ 0, 0, 0, 0 },
    Dims4{ probs_dim.d[0], probs_dim.d[1] - 1, 1, 1 }, Dims4{ 1, 1, 1, 1 });

    auto proposal_deltas = network->addFullyConnected(*box_features_mean->getOutput(0), NUM_CLASSES * 4,
    weightMap["roi_heads.box_predictor.bbox_pred.weight"],
    weightMap["roi_heads.box_predictor.bbox_pred.bias"]);

    // decode
    std::vector<ITensor*> predictorDecodeInput = { score_slice->getOutput(0),
    proposal_deltas->getOutput(0), &proposals };
    auto predictorDecodePlugin = PredictorDecodePlugin(probs_dim.d[0], IMAGE_HEIGHT, IMAGE_WIDTH, BBOX_REG_WEIGHTS);
    auto predictorDecodeLayer = network->addPluginV2(predictorDecodeInput.data(), predictorDecodeInput.size(),
    predictorDecodePlugin);

    // nms
    std::vector<ITensor*> nmsInput = { predictorDecodeLayer->getOutput(0), predictorDecodeLayer->getOutput(1),
    predictorDecodeLayer->getOutput(2) };
    auto batchedNmsPlugin = BatchedNmsPlugin(NMS_THRESH_TEST, DETECTIONS_PER_IMAGE);
    auto batchedNmsLayer = network->addPluginV2(nmsInput.data(), nmsInput.size(), batchedNmsPlugin);
    std::vector<ITensor*> result = { batchedNmsLayer->getOutput(0), batchedNmsLayer->getOutput(1),
    batchedNmsLayer->getOutput(2) };
    return result;
}

ICudaEngine* createEngine_rcnn(unsigned int maxBatchSize,
    const std::string& wtsfile, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& modelType,
    const std::string& quantizationType, const std::string& calibImgListFile, const std::string& calibFile) {
    /*
    description: after fuse bn
    */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {INPUT_H, INPUT_W, 3} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_NODE_NAME, dt, Dims3{ INPUT_H, INPUT_W, 3 });
    assert(data);

    // preprocess
    data = DataPreprocess(network, *data);
    std::map<std::string, Weights> weightMap;
    loadWeights(wtsfile, weightMap);

    // backbone
    ITensor* features = BuildResNet(network, weightMap, *data, BACKBONE_RESNETTYPE, 64, 64, 256);

    auto proposals = RPN(network, weightMap, *features, 1024);
    auto results = ROIHeads(network, weightMap, *proposals, *features);

    // build output
    for (int i = 0; i < results.size(); i++) {
        network->markOutput(*results[i]);
        results[i]->setName(OUTPUT_NAMES[i].c_str());
    }

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 30);
    if (quantizationType == "fp32") {
    } else if (quantizationType == "fp16") {
        config->setFlag(BuilderFlag::kFP16);
    } else if (quantizationType == "int8") {
        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/",
        "int8calib.table", INPUT_NODE_NAME);
        config->setInt8Calibrator(calibrator);
    } else {
        throw("does not support model type");
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // destroy network
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        delete[] mem.second.values;
    }
    return engine;
}

void BuildRcnnModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string& wtsfile,
const std::string& modelType = "faster",
const std::string& quantizationType = "fp32",
const std::string& calibImgListFile = "./imglist.txt",
const std::string& calibFile = "./calib.table") {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    ICudaEngine* engine = createEngine_rcnn(maxBatchSize,
        wtsfile, builder, config, DataType::kFLOAT, modelType, quantizationType, calibImgListFile, calibFile);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

bool parse_args(int argc, char** argv, std::string& wtsFile, std::string& engineFile, std::string& imgDir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s") {
        wtsFile = std::string(argv[2]);
        engineFile = std::string(argv[3]);
    } else if (std::string(argv[1]) == "-d") {
        engineFile = std::string(argv[2]);
        imgDir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wtsFile = "";
    std::string engineFile = "";

    std::string imgDir;
    if (!parse_args(argc, argv, wtsFile, engineFile, imgDir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./rcnn -s [.wts] [.engine] // serialize model to plan file" << std::endl;
        std::cerr << "./rcnn -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    if (!wtsFile.empty()) {
        IHostMemory* modelStream{ nullptr };
        BuildRcnnModel(BATCH_SIZE, &modelStream, wtsFile, "faster", "fp32");
        assert(modelStream != nullptr);
        std::ofstream p(engineFile, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // calculate ratio
    calculateRatio();

    // deserialize the .engine and run inference
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engineFile << " error!" << std::endl;
        return -1;
    }

    std::string trtModelStream;
    size_t modelSize{ 0 };
    file.seekg(0, file.end);
    modelSize = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream.resize(modelSize);
    assert(!trtModelStream.empty());
    file.read(const_cast<char*>(trtModelStream.c_str()), modelSize);
    file.close();

    // build engine
    std::cout << "build engine" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.c_str(), modelSize);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    runtime->destroy();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // prepare input file
    std::vector<std::string> fileList;
    if (read_files_in_dir(imgDir.c_str(), fileList) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data
    std::vector<float> data(BATCH_SIZE * INPUT_H * INPUT_W * 3, 0);
    void *data_d, *scores_d, *boxes_d, *classes_d;
    CUDA_CHECK(cudaMalloc(&data_d, BATCH_SIZE * INPUT_H * INPUT_W * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scores_d, BATCH_SIZE * DETECTIONS_PER_IMAGE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&boxes_d, BATCH_SIZE * DETECTIONS_PER_IMAGE * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&classes_d, BATCH_SIZE * DETECTIONS_PER_IMAGE * sizeof(float)));

    std::unique_ptr<float[]> scores_h(new float[BATCH_SIZE * DETECTIONS_PER_IMAGE]);
    std::unique_ptr<float[]> boxes_h(new float[BATCH_SIZE * DETECTIONS_PER_IMAGE * 4]);
    std::unique_ptr<float[]> classes_h(new float[BATCH_SIZE * DETECTIONS_PER_IMAGE]);

    int fcount = 0;
    int fileLen = fileList.size();
    for (int f = 0; f < fileLen; f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != fileLen) continue;

        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(imgDir + "/" + fileList[f - fcount + 1 + b]);
            img = preprocessImg(img, INPUT_W, INPUT_H);
            if (img.empty()) continue;
            for (int i = 0; i < INPUT_H * INPUT_W * 3; i++)
                data[b*INPUT_H * INPUT_W * 3 + i] = static_cast<float>(*(img.data + i));
        }

        // Run inference
        auto start = std::chrono::system_clock::now();

        CUDA_CHECK(cudaMemcpyAsync(data_d, data.data(), BATCH_SIZE * INPUT_H * INPUT_W * 3 * sizeof(float),
        cudaMemcpyHostToDevice, stream));
        std::vector<void*> buffers = { data_d, scores_d, boxes_d, classes_d };

        context->enqueue(BATCH_SIZE, buffers.data(), stream, nullptr);

        CUDA_CHECK(cudaMemcpyAsync(scores_h.get(), scores_d, BATCH_SIZE * DETECTIONS_PER_IMAGE * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(boxes_h.get(), boxes_d, BATCH_SIZE * DETECTIONS_PER_IMAGE * 4 * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(classes_h.get(), classes_d, BATCH_SIZE * DETECTIONS_PER_IMAGE * sizeof(float),
        cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);

        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        float h_ratio = static_cast<float>(INPUT_H) / IMAGE_HEIGHT;
        float w_ratio = static_cast<float>(INPUT_W) / IMAGE_WIDTH;

        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(imgDir + "/" + fileList[f - fcount + 1 + b]);
            img = preprocessImg(img, INPUT_W, INPUT_H);
            for (int i = 0; i < DETECTIONS_PER_IMAGE; i++) {
                if (scores_h[b * DETECTIONS_PER_IMAGE + i] > SCORE_THRESH) {
                    float x1 = boxes_h[b * DETECTIONS_PER_IMAGE * 4 + i * 4 + 0] * w_ratio;
                    float y1 = boxes_h[b * DETECTIONS_PER_IMAGE * 4 + i * 4 + 1] * h_ratio;
                    float x2 = boxes_h[b * DETECTIONS_PER_IMAGE * 4 + i * 4 + 2] * w_ratio;
                    float y2 = boxes_h[b * DETECTIONS_PER_IMAGE * 4 + i * 4 + 3] * h_ratio;
                    int label = classes_h[b * DETECTIONS_PER_IMAGE + i];
                    float score = scores_h[b * DETECTIONS_PER_IMAGE + i];
                    printf("boxes:[%.6f, %.6f, %.6f, %.6f] scores: %.4f label: %d \n", x1, y1, x2, y2, score, label);
                    cv::Rect r(x1, y1, x2 - x1, y2 - y1);
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, std::to_string(label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
            }
            cv::imwrite("_" + fileList[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(scores_d));
    CUDA_CHECK(cudaFree(boxes_d));
    CUDA_CHECK(cudaFree(classes_d));
    context->destroy();
    engine->destroy();

    return 0;
}
