#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4

// stuff we know about the network and the input/output blobs
static const int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static Logger gLogger;

ILayer* conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s = 1, float leaky = 0.1) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{3, 3}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(leaky);
    assert(lr);
    return lr;
}

ILayer* conv_bn_no_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s = 1) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{3, 3}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    return bn1;
}

ILayer* conv_bn1X1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s = 1, float leaky = 0.1) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{1, 1}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{0, 0});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(leaky);
    assert(lr);
    return lr;
}

ILayer* conv_dw(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int s = 1, float leaky = 0.1) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, inp, DimsHW{3, 3}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(inp);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    auto lr1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr1->setAlpha(leaky);
    assert(lr1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*lr1->getOutput(0), oup, DimsHW{1, 1}, getWeights(weightMap, lname + ".3.weight"), emptywts);
    assert(conv2);
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".4", 1e-5);
    auto lr2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    lr2->setAlpha(leaky);
    assert(lr2);
    return lr2;
}

IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup) {
    auto conv3x3 = conv_bn_no_relu(network, weightMap, input, lname + ".conv3X3", oup / 2);
    auto conv5x5_1 = conv_bn(network, weightMap, input, lname + ".conv5X5_1", oup / 4);
    auto conv5x5 = conv_bn_no_relu(network, weightMap, *conv5x5_1->getOutput(0), lname + ".conv5X5_2", oup / 4);
    auto conv7x7 = conv_bn(network, weightMap, *conv5x5_1->getOutput(0), lname + ".conv7X7_2", oup / 4);
    conv7x7 = conv_bn_no_relu(network, weightMap, *conv7x7->getOutput(0), lname + ".conv7x7_3", oup / 4);
    ITensor* inputTensors[] = {conv3x3->getOutput(0), conv5x5->getOutput(0), conv7x7->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 3);
    IActivationLayer* relu1 = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../retinaface.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone mobilenet0.25  ---------------
    // stage 1
    auto x = conv_bn(network, weightMap, *data, "body.stage1.0", 8, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.1", 8, 16);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.2", 16, 32, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.3", 32, 32);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.4", 32, 64, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.5", 64, 64);
    auto stage1 = x;

    // stage 2
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.0", 64, 128, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.1", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.2", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.3", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.4", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.5", 128, 128);
    auto stage2 = x;

    // stage 3
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage3.0", 128, 256, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage3.1", 256, 256);
    auto stage3 = x;

    //Dims d1 = stage1->getOutput(0)->getDimensions();
    //std::cout << d1.d[0] << " " << d1.d[1] << " " << d1.d[2] << std::endl;
    // ------------- FPN ---------------
    auto output1 = conv_bn1X1(network, weightMap, *stage1->getOutput(0), "fpn.output1", 64);
    auto output2 = conv_bn1X1(network, weightMap, *stage2->getOutput(0), "fpn.output2", 64);
    auto output3 = conv_bn1X1(network, weightMap, *stage3->getOutput(0), "fpn.output3", 64);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* up3 = network->addDeconvolutionNd(*output3->getOutput(0), 64, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStrideNd(DimsHW{2, 2});
    up3->setNbGroups(64);
    weightMap["up3"] = deconvwts;

    output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    output2 = conv_bn(network, weightMap, *output2->getOutput(0), "fpn.merge2", 64);

    IDeconvolutionLayer* up2 = network->addDeconvolutionNd(*output2->getOutput(0), 64, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up2);
    up2->setStrideNd(DimsHW{2, 2});
    up2->setNbGroups(64);
    output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    output1 = conv_bn(network, weightMap, *output1->getOutput(0), "fpn.merge1", 64);

    // ------------- SSH ---------------
    auto ssh1 = ssh(network, weightMap, *output1->getOutput(0), "ssh1", 64);
    auto ssh2 = ssh(network, weightMap, *output2->getOutput(0), "ssh2", 64);
    auto ssh3 = ssh(network, weightMap, *output3->getOutput(0), "ssh3", 64);

    //// ------------- Head ---------------
    auto bbox_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.0.conv1x1.weight"], weightMap["BboxHead.0.conv1x1.bias"]);
    auto bbox_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.1.conv1x1.weight"], weightMap["BboxHead.1.conv1x1.bias"]);
    auto bbox_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.2.conv1x1.weight"], weightMap["BboxHead.2.conv1x1.bias"]);

    auto cls_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.0.conv1x1.weight"], weightMap["ClassHead.0.conv1x1.bias"]);
    auto cls_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.1.conv1x1.weight"], weightMap["ClassHead.1.conv1x1.bias"]);
    auto cls_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.2.conv1x1.weight"], weightMap["ClassHead.2.conv1x1.bias"]);

    auto lmk_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.0.conv1x1.weight"], weightMap["LandmarkHead.0.conv1x1.bias"]);
    auto lmk_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.1.conv1x1.weight"], weightMap["LandmarkHead.1.conv1x1.bias"]);
    auto lmk_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.2.conv1x1.weight"], weightMap["LandmarkHead.2.conv1x1.bias"]);

    //// ------------- Decode bbox, conf, landmark ---------------
    ITensor* inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0), lmk_head1->getOutput(0)};
    auto cat1 = network->addConcatenation(inputTensors1, 3);
    ITensor* inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0), lmk_head2->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 3);
    ITensor* inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0), lmk_head3->getOutput(0)};
    auto cat3 = network->addConcatenation(inputTensors3, 3);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor* inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodelayer = network->addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodelayer);

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./widerface_calib/", "mnet_int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
        mem.second.values = NULL;
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./retina_mnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./retina_mnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("retina_mnet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("retina_mnet.engine", std::ios::binary);
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

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;

    cv::Mat img = cv::imread("worlds-largest-selfie.jpg");
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
    //cv::imwrite("preprocessed.jpg", pr_img);

    // For multi-batch, I feed the same image multiple times.
    // If you want to process different images in a batch, you need adapt it.
    for (int b = 0; b < BATCH_SIZE; b++) {
        float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
            p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
            p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
        }
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

    for (int b = 0; b < BATCH_SIZE; b++) {
        std::vector<decodeplugin::Detection> res;
        nms(res, &prob[b * OUTPUT_SIZE], IOU_THRESH);
        std::cout << "number of detections -> " << prob[b * OUTPUT_SIZE] << std::endl;
        std::cout << " -> " << prob[b * OUTPUT_SIZE + 10] << std::endl;
        std::cout << "after nms -> " << res.size() << std::endl;
        cv::Mat tmp = img.clone();
        for (size_t j = 0; j < res.size(); j++) {
            if (res[j].class_confidence < CONF_THRESH) continue;
            cv::Rect r = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
            cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
            for (int k = 0; k < 10; k += 2) {
                cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
            }
        }
        cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
