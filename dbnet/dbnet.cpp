#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <math.h>

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define EXPANDRATIO 1.4
static const int SHORT_INPUT = 640;
static const int MAX_INPUT_SIZE = 1440; // 32x
static const int MIN_INPUT_SIZE = 608;
static const int OPT_INPUT_W = 1152;
static const int OPT_INPUT_H = 640;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "out";
static Logger gLogger;

cv::RotatedRect expandBox(const cv::RotatedRect& inBox, float ratio = 1.0) {
    cv::Size size = inBox.size;
    int neww = size.width * ratio;
    int newh = size.height *ratio;
    return cv::RotatedRect(inBox.center, cv::Size(neww, newh), inBox.angle);
}
float paddimg(cv::Mat& In_Out_img, int shortsize = 960) {
    int w = In_Out_img.cols;
    int h = In_Out_img.rows;
    float scale = 1.f;
    if (w < h) {
        scale = (float)shortsize / w;
        h = scale * h;
        w = shortsize;
    }
    else {
        scale = (float)shortsize / h;
        w = scale * w;
        h = shortsize;
    }

    if (h % 32 != 0) {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0) {
        w = (w / 32 + 1) * 32;
    }

    cv::resize(In_Out_img, In_Out_img, cv::Size(w, h));
    return scale;
}
// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{ 1, 3, -1, -1 });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("E:\\LearningCodes\\DBNET\\DBNet.pytorch\\tools\\DBNet.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ Resnet18 backbone------ */
      // Add convolution layer with 6 outputs and a 5x5 filter.
    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{ 7, 7 }, weightMap["backbone.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 3, 3 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    assert(pool1);
    pool1->setStride(DimsHW{ 2, 2 });
    pool1->setPadding(DimsHW{ 1, 1 });

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "backbone.layer1.1."); // x2

    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "backbone.layer2.0.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "backbone.layer2.1."); // x3

    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "backbone.layer3.0.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "backbone.layer3.1."); //x4

    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "backbone.layer4.0.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "backbone.layer4.1."); //x5

    /* ------- FPN  neck ------- */
    ILayer* p5 = convBnLeaky(network, weightMap, *relu9->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c5.conv", ".bn"); // k=1 s = 1  p = k/2=1/2=0
    ILayer* c4_1 = convBnLeaky(network, weightMap, *relu7->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c4.conv", ".bn");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts1{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* p4_1 = network->addDeconvolutionNd(*p5->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    p4_1->setStrideNd(DimsHW{ 2, 2 });
    p4_1->setNbGroups(64);
    weightMap["deconv1"] = deconvwts1;

    IElementWiseLayer* p4_add = network->addElementWise(*p4_1->getOutput(0), *c4_1->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* p4 = convBnLeaky(network, weightMap, *p4_add->getOutput(0), 64, 3, 1, 1, "neck.smooth_p4.conv", ".bn");  // smooth
    ILayer* c3_1 = convBnLeaky(network, weightMap, *relu5->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c3.conv", ".bn");

    Weights deconvwts2{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* p3_1 = network->addDeconvolutionNd(*p4->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts2, emptywts);
    p3_1->setStrideNd(DimsHW{ 2, 2 });
    p3_1->setNbGroups(64);

    IElementWiseLayer* p3_add = network->addElementWise(*p3_1->getOutput(0), *c3_1->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* p3 = convBnLeaky(network, weightMap, *p3_add->getOutput(0), 64, 3, 1, 1, "neck.smooth_p3.conv", ".bn");  // smooth
    ILayer* c2_1 = convBnLeaky(network, weightMap, *relu3->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c2.conv", ".bn");

    Weights deconvwts3{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* p2_1 = network->addDeconvolutionNd(*p3->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts3, emptywts);
    p2_1->setStrideNd(DimsHW{ 2, 2 });
    p2_1->setNbGroups(64);

    IElementWiseLayer* p2_add = network->addElementWise(*p2_1->getOutput(0), *c2_1->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* p2 = convBnLeaky(network, weightMap, *p2_add->getOutput(0), 64, 3, 1, 1, "neck.smooth_p2.conv", ".bn");  // smooth

    Weights deconvwts4{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* p3_up_p2 = network->addDeconvolutionNd(*p3->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts4, emptywts);
    p3_up_p2->setStrideNd(DimsHW{ 2, 2 });
    p3_up_p2->setNbGroups(64);

    float *deval2 = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 8 * 8));
    for (int i = 0; i < 64 * 8 * 8; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts5{ DataType::kFLOAT, deval2, 64 * 8 * 8 };
    IDeconvolutionLayer* p4_up_p2 = network->addDeconvolutionNd(*p4->getOutput(0), 64, DimsHW{ 8, 8 }, deconvwts5, emptywts);
    p4_up_p2->setPadding(DimsHW{ 2, 2 });
    p4_up_p2->setStrideNd(DimsHW{ 4, 4 });
    p4_up_p2->setNbGroups(64);
    weightMap["deconv2"] = deconvwts5;

    Weights deconvwts6{ DataType::kFLOAT, deval2, 64 * 8 * 8 };
    IDeconvolutionLayer* p5_up_p2 = network->addDeconvolutionNd(*p5->getOutput(0), 64, DimsHW{ 8, 8 }, deconvwts6, emptywts);
    p5_up_p2->setStrideNd(DimsHW{ 8, 8 });
    p5_up_p2->setNbGroups(64);

    // torch.cat([p2, p3, p4, p5], dim=1)
    ITensor* inputTensors[] = { p2->getOutput(0), p3_up_p2->getOutput(0), p4_up_p2->getOutput(0), p5_up_p2->getOutput(0) };
    IConcatenationLayer* neck_cat = network->addConcatenation(inputTensors, 4);

    ILayer* neck_out = convBnLeaky(network, weightMap, *neck_cat->getOutput(0), 256, 3, 1, 1, "neck.conv.0", ".1");  // smooth
    assert(neck_out);
    ILayer* binarize1 = convBnLeaky(network, weightMap, *neck_out->getOutput(0), 64, 3, 1, 1, "head.binarize.0", ".1");  //  
    Weights deconvwts7{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* binarizeup = network->addDeconvolutionNd(*binarize1->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts7, emptywts);
    binarizeup->setStrideNd(DimsHW{ 2, 2 });
    binarizeup->setNbGroups(64);
    IScaleLayer* binarizebn1 = addBatchNorm2d(network, weightMap, *binarizeup->getOutput(0), "head.binarize.4", 1e-5);
    IActivationLayer* binarizerelu1 = network->addActivation(*binarizebn1->getOutput(0), ActivationType::kRELU);
    assert(binarizerelu1);

    Weights deconvwts8{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* binarizeup2 = network->addDeconvolutionNd(*binarizerelu1->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts8, emptywts);
    binarizeup2->setStrideNd(DimsHW{ 2, 2 });
    binarizeup2->setNbGroups(64);

    IConvolutionLayer* binarize3 = network->addConvolution(*binarizeup2->getOutput(0), 1, DimsHW{ 3, 3 }, weightMap["head.binarize.7.weight"], weightMap["head.binarize.7.bias"]);
    assert(binarize3);
    binarize3->setStride(DimsHW{ 1, 1 });
    binarize3->setPadding(DimsHW{ 1, 1 });
    IActivationLayer* binarize4 = network->addActivation(*binarize3->getOutput(0), ActivationType::kSIGMOID);
    assert(binarize4);

    //threshold_maps = self.thresh(x)
    ILayer* thresh1 = convBnLeaky(network, weightMap, *neck_out->getOutput(0), 64, 3, 1, 1, "head.thresh.0", ".1", false);  //  
    Weights deconvwts9{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* threshup = network->addDeconvolutionNd(*thresh1->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts9, emptywts);
    threshup->setStrideNd(DimsHW{ 2, 2 });
    threshup->setNbGroups(64);
    IConvolutionLayer* thresh2 = network->addConvolution(*threshup->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap["head.thresh.3.1.weight"], weightMap["head.thresh.3.1.bias"]);
    assert(thresh2);
    thresh2->setStride(DimsHW{ 1, 1 });
    thresh2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* threshbn1 = addBatchNorm2d(network, weightMap, *thresh2->getOutput(0), "head.thresh.4", 1e-5);
    IActivationLayer* threshrelu1 = network->addActivation(*threshbn1->getOutput(0), ActivationType::kRELU);
    assert(threshrelu1);

    Weights deconvwts10{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* threshup2 = network->addDeconvolutionNd(*threshrelu1->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts10, emptywts);
    threshup2->setStrideNd(DimsHW{ 2, 2 });
    threshup2->setNbGroups(64);
    IConvolutionLayer* thresh3 = network->addConvolution(*threshup2->getOutput(0), 1, DimsHW{ 3, 3 }, weightMap["head.thresh.6.1.weight"], weightMap["head.thresh.6.1.bias"]);
    assert(thresh3);
    thresh3->setStride(DimsHW{ 1, 1 });
    thresh3->setPadding(DimsHW{ 1, 1 });
    IActivationLayer* thresh4 = network->addActivation(*thresh3->getOutput(0), ActivationType::kSIGMOID);
    assert(thresh4);

    ITensor* inputTensors2[] = { binarize4->getOutput(0), thresh4->getOutput(0) };
    IConcatenationLayer* head_out = network->addConcatenation(inputTensors2, 2);

    // y = F.interpolate(y, size=(H, W)) 
    head_out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*head_out->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, OPT_INPUT_H, OPT_INPUT_W));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
    config->addOptimizationProfile(profile);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
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

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int h_scale, int w_scale) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    context.setBindingDimensions(inputIndex, Dims4(1, 3, h_scale, w_scale));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * h_scale * w_scale * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], 2 * h_scale * w_scale * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * h_scale * w_scale * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], h_scale * w_scale * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("DBNet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file("DBNet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./debnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./debnet -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    std::vector<float> mean_value{ 0.406, 0.456, 0.485 };  // BGR
    std::vector<float> std_value{ 0.225, 0.224, 0.229 };
    int fcount = 0;
    for (auto f : file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        cv::Mat pr_img = cv::imread(std::string(argv[2]) + "/" + f);
        cv::Mat src_img = pr_img.clone();
        if (pr_img.empty()) continue;
        float scale = paddimg(pr_img, SHORT_INPUT);
        std::cout << "letterbox shape: " << pr_img.cols << ", " << pr_img.rows << std::endl;
        if (pr_img.cols < MIN_INPUT_SIZE || pr_img.rows < MIN_INPUT_SIZE) continue;
        float* data = new float[3 * pr_img.rows * pr_img.cols];
        int i = 0;
        for (int row = 0; row < pr_img.rows; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < pr_img.cols; ++col) {
                data[i] = (uc_pixel[2] / 255.0 - mean_value[2]) / std_value[2];
                data[i + pr_img.rows * pr_img.cols] = (uc_pixel[1] / 255.0 - mean_value[1]) / std_value[1];
                data[i + 2 * pr_img.rows * pr_img.cols] = (uc_pixel[0] / 255.0 - mean_value[0]) / std_value[0];
                uc_pixel += 3;
                ++i;
            }
        }

        float* prob = new float[pr_img.rows *pr_img.cols * 2];
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, pr_img.rows, pr_img.cols);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // prob 为 2* 640*640  拿出第一个
        cv::Mat map = cv::Mat::zeros(cv::Size(pr_img.cols, pr_img.rows), CV_8UC1);
        for (int h = 0; h < pr_img.rows; ++h) {
            uchar *ptr = map.ptr(h);
            for (int w = 0; w < pr_img.cols; ++w) {
                ptr[w] = (prob[h * pr_img.cols + w] > 0.3) ? 255 : 0;
            }
        }
        // 提取最小外接矩形
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarcy;
        cv::findContours(map, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> boundRect(contours.size());
        std::vector<cv::RotatedRect> box(contours.size());
        cv::Point2f rect[4];
        for (int i = 0; i < contours.size(); i++) {
            box[i] = cv::minAreaRect(cv::Mat(contours[i]));
            //boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
            //绘制外接矩形和    最小外接矩形（for循环）
            //cv::rectangle(img, cv::Point(boundRect[i].x, boundRect[i].y), cv::Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), cv::Scalar(0, 255, 0), 2, 8);
            cv::RotatedRect expandbox = expandBox(box[i], EXPANDRATIO);
            expandbox.points(rect);//把最小外接矩形四个端点复制给rect数组
            for (int j = 0; j < 4; j++) {
                cv::Point2f p1, p2;
                p1.x = round(rect[j].x / pr_img.cols * src_img.cols);
                p1.y = round(rect[j].y / pr_img.rows * src_img.rows);
                p2.x = round(rect[(j + 1) % 4].x / pr_img.cols * src_img.cols);
                p2.y = round(rect[(j + 1) % 4].y / pr_img.rows * src_img.rows);
                cv::line(src_img, p1, p2, cv::Scalar(0, 0, 255), 2, 8);
            }
        }

        cv::imwrite("_" + f, src_img);
        //cv::waitKey(0);

        delete prob;
        delete data;
    }

    return 0;
}