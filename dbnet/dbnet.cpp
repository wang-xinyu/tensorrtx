#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  
#define EXPANDRATIO 1.4
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int OUTPUT_SIZE = 640*640*2;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "out";
static Logger gLogger;

cv::RotatedRect expandBox(const cv::RotatedRect& inBox, float ratio = 1.0)
{
    cv::Size size = inBox.size;
    int neww = size.width * ratio;
    int newh = size.height *ratio;
    return cv::RotatedRect(inBox.center, cv::Size(neww, newh), inBox.angle);
}
// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

        // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
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
        // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
        // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
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
        // net weight  input,outch, ksize,  s,  g, std::string lname
        // 1
    auto p5 = convBnLeaky(network, weightMap, *relu9->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c5"); // k=1 s = 1  p = k/2=1/2=0
        
    auto c4_1 = convBnLeaky(network, weightMap, *relu7->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c4");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
         deval[i] = 1.0;
    }
    Weights deconvwts1{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* p4_1 = network->addDeconvolutionNd(*p5->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    p4_1->setStrideNd(DimsHW{ 2, 2 });
        p4_1->setNbGroups(64);
        weightMap["deconv1"] = deconvwts1;

        auto p4_add = network->addElementWise(*p4_1->getOutput(0), *c4_1->getOutput(0), ElementWiseOperation::kSUM);
        auto p4 = convBnLeaky(network, weightMap, *p4_add->getOutput(0), 64, 3, 1, 1, "neck.smooth_p4");  // smooth
        
        // 2
        auto c3_1 = convBnLeaky(network, weightMap, *relu5->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c3");

        Weights deconvwts2{ DataType::kFLOAT, deval, 64 * 2 * 2 };
        IDeconvolutionLayer* p3_1 = network->addDeconvolutionNd(*p4->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts2, emptywts);
        p3_1->setStrideNd(DimsHW{ 2, 2 });
        p3_1->setNbGroups(64);

        auto p3_add = network->addElementWise(*p3_1->getOutput(0), *c3_1->getOutput(0), ElementWiseOperation::kSUM);
        auto p3 = convBnLeaky(network, weightMap, *p3_add->getOutput(0), 64, 3, 1, 1, "neck.smooth_p3");  // smooth
        // 3
        auto c2_1 = convBnLeaky(network, weightMap, *relu3->getOutput(0), 64, 1, 1, 1, "neck.reduce_conv_c2");

        Weights deconvwts3{ DataType::kFLOAT, deval, 64 * 2 * 2 };
        IDeconvolutionLayer* p2_1 = network->addDeconvolutionNd(*p3->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts3, emptywts);
        p2_1->setStrideNd(DimsHW{ 2, 2 });
        p2_1->setNbGroups(64);
        //Dims p2_1dim = p2_1->getOutput(0)->getDimensions();
        auto p2_add = network->addElementWise(*p2_1->getOutput(0), *c2_1->getOutput(0), ElementWiseOperation::kSUM);
        auto p2 = convBnLeaky(network, weightMap, *p2_add->getOutput(0), 64, 3, 1, 1, "neck.smooth_p2");  // smooth



        // _upsample_cat
        // p3--p2 (upx2  w p s=2 0 2)
        Weights deconvwts4{ DataType::kFLOAT, deval, 64 * 2 * 2 };
        IDeconvolutionLayer* p3_up_p2 = network->addDeconvolutionNd(*p3->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts4, emptywts);
        p3_up_p2->setStrideNd(DimsHW{ 2, 2 });
        p3_up_p2->setNbGroups(64);

        // p4--p2(upx4 wps=824)
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

        // p5--p2(upx8) wps =808 
        Weights deconvwts6{ DataType::kFLOAT, deval2, 64 * 8 * 8 };
        IDeconvolutionLayer* p5_up_p2 = network->addDeconvolutionNd(*p5->getOutput(0), 64, DimsHW{ 8, 8 }, deconvwts6, emptywts);
        p5_up_p2->setStrideNd(DimsHW{ 8, 8 });
        p5_up_p2->setNbGroups(64);

        // torch.cat([p2, p3, p4, p5], dim=1)
        //Dims p2dim = p2->getOutput(0)->getDimensions();
        //Dims p3dim = p3_up_p2->getOutput(0)->getDimensions();
        //Dims p4dim = p4_up_p2->getOutput(0)->getDimensions();
        //Dims p5dim = p5_up_p2->getOutput(0)->getDimensions();

        ITensor* inputTensors[] = { p2->getOutput(0), p3_up_p2->getOutput(0), p4_up_p2->getOutput(0), p5_up_p2->getOutput(0) };
        auto neck_cat = network->addConcatenation(inputTensors, 4);
        //Dims neck_catdim = neck_cat->getOutput(0)->getDimensions();

        ILayer* neck_out = convBnLeaky2(network, weightMap, *neck_cat->getOutput(0), 256, 3, 1, 1, "neck.conv");  // smooth
        assert(neck_out);
        //Dims neck_outdim = neck_out->getOutput(0)->getDimensions();
        /* -------  head ------- */
        // shrink_maps = self.binarize(x)
        //                           net      weight      input,outch,         ksize, s, g, std::string lname
        auto binarize1 = convBnLeaky2(network, weightMap, *neck_out->getOutput(0), 64, 3, 1, 1, "head.binarize");  //  
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
        auto thresh1 = convBnLeaky2(network, weightMap, *neck_out->getOutput(0), 64, 3, 1, 1, "head.thresh", false);  //  
        
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

        //y = torch.cat((shrink_maps, threshold_maps), dim=1)
        // binarize4 thresh4
        //Dims binarize4dim = binarize4->getOutput(0)->getDimensions();
        //Dims thresh4dim = thresh4->getOutput(0)->getDimensions();

        ITensor* inputTensors2[] = { binarize4->getOutput(0), thresh4->getOutput(0)};
        auto head_out = network->addConcatenation(inputTensors2, 2);

        // y = F.interpolate(y, size=(H, W))  # 使用最近邻训练的可以用TRTAPI实现
        // 最后大小为图片大小
        head_out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        network->markOutput(*head_out->getOutput(0));

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
        for (auto& mem : weightMap)
        {
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

int main(int argc, char** argv)
{
        cudaSetDevice(DEVICE);
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{ nullptr };
        size_t size{ 0 };

        if (argc == 2 && std::string(argv[1]) == "-s")
        {
                IHostMemory* modelStream{ nullptr };
                APIToModel(BATCH_SIZE, &modelStream);
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
        else if (argc == 3 && std::string(argv[1]) == "-d")
        {
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
        else 
        {
                std::cerr << "arguments not right!" << std::endl;
                std::cerr << "./debnet -s  // serialize model to plan file" << std::endl;
                std::cerr << "./debnet -d ../samples  // deserialize plan file and run inference" << std::endl;
                return -1;
        }

        // prepare input data ---------------------------
        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
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
        /*
        std::vector<float> mean_value{0.406, 0.456, 0.485};
        std::vector<float> std_value{0.225, 0.224, 0.229};
        cv::Mat src, dst;
        std::vector<cv::Mat> bgrChannels(3);
        cv::split(src, bgrChannels);
        for (auto i = 0; i < bgrChannels.size(); i++)
         {
                bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
        }
        cv::meger(bgrChannels, dst);

        */
        std::vector<float> mean_value{ 0.406, 0.456, 0.485 };  // BGR
        std::vector<float> std_value{ 0.225, 0.224, 0.229 };
        int fcount = 0;
        for (int f = 0; f < (int)file_names.size(); f++) {
                fcount++;
                if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
                for (int b = 0; b < fcount; b++) {
                        //cv::Mat img = cv::imread(file_names[f - fcount + 1 + b]);
                        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
                        if (img.empty()) continue;
                        cv::Mat pr_img; // letterbox BGR to RGB
                        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
                        int i = 0;
                        for (int row = 0; row < INPUT_H; ++row) {
                                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                                for (int col = 0; col < INPUT_W; ++col) {
                                        data[b * 3 * INPUT_H * INPUT_W + i] = (uc_pixel[2]/255.0 - mean_value[2]) / std_value[2];
                                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (uc_pixel[1]/255.0 - mean_value[1]) / std_value[1];
                                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (uc_pixel[0]/255.0 - mean_value[0]) / std_value[0];
                                        uc_pixel += 3;
                                        ++i;
                                }
                        }
                }

                // Run inference
                auto start = std::chrono::system_clock::now();
                doInference(*context, data, prob, BATCH_SIZE);
                auto end = std::chrono::system_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

                // prob 为 2* 640*640  拿出第一个
                cv::Mat map = cv::Mat::zeros(cv::Size(640, 640), CV_8UC1);
                for (int b = 0; b < fcount; b++) 
                {
                        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
                        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
                        for (int h = 0; h < INPUT_H; ++h)
                        {
                                uchar *ptr = map.ptr(h);
                                for (int w = 0; w < INPUT_W; ++w)
                                {
                                        ptr[w] = (prob[b*OUTPUT_SIZE + h*INPUT_W + w] > 0.3) ? 255 : 0;
                                }
                        }
                        // 提取最小外接矩形
                        std::vector<std::vector<cv::Point>> contours;
                        std::vector<cv::Vec4i> hierarcy;
                        cv::findContours(map, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

                        std::vector<cv::Rect> boundRect(contours.size());
                        std::vector<cv::RotatedRect> box(contours.size());
                        cv::Point2f rect[4];
                        for (int i = 0; i < contours.size(); i++)
                        {
                                box[i] = cv::minAreaRect(cv::Mat(contours[i]));
                                //boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
                                //绘制外接矩形和    最小外接矩形（for循环）
                                //cv::rectangle(img, cv::Point(boundRect[i].x, boundRect[i].y), cv::Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), cv::Scalar(0, 255, 0), 2, 8);
                                cv::RotatedRect expandbox = expandBox(box[i], EXPANDRATIO);
                                expandbox.points(rect);//把最小外接矩形四个端点复制给rect数组
                                for (int j = 0; j < 4; j++)
                                {
                                          cv::line(img, rect[j], rect[(j + 1) % 4], cv::Scalar(0, 0, 255), 2, 8);
                                 }
                        }

                        cv::imshow("result", img);
                        cv::waitKey(0);
                }
                return 0;
        }
}