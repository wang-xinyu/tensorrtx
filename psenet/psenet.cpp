#include "psenet.h"

#define MAX_INPUT_SIZE 1200
#define MIN_INPUT_SIZE 128
#define OPT_INPUT_W 640
#define OPT_INPUT_H 640

PSENet::PSENet(int max_side_len, float threshold, int num_kernel, int stride) : max_side_len_(max_side_len),
                                                                                post_threshold_(threshold),
                                                                                num_kernels_(num_kernel),
                                                                                stride_(stride)
{
}

PSENet::~PSENet()
{
}

// create the engine using only the API and not any parser.
ICudaEngine *PSENet::createEngine(IBuilder *builder, IBuilderConfig *config)
{
    std::map<std::string, Weights> weightMap = loadWeights("./psenet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    ITensor *data = network->addInput(input_name_, dt, Dims4{-1, 3, -1, -1});
    assert(data);

    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["resnet_v1_50/conv1/weights"], emptywts);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});
    assert(conv1);

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "resnet_v1_50/conv1/BatchNorm/", 1e-5);
    assert(bn1);

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // C2
    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});
    assert(pool1);

    IActivationLayer *x;
    x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 1, "resnet_v1_50/block1/unit_1/bottleneck_v1/", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 64, 1, "resnet_v1_50/block1/unit_2/bottleneck_v1/", 0);
    // C3
    IActivationLayer *block1 = bottleneck(network, weightMap, *x->getOutput(0), 64, 2, "resnet_v1_50/block1/unit_3/bottleneck_v1/", 2);

    x = bottleneck(network, weightMap, *block1->getOutput(0), 128, 1, "resnet_v1_50/block2/unit_1/bottleneck_v1/", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 128, 1, "resnet_v1_50/block2/unit_2/bottleneck_v1/", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 128, 1, "resnet_v1_50/block2/unit_3/bottleneck_v1/", 0);
    // C4
    IActivationLayer *block2 = bottleneck(network, weightMap, *x->getOutput(0), 128, 2, "resnet_v1_50/block2/unit_4/bottleneck_v1/", 2);

    x = bottleneck(network, weightMap, *block2->getOutput(0), 256, 1, "resnet_v1_50/block3/unit_1/bottleneck_v1/", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "resnet_v1_50/block3/unit_2/bottleneck_v1/", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "resnet_v1_50/block3/unit_3/bottleneck_v1/", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "resnet_v1_50/block3/unit_4/bottleneck_v1/", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "resnet_v1_50/block3/unit_5/bottleneck_v1/", 0);
    IActivationLayer *block3 = bottleneck(network, weightMap, *x->getOutput(0), 256, 2, "resnet_v1_50/block3/unit_6/bottleneck_v1/", 2);

    x = bottleneck(network, weightMap, *block3->getOutput(0), 512, 1, "resnet_v1_50/block4/unit_1/bottleneck_v1/", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 1, "resnet_v1_50/block4/unit_2/bottleneck_v1/", 0);
    // C5
    IActivationLayer *block4 = bottleneck(network, weightMap, *x->getOutput(0), 512, 1, "resnet_v1_50/block4/unit_3/bottleneck_v1/", 0);

    IActivationLayer *build_p5_r1 = ConvRelu(network, weightMap, *block4->getOutput(0), 256, 1, 1, "build_feature_pyramid/build_P5/");
    assert(build_p5_r1);
    IActivationLayer *build_p4_r1 = ConvRelu(network, weightMap, *block2->getOutput(0), 256, 1, 1, "build_feature_pyramid/build_P4/reduce_dimension/");
    assert(build_p4_r1);

    IResizeLayer *bfp_layer4_resize = network->addResize(*build_p5_r1->getOutput(0));
    auto build_p4_r1_shape = network->addShape(*build_p4_r1->getOutput(0))->getOutput(0);
    bfp_layer4_resize->setInput(1, *build_p4_r1_shape);
    bfp_layer4_resize->setResizeMode(ResizeMode::kNEAREST);
    bfp_layer4_resize->setAlignCorners(false);
    assert(bfp_layer4_resize);

    IElementWiseLayer *bfp_add = network->addElementWise(*bfp_layer4_resize->getOutput(0), *build_p4_r1->getOutput(0), ElementWiseOperation::kSUM);
    assert(bfp_add);

    IActivationLayer *build_p4_r2 = ConvRelu(network, weightMap, *bfp_add->getOutput(0), 256, 3, 1, "build_feature_pyramid/build_P4/avoid_aliasing/");
    assert(build_p4_r2);

    IActivationLayer *build_p3_r1 = ConvRelu(network, weightMap, *block1->getOutput(0), 256, 1, 1, "build_feature_pyramid/build_P3/reduce_dimension/");
    assert(build_p3_r1);

    IResizeLayer *bfp_layer3_resize = network->addResize(*build_p4_r2->getOutput(0));
    bfp_layer3_resize->setResizeMode(ResizeMode::kNEAREST);
    auto build_p3_r1_shape = network->addShape(*build_p3_r1->getOutput(0))->getOutput(0);
    bfp_layer3_resize->setInput(1, *build_p3_r1_shape);
    bfp_layer3_resize->setAlignCorners(false);
    assert(bfp_layer3_resize);
    IElementWiseLayer *bfp_add1 = network->addElementWise(*bfp_layer3_resize->getOutput(0), *build_p3_r1->getOutput(0), ElementWiseOperation::kSUM);
    assert(bfp_add1);

    IActivationLayer *build_p3_r2 = ConvRelu(network, weightMap, *bfp_add1->getOutput(0), 256, 3, 1, "build_feature_pyramid/build_P3/avoid_aliasing/");
    assert(build_p3_r2);

    IActivationLayer *build_p2_r1 = ConvRelu(network, weightMap, *pool1->getOutput(0), 256, 1, 1, "build_feature_pyramid/build_P2/reduce_dimension/");
    assert(build_p2_r1);
    IResizeLayer *bfp_layer2_resize = network->addResize(*build_p3_r2->getOutput(0));
    bfp_layer2_resize->setResizeMode(ResizeMode::kNEAREST);
    auto build_p2_r1_shape = network->addShape(*build_p2_r1->getOutput(0))->getOutput(0);
    bfp_layer2_resize->setInput(1, *build_p2_r1_shape);
    bfp_layer2_resize->setAlignCorners(false);
    assert(bfp_layer2_resize);
    IElementWiseLayer *bfp_add2 = network->addElementWise(*bfp_layer2_resize->getOutput(0), *build_p2_r1->getOutput(0), ElementWiseOperation::kSUM);
    assert(bfp_add2);

    // P2
    IActivationLayer *build_p2_r2 = ConvRelu(network, weightMap, *bfp_add2->getOutput(0), 256, 3, 1, "build_feature_pyramid/build_P2/avoid_aliasing/");
    assert(build_p2_r2);
    auto build_p2_r2_shape = network->addShape(*build_p2_r2->getOutput(0))->getOutput(0);
    // P3 x2
    IResizeLayer *layer1_resize = network->addResize(*build_p3_r2->getOutput(0));
    layer1_resize->setResizeMode(ResizeMode::kLINEAR);
    layer1_resize->setInput(1, *build_p2_r2_shape);
    layer1_resize->setAlignCorners(true);
    assert(layer1_resize);

    // P4 x4
    IResizeLayer *layer2_resize = network->addResize(*build_p4_r2->getOutput(0));
    layer2_resize->setResizeMode(ResizeMode::kLINEAR);
    layer2_resize->setInput(1, *build_p2_r2_shape);
    layer2_resize->setAlignCorners(true);

    assert(layer2_resize);

    // P5 x8
    IResizeLayer *layer3_resize = network->addResize(*build_p5_r1->getOutput(0));
    layer3_resize->setResizeMode(ResizeMode::kLINEAR);
    layer3_resize->setInput(1, *build_p2_r2_shape);
    layer3_resize->setAlignCorners(true);
    assert(layer3_resize);

    // C(P5,P4,P3,P2)
    ITensor *inputTensors[] = {layer3_resize->getOutput(0), layer2_resize->getOutput(0), layer1_resize->getOutput(0), build_p2_r2->getOutput(0)};

    IConcatenationLayer *concat = network->addConcatenation(inputTensors, 4);
    assert(concat);

    IConvolutionLayer *feature_result_conv = network->addConvolutionNd(*concat->getOutput(0), 256, DimsHW{3, 3}, weightMap["feature_results/Conv/weights"], emptywts);
    feature_result_conv->setPaddingNd(DimsHW{1, 1});
    assert(feature_result_conv);

    IScaleLayer *feature_result_bn = addBatchNorm2d(network, weightMap, *feature_result_conv->getOutput(0), "feature_results/Conv/BatchNorm/", 1e-5);
    assert(feature_result_bn);

    IActivationLayer *feature_result_relu = network->addActivation(*feature_result_bn->getOutput(0), ActivationType::kRELU);
    assert(feature_result_relu);
    IConvolutionLayer *feature_result_conv_1 = network->addConvolutionNd(*feature_result_relu->getOutput(0), 6, DimsHW{1, 1}, weightMap["feature_results/Conv_1/weights"], weightMap["feature_results/Conv_1/biases"]);
    assert(feature_result_conv_1);

    IActivationLayer *sigmoid = network->addActivation(*feature_result_conv_1->getOutput(0), ActivationType::kSIGMOID);
    assert(sigmoid);

    sigmoid->getOutput(0)->setName(output_name_);
    std::cout << "Set name out" << std::endl;
    network->markOutput(*sigmoid->getOutput(0));

    // Set profile
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions(input_name_, OptProfileSelector::kMIN, Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
    profile->setDimensions(input_name_, OptProfileSelector::kOPT, Dims4(1, 3, OPT_INPUT_H, OPT_INPUT_W));
    profile->setDimensions(input_name_, OptProfileSelector::kMAX, Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
    config->addOptimizationProfile(profile);

    // Build engine
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    ;
    std::cout << "Build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}

void PSENet::serializeEngine()
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(builder, config);
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory *modelStream{nullptr};
    modelStream = engine->serialize();
    assert(modelStream != nullptr);

    std::ofstream p("./psenet.engine", std::ios::binary | std::ios::out);
    if (!p)
    {
        std::cerr << "Could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    return;
}

void PSENet::deserializeEngine()
{
    std::ifstream file("./psenet.engine", std::ios::binary | std::ios::in);
    if (file.good())
    {
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char *trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        mCudaEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(trtModelStream, size), InferDeleter());
        assert(mCudaEngine != nullptr);
    }
}

void PSENet::inferenceOnce(IExecutionContext &context, float *input, float *output, int input_h, int input_w)
{
    const ICudaEngine &engine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(input_name_);
    const int outputIndex = engine.getBindingIndex(output_name_);

    context.setBindingDimensions(inputIndex, Dims4(1, 3, input_h, input_w));

    int input_size = 3 * input_h * input_w * sizeof(float);
    int output_size = input_h * input_w * 6 / 16 * sizeof(float);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], input_size));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_size, cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void PSENet::init()
{
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger), InferDeleter());
    assert(mRuntime != nullptr);

    std::cout << "Deserialize Engine" << std::endl;
    deserializeEngine();

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext(), InferDeleter());
    assert(mContext != nullptr);

    mContext->setOptimizationProfile(0);

    std::cout << "Finished init" << std::endl;
}
void PSENet::detect(std::string image_path)
{

    int batch_size = 1;

    // Run inference

    cv::Mat image = cv::imread(image_path);
    int resize_h, resize_w;
    float ratio_h, ratio_w;

    auto start = std::chrono::system_clock::now();

    float *input = preProcess(image, resize_h, resize_w, ratio_h, ratio_w);
    float *output = new float[resize_h * resize_w * 6 / 16];

    inferenceOnce(*mContext, input, output, resize_h, resize_w);

    cv::Mat mask;
    postProcess(output, mask, resize_h, resize_w);

    drawRects(image, mask, ratio_h, ratio_w, stride_, 1.4);
    auto end = std::chrono::system_clock::now();

    cv::imwrite("result_" + image_path, image);

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

float *PSENet::preProcess(cv::Mat image, int &resize_h, int &resize_w, float &ratio_h, float &ratio_w)
{
    cv::Mat imageRGB;
    cv::cvtColor(image, imageRGB, CV_BGR2RGB);
    cv::Mat imageProcessed;
    int h = imageRGB.size().height;
    int w = imageRGB.size().width;
    resize_w = w;
    resize_h = h;

    float ratio = 1.0;
    // limit the max side
    if (resize_h > max_side_len_ && resize_w > max_side_len_)
    {
        if (resize_h > resize_w)
        {
            ratio = float(max_side_len_) / float(resize_h);
        }
        else
        {
            ratio = float(max_side_len_) / float(resize_w);
        }
    }
    resize_h = int(resize_h * ratio);
    resize_w = int(resize_w * ratio);

    if (resize_h % 32 != 0)
    {
        resize_h = (resize_h / 32 + 1) * 32;
    }
    if (resize_w % 32 != 0)
    {
        resize_w = (resize_w / 32 + 1) * 32;
    }
    ratio_h = resize_h / float(h);
    ratio_w = resize_w / float(w);

    cv::resize(imageRGB, imageProcessed, cv::Size(resize_w, resize_h));
    float *input = new float[3 * resize_h * resize_w];
    cv::Mat imgFloat;
    imageProcessed.convertTo(imgFloat, CV_32FC3);
    cv::subtract(imgFloat, cv::Scalar(123.68, 116.78, 103.94), imgFloat, cv::noArray(), -1);
    std::vector<cv::Mat> chw;
    for (auto i = 0; i < 3; ++i)
    {
        chw.emplace_back(cv::Mat(cv::Size(resize_w, resize_h), CV_32FC1, input + i * resize_w * resize_h));
    }
    cv::split(imgFloat, chw);
    return input;
}

void PSENet::postProcess(float *origin_output, cv::Mat &label_image, int resize_h, int resize_w)
{
    // BxCxHxW  S0 ===> S5  small ===> large
    const int height = (resize_h + stride_ - 1) / stride_;
    const int width = (resize_w + stride_ - 1) / stride_;
    const int length = height * width;

    std::vector<cv::Mat> kernels(num_kernels_);
    cv::Mat max_kernel(height, width, CV_32F, (void *)(origin_output + (num_kernels_ - 1) * length), 0);
    cv::threshold(max_kernel, max_kernel, post_threshold_, 255, cv::THRESH_BINARY);
    max_kernel.convertTo(max_kernel, CV_8U);
    assert(max_kernel.rows == height && max_kernel.cols == width);
    for (auto i = 0; i < num_kernels_ - 1; ++i)
    {
        cv::Mat kernel = cv::Mat(height, width, CV_32F, (void *)(origin_output + i * length), 0);
        cv::threshold(kernel, kernel, post_threshold_, 255, cv::THRESH_BINARY);
        kernel.convertTo(kernel, CV_8U);
        cv::bitwise_and(kernel, max_kernel, kernel);
        assert(kernel.rows == height && kernel.cols == width);
        kernels[i] = kernel;
    }
    kernels[num_kernels_ - 1] = max_kernel;

    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(kernels[0], label_image, stats, centroids, 4);
    label_image.convertTo(label_image, CV_8U);
    assert(label_image.rows == max_kernel.rows && label_image.cols == max_kernel.cols);

    std::map<int, std::vector<cv::Point>> contourMaps;

    // PSE algorithm
    std::queue<std::tuple<int, int, int>> q;
    std::queue<std::tuple<int, int, int>> q_next;
    for (auto h = 0; h < height; ++h)
    {
        for (auto w = 0; w < width; ++w)
        {
            auto label = *label_image.ptr(h, w);
            if (label > 0)
            {
                q.emplace(std::make_tuple(w, h, label));
                contourMaps[label].emplace_back(cv::Point(w, h));
            }
        }
    }
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    for (auto idx = 1; idx < num_kernels_; ++idx)
    {
        auto *ptr_kernel = kernels[idx].data;
        while (!q.empty())
        {
            auto q_n = q.front();
            q.pop();
            int x = std::get<0>(q_n);
            int y = std::get<1>(q_n);
            int l = std::get<2>(q_n);
            bool is_edge = true;
            for (auto j = 0; j < 4; ++j)
            {
                int tmpx = x + dx[j];
                int tmpy = y + dy[j];
                int offset = tmpy * width + tmpx;
                if (tmpx < 0 || tmpx >= width || tmpy < 0 || tmpy >= height)
                {
                    continue;
                }
                if (!(int)ptr_kernel[offset] || (int)*label_image.ptr(tmpy, tmpx) > 0)
                {
                    continue;
                }
                q.emplace(std::make_tuple(tmpx, tmpy, l));
                *label_image.ptr(tmpy, tmpx) = l;
                contourMaps[l].emplace_back(cv::Point(tmpx, tmpy));
                is_edge = false;
            }
            if (is_edge)
            {
                q_next.emplace(std::make_tuple(x, y, l));
            }
        }
        std::swap(q, q_next);
    }
}
