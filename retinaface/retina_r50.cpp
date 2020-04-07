#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
//#include "plugin_factory.h"
//#include "yololayer.h"
#include <opencv2/opencv.hpp>

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 1  // GPU id

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 384;  // 
static const int INPUT_W = 640;
static const int OUTPUT_SIZE = 256 * 24 * 40;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
static Logger gLogger;

typedef struct {
    float bbox[4];
    float det_confidence;
    float class_id;
    float class_confidence;
} Detection;

cv::Mat preprocess_img(cv::Mat& img, int input_dim) {
    int w, h, x, y;
    if (img.cols > img.rows) {
        w = input_dim;
        h = input_dim * img.rows / img.cols;
        x = 0;
        y = (input_dim - h) / 2;
    } else {
        w = input_dim * img.cols / img.rows;
        h = input_dim;
        x = (input_dim - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(input_dim, input_dim, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, int input_dim, float bbox[4]) {
    int l, r, t, b;
    if (img.cols > img.rows) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (input_dim - input_dim * img.rows / img.cols) / 2;
        b = bbox[1] + bbox[3]/2.f - (input_dim - input_dim * img.rows / img.cols) / 2;
        l = l * img.cols / input_dim;
        r = r * img.cols / input_dim;
        t = t * img.cols / input_dim;
        b = b * img.cols / input_dim;
    } else {
        l = bbox[0] - bbox[2]/2.f - (input_dim - input_dim * img.cols / img.rows) / 2;
        r = bbox[0] + bbox[2]/2.f - (input_dim - input_dim * img.cols / img.rows) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l * img.rows / input_dim;
        r = r * img.rows / input_dim;
        t = t * img.rows / input_dim;
        b = b * img.rows / input_dim;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Detection>& res, float *output, float nms_thresh = 0.4) {
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < OUTPUT_SIZE / 7; i++) {
        if (output[7 * i + 4] <= 0.5) continue;
        Detection det;
        memcpy(&det, &output[7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

// Load weights from files
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
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

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{kernelsize, kernelsize}, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{stride, stride});
    conv1->setPadding(DimsHW{padding, padding});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

    if (!userelu) return bn1;

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt) {
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../retinaface.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone resnet50 ---------------
    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "body.bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});
    pool1->setPadding(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "body.layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.3.");
    IActivationLayer* layer2 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "body.layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.5.");
    IActivationLayer* layer3 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "body.layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.2.");
    IActivationLayer* layer4 = x;

    // ------------- FPN ---------------
    auto output1 = conv_bn_relu(network, weightMap, *layer2->getOutput(0), 256, 1, 1, 0, true, "fpn.output1");
    auto output2 = conv_bn_relu(network, weightMap, *layer3->getOutput(0), 256, 1, 1, 0, true, "fpn.output2");
    auto output3 = conv_bn_relu(network, weightMap, *layer4->getOutput(0), 256, 1, 1, 0, true, "fpn.output3");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* up3 = network->addDeconvolution(*output3->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStride(DimsHW{2, 2});
    up3->setNbGroups(256);
    weightMap["up3"] = deconvwts;

    IElementWiseLayer* ew1 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);

    Dims dims = ew1->getOutput(0)->getDimensions();
    std::cout << ew1->getOutput(0)->getName() << " dims";
    for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << "-" << (int)dims.type[i] << "   ";
    }
    std::cout << std::endl;

    //output2 = conv_bn_relu(network, weightMap, *ew1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge2");

    //IDeconvolutionLayer* up2 = network->addDeconvolution(*output2->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    //assert(up2);
    //up2->setStride(DimsHW{2, 2});
    //up2->setNbGroups(256);
    //output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    //output1 = conv_bn_relu(network, weightMap, *output1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge1");

    ew1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*ew1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
    builder->setFp16Mode(true);
#endif
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "build out" << std::endl;

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

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
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
    std::cout << "beginning" << std::endl;
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./retina_r50 -s   // serialize model to plan file" << std::endl;
        std::cerr << "./retina_r50 -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("retina_r50.engine");
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("retina_r50.engine", std::ios::binary);
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
    float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    //cv::Mat img = cv::imread("../dog.jpg");
    //cv::Mat pr_img = preprocess_img(img, INPUT_H);
    //cv::imwrite("123.jpg", pr_img);
    //for (int i = 0; i < INPUT_H * INPUT_W; i++) {
    //    data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
    //    data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
    //    data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    //}

    //PluginFactory pf;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, &pf);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        //std::vector<Detection> res;
        //nms(res, prob);
        //for (size_t j = 0; j < res.size(); j++) {
        //    float *p = (float*)&res[j];
        //    for (size_t k = 0; k < 7; k++) {
        //        std::cout << p[k] << ", ";
        //    }
        //    std::cout << std::endl;
        //    cv::Rect r = get_rect(img, INPUT_W, res[j].bbox);
        //    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        //}
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        //cv::imwrite("res.jpg", img);
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
