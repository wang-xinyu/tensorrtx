#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "logging.h"
#include "LayerNormPlugin.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

static Logger gLogger;

std::vector<std::string> load_imagenet_labels(const std::string& label_file = "imagenet_classes.txt") {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    if (!file.is_open()) {
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";

void inference(const std::string& engine_file, const std::string& image_file, const std::string& label_file = "imagenet_classes.txt") {
    std::cout << "Running inference..." << std::endl;
    
    // Register LayerNorm plugin
    static LayerNormPluginCreator pluginCreator;
    getPluginRegistry()->registerCreator(pluginCreator, "");
    
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Engine file not found: " << engine_file << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Determine dimensions from engine
    int inputIndex = -1;
    int outputIndex = -1;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i)) {
            inputIndex = i;
        } else {
            outputIndex = i;
        }
    }

    if (inputIndex == -1 || outputIndex == -1) {
        std::cerr << "Error: Could not find input or output bindings in engine." << std::endl;
        return;
    }

    Dims inputDims = engine->getBindingDimensions(inputIndex);
    Dims outputDims = engine->getBindingDimensions(outputIndex);

    // Assuming NCHW format for input
    int input_h = inputDims.d[2];
    int input_w = inputDims.d[3];
    int input_c = inputDims.d[1]; // Usually 3

    // Assuming N x NumClasses or just NumClasses
    int outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) {
        // Skip batch dimension if it is dynamic (-1) or 1
        if (i == 0 && (outputDims.d[i] == -1 || outputDims.d[i] == 1)) continue; 
        outputSize *= outputDims.d[i];
    }
    
    std::cout << "Input Dimensions: " << input_c << "x" << input_h << "x" << input_w << std::endl;
    std::cout << "Output Size: " << outputSize << std::endl;

    // Load image
    cv::Mat img = cv::imread(image_file);
    if (img.empty()) {
        std::cerr << "Error: Image not found: " << image_file << std::endl;
        return;
    }
    cv::resize(img, img, cv::Size(input_w, input_h));
    img.convertTo(img, CV_32F);
    
    // Normalize (Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225])
    // OpenCV is BGR. Pytorch expects RGB.
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img /= 255.0;
    
    float mean[] = {0.485, 0.456, 0.406};
    float std[] = {0.229, 0.224, 0.225};
    
    // HWC -> NCHW and Normalize
    float* hostData = new float[input_c * input_h * input_w];
    for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
            for (int c = 0; c < input_c; ++c) {
                float val = img.at<cv::Vec3f>(h, w)[c];
                hostData[c * input_h * input_w + h * input_w + w] = (val - mean[c]) / std[c];
            }
        }
    }

    void* deviceData;
    cudaMalloc(&deviceData, input_c * input_h * input_w * sizeof(float));
    cudaMemcpy(deviceData, hostData, input_c * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);

    // Output buffer
    float* hostOutput = new float[outputSize];
    void* deviceOutput;
    cudaMalloc(&deviceOutput, outputSize * sizeof(float));

    void* bindings[] = {deviceData, deviceOutput};
    if (engine->getBindingIndex(INPUT_BLOB_NAME) != 0) {
        bindings[inputIndex] = deviceData;
        bindings[outputIndex] = deviceOutput;
    }
    
    // Execute
    context->executeV2(bindings);

    // Copy back
    cudaMemcpy(hostOutput, deviceOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Argmax
    float maxVal = -1e9;
    int maxIdx = -1;
    for (int i = 0; i < outputSize; ++i) {
        if (hostOutput[i] > maxVal) {
            maxVal = hostOutput[i];
            maxIdx = i;
        }
    }
    
    auto labels = load_imagenet_labels(label_file);
    if (!labels.empty() && maxIdx < static_cast<int>(labels.size())) {
        std::cout << "Predicted Class: " << maxIdx << " - " << labels[maxIdx] << " (Score: " << maxVal << ")" << std::endl;
    } else {
        std::cout << "Predicted Class: " << maxIdx << " (Score: " << maxVal << ")" << std::endl;
    }

    cudaFree(deviceData);
    cudaFree(deviceOutput);
    delete[] hostData;
    delete[] hostOutput;
    delete context;
    delete engine;
    delete runtime;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path> [label_file]" << std::endl;
        std::cerr << "Example: " << argv[0] << " convnextv2.engine images/test.jpg" << std::endl;
        std::cerr << "         " << argv[0] << " convnextv2.engine images/test.jpg custom_labels.txt" << std::endl;
        return -1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    std::string label_file = (argc == 4) ? argv[3] : "imagenet_classes.txt";
    
    inference(engine_path, image_path, label_file);
    
    return 0;
}
