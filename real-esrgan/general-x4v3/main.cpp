#include <NvInfer.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "config/config.hpp"
#include "cuda_utils.h"
#include "logging/logging.h"
#include "pixel_shuffle/pixel_shuffle.hpp"
#include "preprocess/preprocess.hpp"

static Logger gLogger;

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

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

auto* ConvPRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int conv_nb,
                int index) {

    IConvolutionLayer* conv = network->addConvolutionNd(input, conv_nb, DimsHW{3, 3},
                                                        weightMap["body." + std::to_string(index) + ".weight"],
                                                        weightMap["body." + std::to_string(index) + ".bias"]);
    assert(conv);
    conv->setName(("body." + std::to_string(index) + ".weight").c_str());
    conv->setStrideNd(DimsHW{1, 1});
    conv->setPaddingNd(DimsHW{1, 1});
    auto conv_res = conv->getOutput(0);

    // add prelu layer
    // slope 64 number

    //auto slope = network->addConstant( {64}, weightMap["body." + std::to_string(index + 1) + ".weight"] );
    auto slope = network->addConstant(Dims4{1, 64, 1, 1}, weightMap["body." + std::to_string(index + 1) + ".weight"]);
    assert(slope);
    slope->setName(("body." + std::to_string(index + 1) + ".weight").c_str());

    auto prelu = network->addParametricReLU(*conv_res, *slope->getOutput(0));
    assert(prelu);

    return prelu;
}

void build_engine(DataType dt, std::string& wts_path) {

    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);

    auto data = network->addInput(INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W});

    // first
    auto layer = ConvPRelu(network, weightMap, *data, 64, 0);

    for (int i = 0; i < 32; ++i) {
        layer = ConvPRelu(network, weightMap, *layer->getOutput(0), 64, 2 * i + 2);
    }

    auto conv_last = network->addConvolutionNd(*layer->getOutput(0), 48, DimsHW{3, 3}, weightMap["body.66.weight"],
                                               weightMap["body.66.bias"]);
    assert(conv_last);
    conv_last->setName("body.66.weight");
    conv_last->setStrideNd(DimsHW{1, 1});
    conv_last->setPaddingNd(DimsHW{1, 1});
    auto conv_last_res = conv_last->getOutput(0);

    // add pixel shuffle layer by plugin
    IPluginCreator* creator = getPluginRegistry()->getPluginCreator("PixelShufflePlugin", "1");
    const PluginFieldCollection* pluginFC = creator->getFieldNames();
    std::vector<PluginField> pluginData;
    int upscaleFactor = 4;
    pluginData.emplace_back(PluginField{"upscaleFactor", &upscaleFactor, PluginFieldType::kINT32, 1});
    PluginFieldCollection pluginFCWithData = {static_cast<int>(pluginData.size()), pluginData.data()};
    auto pluginObj = creator->createPlugin("PixelShuffle", &pluginFCWithData);

    auto pixelShuffleLayer = network->addPluginV2(&conv_last_res, 1, *pluginObj);

    // the input "data" interpolate 4x and add to pixelShuffleLayer->getOutput(0)

    auto interpolateLayer = network->addResize(*data);
    interpolateLayer->setResizeMode(ResizeMode::kNEAREST);
    // Define scale factors
    float scales[] = {1.0f, 1.0f, 1.0 * OUT_SCALE, 1.0 * OUT_SCALE};  // scale_factor=4 for height and width
    interpolateLayer->setScales(scales, OUT_SCALE);

    // Add the two tensor as output
    auto addLayer = network->addElementWise(*interpolateLayer->getOutput(0), *pixelShuffleLayer->getOutput(0),
                                            ElementWiseOperation::kSUM);

    // output
    addLayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*addLayer->getOutput(0));

    // fp16
    if (USE_FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    std::ofstream ofs("../weights/real-esrgan.engine", std::ios::binary);

    assert(serialized_model != nullptr);
    ofs.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());

    delete config;
    delete serialized_model;
    delete builder;
}

static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output) {
    context.setBindingDimensions(0, Dims4(BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W));
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int main(int argc, char** argv) {
    std::string img_dir;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_dir>" << std::endl;
        return -1;
    } else {
        img_dir = argv[1];
    }

    std::string wts_path = "../weights/real-esrgan.wts";
    build_engine(DataType::kFLOAT, wts_path);

    std::string engine_name = "../weights/real-esrgan.engine";
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
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
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    std::vector<float> data;
    std::vector<float> output;
    //std::vector<float> res;

    //data.resize(BATCH_SIZE * 3 * INPUT_H * INPUT_W);
    data.resize(BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W);
    output.resize(BATCH_SIZE * OUTPUT_SIZE);

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    for (int index = 0; index < file_names.size(); ++index) {

        auto img = cv::imread(img_dir + "/" + file_names[index]);
        auto begin = std::chrono::high_resolution_clock::now();

        // BATCH_SIZE = 1
        for (int b = 0; b < BATCH_SIZE; b++) {
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = img.data + row * img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    //    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
                    // BGR2RGB and normalization
                    data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], data.data(),
                                   BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice,
                                   stream));
        doInference(*context, stream, (void**)buffers, output.data());
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                  << " ms" << std::endl;

        int OUTPUT_C = 3;
        int OUTPUT_H = INPUT_H * OUT_SCALE;
        int OUTPUT_W = INPUT_W * OUT_SCALE;

        for (int b = 0; b < BATCH_SIZE; b++) {
            cv::Mat img_res(OUTPUT_H, OUTPUT_W, CV_8UC3);
            int i = 0;
            for (int row = 0; row < OUTPUT_H; ++row) {
                uchar* uc_pixel = img_res.data + row * img_res.step;
                for (int col = 0; col < OUTPUT_W; ++col) {
                    // RGB2BGR and de_normalization
                    auto r2 = std::round(output[b * OUTPUT_C * OUTPUT_H * OUTPUT_W + i] * 255.0);
                    if (r2 < 0)
                        r2 = 0;
                    if (r2 > 255)
                        r2 = 255;
                    auto g2 = std::round(output[b * OUTPUT_C * OUTPUT_H * OUTPUT_W + i + 1 * OUTPUT_H * OUTPUT_W] *
                                         255.0);
                    if (g2 < 0)
                        g2 = 0;
                    if (g2 > 255)
                        g2 = 255;
                    auto b2 = std::round(output[b * OUTPUT_C * OUTPUT_H * OUTPUT_W + i + 2 * OUTPUT_H * OUTPUT_W] *
                                         255.0);
                    if (b2 < 0)
                        b2 = 0;
                    if (b2 > 255)
                        b2 = 255;

                    uc_pixel[0] = static_cast<uchar>(b2);  // B
                    uc_pixel[1] = static_cast<uchar>(g2);  // G
                    uc_pixel[2] = static_cast<uchar>(r2);  // R

                    // uc_pixel[0] = static_cast<uchar>(std::round(output[b * OUTPUT_C * OUTPUT_H * OUTPUT_W + i + 2 * OUTPUT_H * OUTPUT_W] * 255.0)); // B
                    // uc_pixel[1] = static_cast<uchar>(std::round(output[b * OUTPUT_C * OUTPUT_H * OUTPUT_W + i + 1 * OUTPUT_H * OUTPUT_W] * 255.0)); // G
                    // uc_pixel[2] = static_cast<uchar>(std::round(output[b * OUTPUT_C * OUTPUT_H * OUTPUT_W + i] * 255.0)); // R
                    uc_pixel += 3;
                    ++i;
                }
            }
            cv::imwrite("_" + file_names[index] + ".jpg", img_res);
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}
