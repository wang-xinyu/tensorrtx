#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
static const int INPUT_C = 3;
static const int INPUT_H = 288;
static const int INPUT_W = 800;
static const int OUTPUT_C = 101;
static const int OUTPUT_H = 56;
static const int OUTPUT_W = 4;
static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder,  DataType dt) {
    INetworkDefinition* network = builder->createNetwork();
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_C, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../lane.wts");
#if 0
    /* print layer names */
    for(std::map<std::string, Weights>::iterator iter = weightMap.begin(); iter != weightMap.end() ; iter++)
    {
        std::cout << iter->first << std::endl;
    }
#endif
    auto conv1 = network->addConvolution(*data, 64, DimsHW{ 7, 7 }, weightMap["model.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});
    conv1->setNbGroups(1);

    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "model.bn1", 1e-5);
    auto relu0 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool0 = network->addPooling(*relu0->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    pool0->setStride( DimsHW{ 2, 2 } );
    pool0->setPadding( DimsHW{ 1, 1 } );
    assert(pool0);

    auto basic0 = basicBlock(network, weightMap, *pool0->getOutput(0), 64, 64, 1, "model.layer1.0.");
    auto basic1 = basicBlock(network, weightMap, *basic0->getOutput(0), 64, 64, 1, "model.layer1.1.");
    auto basic2_0 = basicBlock(network, weightMap, *basic1->getOutput(0), 64, 128, 2, "model.layer2.0.");

    auto basic2_1 = basicBlock(network, weightMap, *basic2_0->getOutput(0), 128, 128, 1, "model.layer2.1.");

    auto basic3_0 = basicBlock(network, weightMap, *basic2_1->getOutput(0), 128, 256, 2, "model.layer3.0.");

    auto basic3_1 = basicBlock(network, weightMap, *basic3_0->getOutput(0), 256, 256, 1, "model.layer3.1.");

    auto basic4_0 = basicBlock(network, weightMap, *basic3_1->getOutput(0), 256, 512, 2, "model.layer4.0.");

    auto basic4_1 = basicBlock(network, weightMap, *basic4_0->getOutput(0), 512, 512, 1, "model.layer4.1.");

#if 0
    /* just for debug */
    Dims dims1 = basic4_1->getOutput(0)->getDimensions();
    for (int i = 0; i < dims1.nbDims; i++)
    {
        std::cout << dims1.d[i] << "-" << (int)dims1.type[i] << "   ";
    }
    std::cout << std::endl;
#endif

    auto conv2 = network->addConvolution(*basic4_1->getOutput(0), 8, DimsHW{ 1, 1 }, weightMap["pool.weight"], weightMap["pool.bias"]);
    assert(conv2);
    conv2->setStride(DimsHW{1, 1});
    conv2->setPadding(DimsHW{0, 0});
    conv2->setNbGroups(1);

    IShuffleLayer* permute0 = network->addShuffle(*conv2->getOutput(0));
    assert(permute0);
    permute0->setReshapeDimensions( Dims2{1, 1800});

    auto fcwts0 = network->addConstant(nvinfer1::Dims2(2048, 1800), weightMap["cls.0.weight"]);
    auto matrixMultLayer0 = network->addMatrixMultiply(*permute0->getOutput(0), false, *fcwts0->getOutput(0), true);

    assert(matrixMultLayer0 != nullptr);
    // Add elementwise layer for adding bias
    auto fcbias0 = network->addConstant(nvinfer1::Dims2(1, 2048), weightMap["cls.0.bias"]);

    auto addBiasLayer0 = network->addElementWise(*matrixMultLayer0->getOutput(0), *fcbias0->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer0 != nullptr);

    auto relu = network->addActivation(*addBiasLayer0->getOutput(0), ActivationType::kRELU);

    auto fcwts1 = network->addConstant(nvinfer1::Dims2(22624, 2048), weightMap["cls.2.weight"]);
    auto matrixMultLayer1 = network->addMatrixMultiply(*relu->getOutput(0), false, *fcwts1->getOutput(0), true);

    assert(matrixMultLayer1 != nullptr);
    // Add elementwise layer for adding bias
    auto fcbias1 = network->addConstant(nvinfer1::Dims2(1, 22624), weightMap["cls.2.bias"]);

    auto addBiasLayer1 = network->addElementWise(*matrixMultLayer1->getOutput(0), *fcbias1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer1 != nullptr);

    IShuffleLayer* permute1 = network->addShuffle(*addBiasLayer1->getOutput(0));
    assert(permute1);
    permute1->setReshapeDimensions( Dims3{ 101, 56, 4 });

    permute1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*permute1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    if(builder->platformHasFastFp16()) {
        std::cout << "Platform supports fp16 mode and use it !!!" << std::endl;
        builder->setFp16Mode(true);
    } else {
        std::cout << "Platform doesn't support fp16 mode so you can't use it !!!" << std::endl;
    }
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
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
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float),
          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

std::vector<float> prepareImage(cv::Mat & img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat img_float;

    resized.convertTo(img_float, CV_32FC3, 1. / 255.);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(INPUT_C);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(INPUT_H * INPUT_W * INPUT_C);
    auto data = result.data();
    int channelLength = INPUT_H * INPUT_W;
    static float mean[]= {0.485, 0.456, 0.406};
    static float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < INPUT_C; ++i) {
        cv::Mat normed_channel = (input_channels[i] - mean[i]) / std[i];
        memcpy(data, normed_channel.data, channelLength * sizeof(float));
        data += channelLength;
    }

    return result;
}

/* (101,56,4), add softmax on 101_axis and calculate Expect */
void softmax_mul(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0, wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            float sum = 0.0;
            float expect = 0.0;
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = exp(x[k * wh + i * cols + j]);
                sum += x[k * wh + i * cols + j];
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] /= sum;
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = x[k * wh + i * cols + j] * (k + 1);
                expect += x[k * wh + i * cols + j];
            }
            y[i * cols + j] = expect;
        }
    }
}
/* (101,56,4), calculate max index on 101_axis */
void argmax(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0,wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int max = -10000000;
            int max_ind = -1;
            for(int k = 0; k < chan; k++)
            {
                if(x[k * wh + i * cols + j] > max)
                {
                    max = x[k * wh + i * cols + j];
                    max_ind = k;
                }
            }
            y[i * cols + j] = max_ind;
        }
    }
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
            std::ofstream p("lane_det.engine", std::ios::binary);
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
            std::ifstream file("lane_det.engine", std::ios::binary);
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
            std::cerr << "./crnn -s  // serialize model to plan file" << std::endl;
            std::cerr << "./crnn -d ../samples  // deserialize plan file and run inference" << std::endl;
            return -1;
    }

    /* prepare input data */
    static float data[BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W];
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

    int fcount = 0;
    int vis_h = 720;
    int vis_w = 1280;
    int col_sample_w = 8;
    for (int f = 0; f < (int)file_names.size(); f++)
    {
        cv::Mat vis;
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++)
        {
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b], 1);
            if (img.empty()) continue;
            cv::resize(img, vis, cv::Size(vis_w, vis_h));
            std::vector<float> result(INPUT_C * INPUT_W * INPUT_H);
            result = prepareImage(img);
            memcpy(data, &result[0], INPUT_C * INPUT_W * INPUT_H * sizeof(float));
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE); //prob: size (101, 56, 4)
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time is "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;

        std::vector<int> tusimple_row_anchor
            { 64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284 };

        float max_ind[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
        float prob_reverse[BATCH_SIZE * OUTPUT_SIZE];
        /* do out_j = out_j[:, ::-1, :] in python list*/
        float expect[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
        for (int k = 0, wh = OUTPUT_W * OUTPUT_H; k < OUTPUT_C; k++)
        {
            for(int j = 0; j < OUTPUT_H; j ++)
            {
                for(int l = 0; l < OUTPUT_W; l++)
                {
                    prob_reverse[k * wh + (OUTPUT_H - 1 - j) * OUTPUT_W + l] =
                        prob[k * wh + j * OUTPUT_W + l];
                }
            }
        }

        argmax(prob_reverse, max_ind, OUTPUT_H, OUTPUT_W, OUTPUT_C);
        /* calculate softmax and Expect */
        softmax_mul(prob_reverse, expect, OUTPUT_H, OUTPUT_W, OUTPUT_C);
        for(int k = 0; k < OUTPUT_H; k++) {
            for(int j = 0; j < OUTPUT_W; j++) {
                max_ind[k * OUTPUT_W + j] == 100 ? expect[k * OUTPUT_W + j] = 0 :
                    expect[k * OUTPUT_W + j] = expect[k * OUTPUT_W + j];
            }
        }
        std::vector<int> i_ind;
        for(int k = 0; k < OUTPUT_W; k++) {
            int ii = 0;
            for(int g = 0; g < OUTPUT_H; g++) {
                if(expect[g * OUTPUT_W + k] != 0)
                    ii++;
            }
            if(ii > 2) {
                i_ind.push_back(k);
            }
        }
        for(int k = 0; k < OUTPUT_H; k++) {
            for(int ll = 0; ll < i_ind.size(); ll++) {
                if(expect[OUTPUT_W * k + i_ind[ll]] > 0) {
                    cv::Point pp =
                        { int(expect[OUTPUT_W * k + i_ind[ll]] * col_sample_w * vis_w / INPUT_W) - 1,
                          int( vis_h * tusimple_row_anchor[OUTPUT_H - 1 - k] / INPUT_H) - 1 };
                    cv::circle(vis, pp, 8, CV_RGB(0, 255 ,0), 2);
                }
            }
        }
        cv::imshow("lane_vis",vis);
        cv::waitKey(0);
    }

    return 0;
}
