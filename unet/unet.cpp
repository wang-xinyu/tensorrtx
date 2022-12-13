#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define DEVICE 0
#define USE_FP32  // USE_FP32 or USE_FP16
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define cls 2
#define BILINEAR false

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 959;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W * cls;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

using namespace nvinfer1;

ILayer* doubleConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {
  Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
  IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".double_conv.0.weight"], emptywts);
  conv1->setStrideNd(DimsHW{ 1, 1 });
  conv1->setPaddingNd(DimsHW{ 1, 1 });
  conv1->setNbGroups(1);
  IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".double_conv.1", 0);
  IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
  IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".double_conv.3.weight"], emptywts);
  conv2->setStrideNd(DimsHW{ 1, 1 });
  conv2->setPaddingNd(DimsHW{ 1, 1 });
  conv2->setNbGroups(1);
  IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".double_conv.4", 0);
  IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
  assert(relu2);
  return relu2;
}

ILayer* down(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int p, std::string lname) {
  IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
  assert(pool1);
  ILayer* dcov1 = doubleConv(network, weightMap, *pool1->getOutput(0), outch, 3, lname + ".maxpool_conv.1", outch);
  assert(dcov1);
  return dcov1;
}

ILayer* up(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
  if (BILINEAR) {
    // add upsample bilinear
    IResizeLayer* deconv1 = network->addResize(input1);
    auto outdims = input2.getDimensions();
    deconv1->setOutputDimensions(outdims);
    deconv1->setResizeMode(ResizeMode::kLINEAR);
    deconv1->setAlignCorners(true);

    int diffx = input2.getDimensions().d[1] - deconv1->getOutput(0)->getDimensions().d[1];
    int diffy = input2.getDimensions().d[2] - deconv1->getOutput(0)->getDimensions().d[2];

    ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ diffx / 2, diffy / 2 }, DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
    // dcov1->setPaddingNd(DimsHW{diffx / 2, diffx - diffx / 2},DimsHW{diffy / 2, diffy - diffy / 2});
    ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);
    assert(cat);
    if (midch == 64) {
      ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), outch, 3, lname + ".conv", outch);
      assert(dcov1);
      return dcov1;
    } else {
      int midch1 = outch / 2;
      ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), midch1, 3, lname + ".conv", outch);
      assert(dcov1);
      return dcov1;
    }
  } else {
    IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, weightMap[lname + ".up.weight"], weightMap[lname + ".up.bias"]);
    deconv1->setStrideNd(DimsHW{ 2, 2 });
    deconv1->setNbGroups(1);

    int diffx = input2.getDimensions().d[1] - deconv1->getOutput(0)->getDimensions().d[1];
    int diffy = input2.getDimensions().d[2] - deconv1->getOutput(0)->getDimensions().d[2];

    ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ diffx / 2, diffy / 2 }, DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
    // dcov1->setPaddingNd(DimsHW{diffx / 2, diffx - diffx / 2},DimsHW{diffy / 2, diffy - diffy / 2});
    ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);
    assert(cat);
    ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), midch, 3, lname + ".conv", outch);
    assert(dcov1);
    return dcov1;
  }
}

ILayer* outConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
  // Weights emptywts{DataType::kFLOAT, nullptr, 0};
  IConvolutionLayer* conv1 = network->addConvolutionNd(input, cls, DimsHW{ 1, 1 }, weightMap[lname + ".conv.weight"], weightMap[lname + ".conv.bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{ 1, 1 });
  conv1->setPaddingNd(DimsHW{ 0, 0 });
  conv1->setNbGroups(1);
  return conv1;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string wts_path) {
  INetworkDefinition* network = builder->createNetworkV2(0U);

  // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
  ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
  assert(data);

  std::map<std::string, Weights> weightMap = loadWeights(wts_path);
  Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

  // build network
  auto x1 = doubleConv(network, weightMap, *data, 64, 3, "inc", 64);
  auto x2 = down(network, weightMap, *x1->getOutput(0), 128, 1, "down1");
  auto x3 = down(network, weightMap, *x2->getOutput(0), 256, 1, "down2");
  auto x4 = down(network, weightMap, *x3->getOutput(0), 512, 1, "down3");
  auto channel = 512;
  if (!BILINEAR) {
    channel = 1024;
  }
  auto x5 = down(network, weightMap, *x4->getOutput(0), channel, 1, "down4");
  ILayer* x6 = up(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 512, 512, 512, "up1");
  ILayer* x7 = up(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up2");
  ILayer* x8 = up(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up3");
  ILayer* x9 = up(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up4");
  ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), OUTPUT_SIZE, "outc");

  x10->getOutput(0)->setName(OUTPUT_BLOB_NAME);
  network->markOutput(*x10->getOutput(0));

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

void APIToModel(unsigned int maxBatchSize, IHostMemory** model_stream, std::string wts_path) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wts_path);
  assert(engine != nullptr);

  // Serialize the engine
  (*model_stream) = engine->serialize();

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
  cudaSetDevice(DEVICE);

  char* trt_model_stream = nullptr;
  size_t size = 0;
  std::string engine_name = "unet.engine";
  std::string wts_path = "unet.wts";

  if (argc == 2 && std::string(argv[1]) == "-s") {
    // Create a TensorRT model and serialize it to a file
    IHostMemory* model_stream{ nullptr };
    APIToModel(BATCH_SIZE, &model_stream, wts_path);
    assert(model_stream != nullptr);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
      std::cerr << "could not open plan output file" << std::endl;
      return -1;
    }
    p.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
    model_stream->destroy();
    return 0;
  } else if (argc == 3 && std::string(argv[1]) == "-d") {
    // Load engine file
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trt_model_stream = new char[size];
      assert(trt_model_stream);
      file.read(trt_model_stream, size);
      file.close();
    }
  } else {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./unet -s  // serialize model to plan file" << std::endl;
    std::cerr << "./unet -d ../samples  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  // Prepare input output data
  static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
  static float prob[BATCH_SIZE * OUTPUT_SIZE];

  // Deserialize engine
  IRuntime* runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream, size);
  assert(engine != nullptr);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trt_model_stream;

  cv::Mat img = cv::imread(argv[2]);

  // Preprocess
  cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
  for (int i = 0; i < INPUT_H * INPUT_W; i++) {
    data[i] = (img.at<cv::Vec3b>(i)[2]) / 255.0;
    data[i + INPUT_H * INPUT_W] = (img.at<cv::Vec3b>(i)[1]) / 255.0;
    data[i + 2 * INPUT_H * INPUT_W] = (img.at<cv::Vec3b>(i)[0]) / 255.0;
  }

  // Run inference
  auto start = std::chrono::system_clock::now();
  doInference(*context, data, prob, BATCH_SIZE);
  auto end = std::chrono::system_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  // Postprocess
  cv::Mat result = cv::Mat::zeros(INPUT_H, INPUT_W, CV_8UC3);
  for (int i = 0; i < INPUT_H * INPUT_W; i++) {
    float fmax = prob[i];
    int index = 0;
    for (int j = 1; j < cls; j++) {
      if (prob[i + j * INPUT_H * INPUT_W] > fmax) {
        index = j;
        fmax = prob[i + j * INPUT_H * INPUT_W];
      }
    }

    if (index == 1) {
      result.at<cv::Vec3b>(i) = cv::Vec3b(255, 255, 255);
    }
  }

  cv::imwrite("result.jpg", result);

  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
