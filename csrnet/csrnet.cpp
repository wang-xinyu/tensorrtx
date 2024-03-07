#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <chrono>
#include <config.h>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <logging.h>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace nvinfer1;

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cerr << "Cuda failure: " << ret << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static Logger gLogger;
static char *kWTSFile = "";
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

  while (count--) {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    wt.type = DataType::kFLOAT;

    // Load blob
    uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
    for (uint32_t x = 0, y = size; x < y; ++x) {
      input >> std::hex >> val[x];
    }
    wt.values = val;

    wt.count = size;
    weightMap[name] = wt;
  }

  return weightMap;
}
// clang-format off
/*
CSRNet(
 (frontend): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
  )
  (backend): Sequential(
    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2),
    dilation=(2, 2)) (1): ReLU(inplace=True) (2): Conv2d(512, 512,
    kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2)) (3):
    ReLU(inplace=True) (4): Conv2d(512, 512, kernel_size=(3, 3), stride=(1,
    1), padding=(2, 2), dilation=(2, 2)) (5): ReLU(inplace=True) (6):
    Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2),
    dilation=(2, 2)) (7): ReLU(inplace=True) (8): Conv2d(256, 128,
    kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2)) (9):
    ReLU(inplace=True) (10): Conv2d(128, 64, kernel_size=(3, 3), stride=(1,
    1), padding=(2, 2), dilation=(2, 2)) (11): ReLU(inplace=True)
  )
  (output_layer): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
)
*/
// clang-format on
void doInference(IExecutionContext &context, float *input, float *output,
                 int input_h, int input_w) {
  const ICudaEngine &engine = context.getEngine();

  uint64_t input_size = 3 * input_h * input_w * sizeof(float);
  uint64_t output_size = ((input_h * input_w) >> 6) * sizeof(float);

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(kInputTensorName);
  const int outputIndex = engine.getBindingIndex(kOutputTensorName);
  context.setBindingDimensions(inputIndex, Dims4(1, 3, input_h, input_w));

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], input_size));
  CHECK(cudaMalloc(&buffers[outputIndex], output_size));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_size,
                        cudaMemcpyHostToDevice, stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  context.enqueueV2(buffers, stream, nullptr);
  std::cout << "enqueueV2 time: "
            << std::chrono::duration<float>(
                   std::chrono::high_resolution_clock::now() - t1)
                   .count()
            << "s" << std::endl;
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size,
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder,
                          IBuilderConfig *config, DataType dt) {

  //   INetworkDefinition *network = builder->createNetworkV2(0U);
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
  ITensor *data = network->addInput(kInputTensorName, dt, Dims4{1, 3, -1, -1});
  assert(data);
  std::map<std::string, Weights> weightMap = loadWeights(kWTSFile);

  IConvolutionLayer *conv1 = network->addConvolutionNd(
      *data, 64, DimsHW{3, 3}, weightMap["frontend.0.weight"],
      weightMap["frontend.0.bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{1, 1});
  conv1->setPaddingNd(DimsHW{1, 1});

  IActivationLayer *relu1 =
      network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

  assert(relu1);

  auto conv2 = network->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3},
                                         weightMap["frontend.2.weight"],
                                         weightMap["frontend.2.bias"]);
  assert(conv2);
  conv2->setStrideNd(DimsHW{1, 1});
  conv2->setPaddingNd(DimsHW{1, 1});
  auto relu2 =
      network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
  assert(relu2);
  auto pool1 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX,
                                     DimsHW{2, 2});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});
  auto conv3 = network->addConvolutionNd(
      *pool1->getOutput(0), 128, DimsHW{3, 3}, weightMap["frontend.5.weight"],
      weightMap["frontend.5.bias"]);
  assert(conv3);
  conv3->setStrideNd(DimsHW{1, 1});

  conv3->setPaddingNd(DimsHW{1, 1});
  auto relu3 =
      network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
  assert(relu3);

  auto conv4 = network->addConvolutionNd(
      *relu3->getOutput(0), 128, DimsHW{3, 3}, weightMap["frontend.7.weight"],
      weightMap["frontend.7.bias"]);
  assert(conv4);
  conv4->setStrideNd(DimsHW{1, 1});
  conv4->setPaddingNd(DimsHW{1, 1});
  auto relu4 =
      network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
  assert(relu4);

  auto pool2 = network->addPoolingNd(*relu4->getOutput(0), PoolingType::kMAX,
                                     DimsHW{2, 2});
  assert(pool2);
  pool2->setStrideNd(DimsHW{2, 2});

  auto conv5 = network->addConvolutionNd(
      *pool2->getOutput(0), 256, DimsHW{3, 3}, weightMap["frontend.10.weight"],
      weightMap["frontend.10.bias"]);
  assert(conv5);
  conv5->setStrideNd(DimsHW{1, 1});
  conv5->setPaddingNd(DimsHW{1, 1});
  auto relu5 =
      network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
  assert(relu5);

  auto conv6 = network->addConvolutionNd(
      *relu5->getOutput(0), 256, DimsHW{3, 3}, weightMap["frontend.12.weight"],
      weightMap["frontend.12.bias"]);
  assert(conv6);
  conv6->setStrideNd(DimsHW{1, 1});
  conv6->setPaddingNd(DimsHW{1, 1});
  auto relu6 =
      network->addActivation(*conv6->getOutput(0), ActivationType::kRELU);
  assert(relu6);
  auto conv7 = network->addConvolutionNd(
      *relu6->getOutput(0), 256, DimsHW{3, 3}, weightMap["frontend.14.weight"],
      weightMap["frontend.14.bias"]);
  assert(conv7);
  conv7->setStrideNd(DimsHW{1, 1});
  conv7->setPaddingNd(DimsHW{1, 1});
  auto relu7 =
      network->addActivation(*conv7->getOutput(0), ActivationType::kRELU);
  assert(relu7);
  auto pool3 = network->addPoolingNd(*relu7->getOutput(0), PoolingType::kMAX,
                                     DimsHW{2, 2});
  assert(pool3);
  pool3->setStrideNd(DimsHW{2, 2});
  auto conv8 = network->addConvolutionNd(
      *pool3->getOutput(0), 512, DimsHW{3, 3}, weightMap["frontend.17.weight"],
      weightMap["frontend.17.bias"]);
  assert(conv8);
  conv8->setStrideNd(DimsHW{1, 1});
  conv8->setPaddingNd(DimsHW{1, 1});
  auto relu8 =
      network->addActivation(*conv8->getOutput(0), ActivationType::kRELU);
  assert(relu8);
  auto conv9 = network->addConvolutionNd(
      *relu8->getOutput(0), 512, DimsHW{3, 3}, weightMap["frontend.19.weight"],
      weightMap["frontend.19.bias"]);
  assert(conv9);
  conv9->setStrideNd(DimsHW{1, 1});
  conv9->setPaddingNd(DimsHW{1, 1});
  auto relu9 =
      network->addActivation(*conv9->getOutput(0), ActivationType::kRELU);
  assert(relu9);
  auto conv10 = network->addConvolutionNd(
      *relu9->getOutput(0), 512, DimsHW{3, 3}, weightMap["frontend.21.weight"],
      weightMap["frontend.21.bias"]);
  assert(conv10);
  conv10->setStrideNd(DimsHW{1, 1});
  conv10->setPaddingNd(DimsHW{1, 1});
  auto relu10 =
      network->addActivation(*conv10->getOutput(0), ActivationType::kRELU);
  assert(relu10);
  // backend
  auto conv11 = network->addConvolutionNd(
      *relu10->getOutput(0), 512, DimsHW{3, 3}, weightMap["backend.0.weight"],
      weightMap["backend.0.bias"]);
  assert(conv11);
  conv11->setPaddingNd(DimsHW{2, 2});
  conv11->setStrideNd(DimsHW{1, 1});
  conv11->setDilationNd(DimsHW{2, 2});
  auto relu11 =
      network->addActivation(*conv11->getOutput(0), ActivationType::kRELU);

  assert(relu11);
  auto conv12 = network->addConvolutionNd(
      *relu11->getOutput(0), 512, DimsHW{3, 3}, weightMap["backend.2.weight"],
      weightMap["backend.2.bias"]);
  assert(conv12);
  conv12->setPaddingNd(DimsHW{2, 2});
  conv12->setStrideNd(DimsHW{1, 1});
  conv12->setDilationNd(DimsHW{2, 2});
  auto relu12 =
      network->addActivation(*conv12->getOutput(0), ActivationType::kRELU);
  assert(relu12);

  auto conv13 = network->addConvolutionNd(
      *relu12->getOutput(0), 512, DimsHW{3, 3}, weightMap["backend.4.weight"],
      weightMap["backend.4.bias"]);
  assert(conv13);
  conv13->setPaddingNd(DimsHW{2, 2});
  conv13->setStrideNd(DimsHW{1, 1});
  conv13->setDilationNd(DimsHW{2, 2});
  auto relu13 =
      network->addActivation(*conv13->getOutput(0), ActivationType::kRELU);
  assert(relu13);

  auto conv14 = network->addConvolutionNd(
      *relu13->getOutput(0), 256, DimsHW{3, 3}, weightMap["backend.6.weight"],
      weightMap["backend.6.bias"]);
  assert(conv14);
  conv14->setPaddingNd(DimsHW{2, 2});
  conv14->setStrideNd(DimsHW{1, 1});
  conv14->setDilationNd(DimsHW{2, 2});
  auto relu14 =
      network->addActivation(*conv14->getOutput(0), ActivationType::kRELU);
  assert(relu14);
  auto conv15 = network->addConvolutionNd(
      *relu14->getOutput(0), 128, DimsHW{3, 3}, weightMap["backend.8.weight"],
      weightMap["backend.8.bias"]);
  assert(conv15);
  conv15->setPaddingNd(DimsHW{2, 2});
  conv15->setStrideNd(DimsHW{1, 1});
  conv15->setDilationNd(DimsHW{2, 2});
  auto relu15 =
      network->addActivation(*conv15->getOutput(0), ActivationType::kRELU);
  assert(relu15);
  auto conv16 = network->addConvolutionNd(
      *relu15->getOutput(0), 64, DimsHW{3, 3}, weightMap["backend.10.weight"],
      weightMap["backend.10.bias"]);
  assert(conv16);
  conv16->setPaddingNd(DimsHW{2, 2});
  conv16->setStrideNd(DimsHW{1, 1});
  conv16->setDilationNd(DimsHW{2, 2});
  auto relu16 =
      network->addActivation(*conv16->getOutput(0), ActivationType::kRELU);

  assert(relu16);

  auto conv17 = network->addConvolutionNd(
      *relu16->getOutput(0), 1, DimsHW{1, 1}, weightMap["output_layer.weight"],
      weightMap["output_layer.bias"]);
  assert(conv17);

  conv17->setStrideNd(DimsHW{1, 1});
  conv17->getOutput(0)->setName(kOutputTensorName);
  network->markOutput(*conv17->getOutput(0));

  IOptimizationProfile *profile = builder->createOptimizationProfile();
  profile->setDimensions(kInputTensorName, OptProfileSelector::kMIN,
                         Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
  profile->setDimensions(kInputTensorName, OptProfileSelector::kOPT,
                         Dims4(1, 3, OPT_INPUT_H, OPT_INPUT_W));
  profile->setDimensions(kInputTensorName, OptProfileSelector::kMAX,
                         Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
  config->addOptimizationProfile(profile);

  builder->setMaxBatchSize(kBatchSize);
  config->setMaxWorkspaceSize(16 << 20);
#ifdef USE_FP16
  config->setFlag(BuilderFlag::kFP16);
#endif
  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

  printf("build engine successfully : %s\n", kEngineFile);
  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto &mem : weightMap) {
    free((void *)(mem.second.values));
  }

  return engine;
}
void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  ICudaEngine *engine =
      createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  config->destroy();
  builder->destroy();
}

int read_files_in_dir(const char *p_dir_name,
                      std::vector<std::string> &file_names) {
  DIR *p_dir = opendir(p_dir_name);
  if (p_dir == nullptr) {
    return -1;
  }

  struct dirent *p_file = nullptr;
  while ((p_file = readdir(p_dir)) != nullptr) {
    if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
      std::string cur_file_name(p_file->d_name);
      file_names.push_back(cur_file_name);
    }
  }
  closedir(p_dir);
  return 0;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./csrnet -s  ./csrnet.wts // serialize model to plan file"
              << std::endl;
    std::cerr
        << "./csrnet -d  ../images  // deserialize plan file and run inference"
        << std::endl;
    return -1;
  }
  char *trtModelStream{nullptr};
  size_t size{0};

  if (std::string(argv[1]) == "-s") {
    IHostMemory *modelStream{nullptr};
    kWTSFile = argv[2];
    APIToModel(kBatchSize, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p(kEngineFile, std::ios::binary);
    if (!p) {
      std::cerr << "could not open plan output file" << std::endl;
      return -1;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()),
            modelStream->size());
    modelStream->destroy();
    return 1;
  } else if (std::string(argv[1]) == "-d") {
    std::ifstream file(kEngineFile, std::ios::binary);
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
  IRuntime *runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;

  std::vector<std::string> file_names;
  if (read_files_in_dir(argv[2], file_names) < 0) {
    std::cout << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  std::vector<float> mean_value{0.406, 0.456, 0.485}; // BGR
  std::vector<float> std_value{0.225, 0.224, 0.229};

  int fcount = 0;

  float *data = new float[kMaxInputImageSize];
  float *prob = new float[kMaxOutputProbSize];

  for (auto f : file_names) {
    fcount++;
    cv::Mat src_img = cv::imread(std::string(argv[2]) + "/" + f);
    if (src_img.empty())
      continue;

    int i = 0;
    for (int row = 0; row < src_img.rows; ++row) {
      uchar *uc_pixel = src_img.data + row * src_img.step;
      for (int col = 0; col < src_img.cols; ++col) {
        data[i] = (uc_pixel[2] / 255.0 - mean_value[2]) / std_value[2];
        data[i + src_img.rows * src_img.cols] =
            (uc_pixel[1] / 255.0 - mean_value[1]) / std_value[1];
        data[i + 2 * src_img.rows * src_img.cols] =
            (uc_pixel[0] / 255.0 - mean_value[0]) / std_value[0];
        uc_pixel += 3;
        ++i;
      }
    }
    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, src_img.rows, src_img.cols);
    auto end = std::chrono::system_clock::now();
    std::cout << "detect time:"
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    float num = std::accumulate(
        prob, prob + ((src_img.rows * src_img.cols) >> 6), 0.0f);

    cv::Mat densityMap(src_img.rows >> 3, src_img.cols >> 3, CV_32FC1,
                       (void *)prob);

    cv::Mat densityMapScaled;
    cv::normalize(densityMap, densityMapScaled, 0, 255, cv::NORM_MINMAX,
                  CV_8UC1);
    cv::Mat densityColorMap;
    cv::applyColorMap(densityMapScaled, densityColorMap, cv::COLORMAP_VIRIDIS);

    cv::resize(densityColorMap, densityColorMap, src_img.size());
    cv::addWeighted(densityColorMap, 0.5, src_img, 0.5, 0, src_img);

    // write to jpg
    cv::putText(src_img, std::string("people num: ") + std::to_string(num),
                cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
    std::string write_path = std::string(argv[2]) + "result_" + f;
    std::cout << "people num :" << num << " write_path: " << write_path
              << std::endl;
    cv::imwrite(write_path, src_img);
  }
  delete[] data;
  delete[] prob;

  return 0;
}