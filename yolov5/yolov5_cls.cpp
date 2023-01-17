#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "model.h"
#include "config.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kClsNumClass;

void batch_preprocess(std::vector<cv::Mat>& imgs, float* output) {
  for (size_t b = 0; b < imgs.size(); b++) {
    cv::Mat img;
    cv::resize(imgs[b], img, cv::Size(kClsInputW, kClsInputH));
    int i = 0;
    for (int row = 0; row < img.rows; ++row) {
      uchar* uc_pixel = img.data + row * img.step;
      for (int col = 0; col < img.cols; ++col) {
        output[b * 3 * img.rows * img.cols  + i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;  // R - 0.485
        output[b * 3 * img.rows * img.cols + i + img.rows * img.cols] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
        output[b * 3 * img.rows * img.cols + i + 2 * img.rows * img.cols] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
        uc_pixel += 3;
        ++i;
      }
    }
  }
}

std::vector<float> softmax(float *prob, int n) {
  std::vector<float> res;
  float sum = 0.0f;
  float t;
  for (int i = 0; i < n; i++) {
    t = expf(prob[i]);
    res.push_back(t);
    sum += t;
  }
  for (int i = 0; i < n; i++) {
    res[i] /= sum;
  }
  return res;
}

std::vector<int> topk(const std::vector<float>& vec, int k) {
  std::vector<int> topk_index;
  std::vector<size_t> vec_index(vec.size());
  std::iota(vec_index.begin(), vec_index.end(), 0);

  std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

  int k_num = std::min<int>(vec.size(), k);

  for (int i = 0; i < k_num; ++i) {
    topk_index.push_back(vec_index[i]);
  }

  return topk_index;
}

std::vector<std::string> read_classes(std::string file_name) {
  std::vector<std::string> classes;
  std::ifstream ifs(file_name, std::ios::in);
  if (!ifs.is_open()) {
    std::cerr << file_name << " is not found, pls refer to README and download it." << std::endl;
    assert(0);
  }
  std::string s;
  while (std::getline(ifs, s)) {
    classes.push_back(s);
  }
  ifs.close();
  return classes;
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    auto net = std::string(argv[4]);
    if (net[0] == 'n') {
      gd = 0.33;
      gw = 0.25;
    } else if (net[0] == 's') {
      gd = 0.33;
      gw = 0.50;
    } else if (net[0] == 'm') {
      gd = 0.67;
      gw = 0.75;
    } else if (net[0] == 'l') {
      gd = 1.0;
      gw = 1.0;
    } else if (net[0] == 'x') {
      gd = 1.33;
      gw = 1.25;
    } else if (net[0] == 'c' && argc == 7) {
      gd = atof(argv[5]);
      gw = atof(argv[6]);
    } else {
      return false;
    }
  } else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
  } else {
    return false;
  }
  return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_input_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kClsInputH * kClsInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_input_buffer = new float[kBatchSize * 3 * kClsInputH * kClsInputW];
  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
  CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * kClsInputH * kClsInputW * sizeof(float), cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;

  engine = build_cls_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);

  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  std::string wts_name = "";
  std::string engine_name = "";
  float gd = 0.0f, gw = 0.0f;
  std::string img_dir;

  if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_cls -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov5_cls -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  // Create a model using the API directly and serialize it to a file
  if (!wts_name.empty()) {
    serialize_engine(kBatchSize, gd, gw, wts_name, engine_name);
    return 0;
  }

  // Deserialize the engine from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Prepare cpu and gpu buffers
  float* gpu_buffers[2];
  float* cpu_input_buffer = nullptr;
  float* cpu_output_buffer = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_input_buffer, &cpu_output_buffer);

  // Read images from directory
  std::vector<std::string> file_names;
  if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // Read imagenet labels
  auto classes = read_classes("imagenet_classes.txt");

  // batch predict
  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
      cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
      img_batch.push_back(img);
      img_name_batch.push_back(file_names[j]);
    }

    // Preprocess
    batch_preprocess(img_batch, cpu_input_buffer);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_input_buffer, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Postprocess and get top-k result
    for (size_t b = 0; b < img_name_batch.size(); b++) {
      float* p = &cpu_output_buffer[b * kOutputSize];
      auto res = softmax(p, kOutputSize);
      auto topk_idx = topk(res, 3);
      std::cout << img_name_batch[b] << std::endl;
      for (auto idx: topk_idx) {
        std::cout << "  " << classes[idx] << " " << res[idx] << std::endl;
      }
    }
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_input_buffer;
  delete[] cpu_output_buffer;
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}

