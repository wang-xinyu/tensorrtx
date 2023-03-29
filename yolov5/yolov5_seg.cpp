#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir, std::string& labels_filename) {
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
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        labels_filename = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2) {
  assert(engine->getNbBindings() == 3);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
  const int outputIndex2 = engine->getBindingIndex("proto");
  assert(inputIndex == 0);
  assert(outputIndex1 == 1);
  assert(outputIndex2 == 2);

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float)));

  // Alloc CPU buffers
  *cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
  *cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize) {
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;

  engine = build_seg_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);

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
  std::string labels_filename = "";
  float gd = 0.0f, gw = 0.0f;

  std::string img_dir;
  if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir, labels_filename)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_seg -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov5_seg -d [.engine] ../images coco.txt  // deserialize plan file, read the labels file and run inference" << std::endl;
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

  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[3];
  float* cpu_output_buffer1 = nullptr;
  float* cpu_output_buffer2 = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &gpu_buffers[2], &cpu_output_buffer1, &cpu_output_buffer2);

  // Read images from directory
  std::vector<std::string> file_names;
  if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // Read the txt file for classnames
  std::ifstream labels_file(labels_filename, std::ios::binary);
  if (!labels_file.good()) {
    std::cerr << "read " << labels_filename << " error!" << std::endl;
    return -1;
  }
  std::unordered_map<int, std::string> labels_map;
  read_labels(labels_filename, labels_map);
  assert(kNumClass == labels_map.size());

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
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);

    // Draw result and save image
    for (size_t b = 0; b < img_name_batch.size(); b++) {
      auto& res = res_batch[b];
      cv::Mat img = img_batch[b];

      auto masks = process_mask(&cpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);
      draw_mask_bbox(img, res, masks, labels_map);
      cv::imwrite("_" + img_name_batch[b], img);
    }
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  CUDA_CHECK(cudaFree(gpu_buffers[2]));
  delete[] cpu_output_buffer1;
  delete[] cpu_output_buffer2;
  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}

