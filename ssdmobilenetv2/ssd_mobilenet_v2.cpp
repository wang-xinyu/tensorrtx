#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "box_utils.h"

// stuff we know about the network and the input/output blobs
static const int OUTPUT_SIZE_CNF = ssd::NUM_CLASSES*ssd::NUM_DETECTIONS;
static const int OUTPUT_SIZE_BX = ssd::LOCATIONS*ssd::NUM_DETECTIONS;
int COUNTER = 0; // verbose mode to count the layers

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME_CNF = "confidences";
const char* OUTPUT_BLOB_NAME_BX = "locations";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    std::cout << "Total weights: " << count << std::endl;

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

void print_dims(Dims dm, std::string layer, std::string lname){
  if (dm.nbDims == -1) {
    std::cerr << "Error: Dimension size -1" << std::endl;
    return;
  }
  // std::cout << layer << "-" << COUNTER << "          [-1";
  std::cout << std::right << std::setw(21) << layer << "-" << COUNTER;
  std::cout << "          [-1";
  for (size_t i = 0; i < dm.nbDims; i++) {
    std::cout << ", " << dm.d[i];
  }
  std::cout << "]\t\t" << lname << std::endl;
   // std::cout << "]" << std::endl;
  COUNTER = COUNTER + 1;
}

void shape(std::string layer, ITensor& t, std::string lname){
  Dims dm = t.getDimensions();
  print_dims(dm, layer, lname);
}

void shape(std::string layer, ITensor* t, std::string lname){
  Dims dm = t->getDimensions();
  print_dims(dm, layer, lname);
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;

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

    // shape("BatchNorm2d", *scale_1->getOutput(0), lname);
    return scale_1;
}

IElementWiseLayer* convBnRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (ksize - 1) / 2;
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{s, s});
    conv1->setPadding(DimsHW{p, p});
    conv1->setNbGroups(g);

    // shape("Conv2d", *conv1->getOutput(0), lname+"0.weight");

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shval[0] = -6.0;
    scval[0] = 1.0;
    pval[0] = 1.0;
    Weights shift{DataType::kFLOAT, shval, 1};
    Weights scale{DataType::kFLOAT, scval, 1};
    Weights power{DataType::kFLOAT, pval, 1};
    weightMap[lname + "cbr.scale"] = scale;
    weightMap[lname + "cbr.shift"] = shift;
    weightMap[lname + "cbr.power"] = power;
    IScaleLayer* scale1 = network->addScale(*bn1->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power);
    assert(scale1);

    IActivationLayer* relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IElementWiseLayer* ew1 = network->addElementWise(*relu1->getOutput(0), *relu2->getOutput(0), ElementWiseOperation::kSUB);
    assert(ew1);

    // shape("ReLU6", *relu1->getOutput(0), lname);
    return ew1;
}

ILayer* invertedRes(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s, int exp, bool skipFirstConv) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int hidden = inch * exp;
    bool use_res_connect = (s == 1 && inch == outch);

    IScaleLayer *bn1 = nullptr;
    if (exp != 1) {
        // Special case of expanded convolution
        std::map<std::string, Weights> weightMapSubset;
        weightMapSubset["0.weight"] = weightMap[lname+"conv.3.weight"];
        weightMapSubset["1.weight"] = weightMap[lname+"conv.4.weight"];
        weightMapSubset["1.bias"] = weightMap[lname+"conv.4.bias"];
        weightMapSubset["1.running_mean"] = weightMap[lname+"conv.4.running_mean"];
        weightMapSubset["1.running_var"] = weightMap[lname+"conv.4.running_var"];
        weightMapSubset["1.num_batches_tracked"] = weightMap[lname+"conv.4.num_batches_tracked"];
        IElementWiseLayer* ew2;
        if (skipFirstConv) {
          ew2 = convBnRelu(network, weightMapSubset, input, hidden, 3, s, hidden, "");
        } else {
          IElementWiseLayer* ew1 = convBnRelu(network, weightMap, input, hidden, 1, 1, 1, lname + "conv.");
          ew2 = convBnRelu(network, weightMapSubset, *ew1->getOutput(0), hidden, 3, s, hidden, "");
        }
        IConvolutionLayer* conv1 = network->addConvolution(*ew2->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "conv.6.weight"], emptywts);
        assert(conv1);
        // shape("Conv2d", *conv1->getOutput(0), lname+"conv.6.weight");
        bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv.7", 1e-5);
    } else {
        IConvolutionLayer* conv1;
        if (skipFirstConv) {
          conv1 = network->addConvolution(input, outch, DimsHW{1, 1}, weightMap[lname + "conv.3.weight"], emptywts);
        } else{
          IElementWiseLayer* ew1 = convBnRelu(network, weightMap, input, hidden, 3, s, hidden, lname + "conv.");
          conv1 = network->addConvolution(*ew1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "conv.3.weight"], emptywts);
        }
        assert(conv1);
        // shape("Conv2d", *conv1->getOutput(0), lname+"conv.3.weight");
        bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv.4", 1e-5);
    }
    // shape("InvertedResidual", *bn1->getOutput(0), lname);

    if (!use_res_connect) return bn1;
    IElementWiseLayer* ew3 = network->addElementWise(input, *bn1->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew3);
    return ew3;
}

IShuffleLayer* reshapeHeader(INetworkDefinition *network, ITensor& input, int size){
  // Permutation and reshape the tensor for classification
  IShuffleLayer* re = network->addShuffle(input);
  assert(re);

  Permutation p1 = {{1, 2, 0}};
  re->setFirstTranspose(p1);
  re->setReshapeDimensions(Dims2(-1, size));
  // shape("Reshape", *re->getOutput(0), "");
  return re;
}

ILayer* detectionHeader(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int outsize) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IElementWiseLayer* ew1 = convBnRelu(network, weightMap, input, inch, 3, 1, inch, lname);
    IConvolutionLayer* conv1 = network->addConvolution(*ew1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "3.weight"], emptywts);
    conv1->setStride(DimsHW{1, 1});
    assert(conv1);

    // std::cout << std::endl;
    // shape("Conv2d", *conv1->getOutput(0), lname+"3.weight");
    IShuffleLayer* re1 = reshapeHeader(network, *conv1->getOutput(0), outsize);
    return re1;
}

// Create the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt)
{
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ssd::INPUT_C, ssd::INPUT_H, ssd::INPUT_W});
    assert(data);
    // shape("Input", data, "");

    std::map<std::string, Weights> weightMap = loadWeights("../ssd_mobilenet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto ew1 = convBnRelu(network, weightMap, *data, 32, 3, 2, 1, "base_net.0.");
    ILayer* ir1 = invertedRes(network, weightMap, *ew1->getOutput(0), "base_net.1.", 32, 16, 1, 1, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.2.", 16, 24, 2, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.3.", 24, 24, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.4.", 24, 32, 2, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.5.", 32, 32, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.6.", 32, 32, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.7.", 32, 64, 2, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.8.", 64, 64, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.9.", 64, 64, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.10.", 64, 64, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.11.", 64, 96, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.12.", 96, 96, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.13.", 96, 96, 1, 6, false);

    // ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.14.", 96, 160, 2, 6);
    IElementWiseLayer* ew14 = convBnRelu(network, weightMap, *ir1->getOutput(0), 576, 1, 1, 1, "base_net.14.conv.");
    ir1 = invertedRes(network, weightMap, *ew14->getOutput(0), "base_net.14.", 96, 160, 2, 6, true);

    ILayer* ch0 = detectionHeader(network, weightMap, *ew14->getOutput(0), "classification_headers.0.", 576, 126, ssd::NUM_CLASSES);
    ILayer* rh0 = detectionHeader(network, weightMap, *ew14->getOutput(0), "regression_headers.0.", 576, 24, ssd::LOCATIONS);

    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.15.", 160, 160, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.16.", 160, 160, 1, 6, false);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "base_net.17.", 160, 320, 1, 6, false);
    IElementWiseLayer* ew2 = convBnRelu(network, weightMap, *ir1->getOutput(0), 1280, 1, 1, 1, "base_net.18.");

    ILayer* ch1 = detectionHeader(network, weightMap, *ew2->getOutput(0), "classification_headers.1.", 1280, 126, ssd::NUM_CLASSES);
    ILayer* rh1 = detectionHeader(network, weightMap, *ew2->getOutput(0), "regression_headers.1.", 1280, 24, ssd::LOCATIONS);

    ir1 = invertedRes(network, weightMap, *ew2->getOutput(0), "extras.0.", 128, 512, 2, 2, false);
    ILayer* ch2 = detectionHeader(network, weightMap, *ir1->getOutput(0), "classification_headers.2.", 512, 126, ssd::NUM_CLASSES);
    ILayer* rh2 = detectionHeader(network, weightMap, *ir1->getOutput(0), "regression_headers.2.", 512, 24, ssd::LOCATIONS);

    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "extras.1.", 64, 256, 2, 2, false);
    ILayer* ch3 = detectionHeader(network, weightMap, *ir1->getOutput(0), "classification_headers.3.", 256, 126, ssd::NUM_CLASSES);
    ILayer* rh3 = detectionHeader(network, weightMap, *ir1->getOutput(0), "regression_headers.3.", 256, 24, ssd::LOCATIONS);

    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "extras.2.", 64, 256, 2, 2, false);
    ILayer* ch4 = detectionHeader(network, weightMap, *ir1->getOutput(0), "classification_headers.4.", 256, 126, ssd::NUM_CLASSES);
    ILayer* rh4 = detectionHeader(network, weightMap, *ir1->getOutput(0), "regression_headers.4.", 256, 24, ssd::LOCATIONS);

    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "extras.3.", 32, 64, 2, 2, false);
    IConvolutionLayer* conv1 = network->addConvolution(*ir1->getOutput(0), 126, DimsHW{1, 1}, weightMap["classification_headers.5.weight"], emptywts);
    IShuffleLayer* ch5 = reshapeHeader(network, *conv1->getOutput(0), ssd::NUM_CLASSES);

    IConvolutionLayer* conv2 = network->addConvolution(*ir1->getOutput(0), 24, DimsHW{1, 1}, weightMap["regression_headers.5.weight"], emptywts);
    IShuffleLayer* rh5 = reshapeHeader(network, *conv2->getOutput(0), ssd::LOCATIONS);

    // std::cout << std::endl;
    // shape("InvertedResidual", *ir1->getOutput(0), "classification_headers.5.weights");
    // shape("Conv2d", *ch5->getOutput(0), "classification_headers.5.weight");
    // shape("Conv2d", *rh5->getOutput(0), "regression_headers.5.weight");

    // std::cout << std::endl << "Detection output layers" << endl;
    ITensor* chTensor[] = {ch0->getOutput(0), ch1->getOutput(0), ch2->getOutput(0), ch3->getOutput(0), ch4->getOutput(0), ch5->getOutput(0)};
    IConcatenationLayer* chConcat = network->addConcatenation(chTensor, 6);
    ISoftMaxLayer* conf = network->addSoftMax(*chConcat->getOutput(0));
    conf->setAxes(1<<1);

    // shape("Concat", *chConcat->getOutput(0), "classification");
    // shape("Softmax", *conf->getOutput(0), "classification");

    ITensor* rhTensor[] = {rh0->getOutput(0), rh1->getOutput(0), rh2->getOutput(0), rh3->getOutput(0), rh4->getOutput(0), rh5->getOutput(0)};
    IConcatenationLayer* rhConcat = network->addConcatenation(rhTensor, 6);
    // shape("Concat", *rhConcat->getOutput(0), "regression");

    std::cout << std::endl << "Total layers in the network: " << network->getNbLayers() << std::endl;

    // Mobilenet classifier code
    std::cout <<endl << "Building engine ..." << endl;
    conf->getOutput(0)->setName(OUTPUT_BLOB_NAME_CNF);
    network->markOutput(*conf->getOutput(0));

    rhConcat->getOutput(0)->setName(OUTPUT_BLOB_NAME_BX);
    network->markOutput(*rhConcat->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildCudaEngine(*network);

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
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

void doInference(IExecutionContext& context, float* input, float* output_cnf, float* output_bx, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndexCnf = engine.getBindingIndex(OUTPUT_BLOB_NAME_CNF);
    const int outputIndexBx = engine.getBindingIndex(OUTPUT_BLOB_NAME_BX);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndexCnf], batchSize * OUTPUT_SIZE_CNF * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndexBx], batchSize * OUTPUT_SIZE_BX * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output_cnf, buffers[outputIndexCnf], batchSize * OUTPUT_SIZE_CNF * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_bx, buffers[outputIndexBx], batchSize * OUTPUT_SIZE_BX * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndexCnf]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mobilenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mobilenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("ssd_mobilenet.engine");
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        std::cout << "Saved as ssd_mobilenet.engine" << endl;
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("ssd_mobilenet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
            std::cout << "Engine file read successful" << std::endl;
        }
    } else {
      return -1;
    }

    // Subtract mean from image
    float data[ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W];
    for (int i = 0; i < ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W; i++)
        data[i] = 1.0;
    // prepare_input(argv[2], data);

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE_CNF], locations[OUTPUT_SIZE_BX];
    float av = 0.0;
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, locations, 1);
        auto end = std::chrono::system_clock::now();
        av = av + std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    std::cout << "Average inference time: " << av/100 << "ms" << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::vector<ssd::Detection> dt = post_process_output(prob, locations, 0.5);
    std::cout << "Num detections " << dt.size() << endl;
    // save_inference(std::string(argv[2]), dt);

    for (size_t i = 0; i < dt.size(); i++) {
      std::cout << endl << "Det " << i << ":";
      std::cout << " Class: " << dt[i].class_id;
      std::cout << " Conf%: " << dt[i].conf *100;
      std::cout << " BB: [ ";
      for (size_t j = 0; j < ssd::LOCATIONS; j++) {
        std::cout << dt[i].bbox[j] << " ";
      }
      std::cout << "]";
    }
    std::cout << endl;
    return 0;
}
