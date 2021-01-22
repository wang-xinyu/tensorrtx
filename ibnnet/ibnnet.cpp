#include "ibnnet.h"

//#define USE_FP16

namespace trt {

    IBNNet::IBNNet(trt::EngineConfig &enginecfg, const IBN ibn) : _engineCfg(enginecfg) {
        switch(ibn) {
            case IBN::A:
                _ibn = "a"; 
                break;
            case IBN::B:
                _ibn = "b"; 
                break;
            case IBN::NONE:
            default:
                _ibn = "";
                break;
        }
    }

    // create the engine using only the API and not any parser.
    ICudaEngine *IBNNet::createEngine(IBuilder* builder, IBuilderConfig* config) {
        // resnet50-ibna, resnet50-ibnb, resnet50
        assert(_ibn == "a" or _ibn == "b" or _ibn == "");
        INetworkDefinition* network = builder->createNetworkV2(0U);

        // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
        ITensor* data = network->addInput(_engineCfg.input_name, _dt, Dims3{3, _engineCfg.input_h, _engineCfg.input_w});
        assert(data);

        std::string path;
        if(_ibn == "") {
            path = "../resnet50.wts";
        } else {
            path = "../resnet50-ibn" + _ibn + ".wts";
        }

        std::map<std::string, Weights> weightMap = loadWeights(path);
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        std::map<std::string, std::vector<std::string>> ibn_layers{ 
            { "a", {"a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "", "", ""}},
            { "b", {"", "", "b", "", "", "","b", "", "", "", "", "", "", "", "", "",}},
            { "", {16, ""}}};

        const float mean[3] = {0.485, 0.456, 0.406}; // rgb
        const float std[3] = {0.229, 0.224, 0.225};
        ITensor* pre_input = MeanStd(network, weightMap, data, "", mean, std, false);

        IConvolutionLayer* conv1 = network->addConvolutionNd(*pre_input, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{3, 3});

        IActivationLayer* relu1{nullptr};
        if (_ibn == "b") {
            IScaleLayer* bn1 = addInstanceNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
            relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        } else {
            IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
            relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        }
        assert(relu1);

        // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
        assert(pool1);
        pool1->setStrideNd(DimsHW{2, 2});
        pool1->setPaddingNd(DimsHW{1, 1});

        IActivationLayer* x = bottleneck_ibn(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.", ibn_layers[_ibn][0]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.", ibn_layers[_ibn][1]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.", ibn_layers[_ibn][2]);

        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0.", ibn_layers[_ibn][3]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1.", ibn_layers[_ibn][4]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2.", ibn_layers[_ibn][5]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3.", ibn_layers[_ibn][6]);

        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0.", ibn_layers[_ibn][7]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1.", ibn_layers[_ibn][8]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2.", ibn_layers[_ibn][9]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3.", ibn_layers[_ibn][10]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4.", ibn_layers[_ibn][11]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5.", ibn_layers[_ibn][12]);

        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.", ibn_layers[_ibn][13]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.", ibn_layers[_ibn][14]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2.", ibn_layers[_ibn][15]);

        IPoolingLayer* pool2 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
        assert(pool2);
        pool2->setStrideNd(DimsHW{1, 1});
        
        IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
        assert(fc1);

        fc1->getOutput(0)->setName(_engineCfg.output_name);
        std::cout << "set name out" << std::endl;
        network->markOutput(*fc1->getOutput(0));

        // Build engine
        builder->setMaxBatchSize(_engineCfg.max_batch_size);
        config->setMaxWorkspaceSize(1 << 20);

    #ifdef USE_FP16
        config->setFlag(BuilderFlag::kFP16);
    #endif
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        std::cout << "build out" << std::endl;

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto& mem : weightMap) {
            free((void*) (mem.second.values));
        }

        return engine;
    }

    bool IBNNet::serializeEngine() {
        // Create builder
        auto builder = make_holder(createInferBuilder(gLogger));
        auto config = make_holder(builder->createBuilderConfig());
        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine *engine = createEngine(builder.get(), config.get());
        assert(engine);

        // Serialize the engine
        TensorRTHolder<IHostMemory> modelStream = make_holder(engine->serialize());
        assert(modelStream);

        std::ofstream p("./ibnnet.engine", std::ios::binary | std::ios::out);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return false;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

        return true;
    }

    bool IBNNet::deserializeEngine() {
        std::ifstream file("./ibnnet.engine", std::ios::binary | std::ios::in);
        if (file.good()) {
            file.seekg(0, file.end);
            _engineCfg.stream_size = file.tellg();
            file.seekg(0, file.beg);
            _engineCfg.trtModelStream = std::shared_ptr<char>( new char[_engineCfg.stream_size], []( char* ptr ){ delete [] ptr; } );
            assert(_engineCfg.trtModelStream.get());
            file.read(_engineCfg.trtModelStream.get(), _engineCfg.stream_size);
            file.close();
    
            _inferEngine = make_unique<trt::InferenceEngine>(_engineCfg);
            return true;
        }
        return false;
    }

    void IBNNet::preprocessing(const cv::Mat& img, float* const data, const std::size_t stride) {
        for (std::size_t i = 0; i < stride; ++i) { 
            data[i] = img.at<cv::Vec3b>(i)[2] / 255.0; 
            data[i + stride] = img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + (stride<<1)] = img.at<cv::Vec3b>(i)[0] / 255.0;
        }
    }

    bool IBNNet::inference(std::vector<cv::Mat> &input) {
        if(_inferEngine != nullptr) {
            const std::size_t stride = _engineCfg.input_w * _engineCfg.input_h;
            return _inferEngine.get()->doInference(input.size(), 
                [&](float* data) {
                    for(const auto &img : input) {
                        preprocessing(img, data, stride);
                        data += 3 * stride;
                    }
                }
            );
        } else {
            return false;
        }
    }

    float* IBNNet::getOutput() { 
        if(_inferEngine != nullptr) 
            return _inferEngine.get()->getOutput(); 
        return nullptr;
    }

    int IBNNet::getDeviceID() { 
        return _engineCfg.device_id; 
    }

}