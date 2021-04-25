#include "layers_api.h"

namespace trtxlayers {
    IScaleLayer* addBatchNorm2d(
        INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        std::string lname, 
        float eps
    )
    {
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

    IActivationLayer* basicConv2d(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        int outch, 
        DimsHW ksize, 
        int s, 
        DimsHW p, 
        std::string lname
    )
    {
        // empty wts for bias
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        // add conv -> bn -> relu
        IConvolutionLayer* conv = network -> addConvolutionNd(input, outch, ksize, weightMap[lname + ".conv.weight"], emptywts);
        assert(conv);
        conv -> setStrideNd(DimsHW{s, s});
        conv -> setPaddingNd(p);

        IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv -> getOutput(0), lname + ".bn", 1e-3);
        
        IActivationLayer* relu = network -> addActivation(*bn -> getOutput(0), ActivationType::kRELU);
        assert(relu); 
        return relu;
    }

    IConcatenationLayer* mixed_3a(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        // branch 0
        IPoolingLayer* pool = network -> addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
        assert(pool);
        pool -> setStrideNd(DimsHW{2, 2});

        // branch 1
        IActivationLayer* relu = basicConv2d(network, weightMap, input, 96, DimsHW{ 3, 3 }, 2, DimsHW{ 0, 0 }, lname + ".conv");
        
        // concatenate two branches
        ITensor* inputTensors[] = { pool -> getOutput(0), relu -> getOutput(0) };
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 2);
        assert(cat);
        return cat;
    }

    IConcatenationLayer* mixed_4a(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        // branch 0
        IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 64, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch0.0");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 96, DimsHW{ 3, 3 }, 1, DimsHW{ 0, 0 }, lname + ".branch0.1");

        // branch 1
        IActivationLayer* relu2 = basicConv2d(network, weightMap, input, 64, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch1.0");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 64, DimsHW{ 1, 7 }, 1, DimsHW{ 0, 3 }, lname + ".branch1.1");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 64, DimsHW{ 7, 1 }, 1, DimsHW{ 3, 0 }, lname + ".branch1.2");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 96, DimsHW{ 3, 3 }, 1, DimsHW{ 0, 0 }, lname + ".branch1.3");

        // concatenate two branches
        ITensor* inputTensors[] = { relu1 -> getOutput(0), relu2 -> getOutput(0) };
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 2);
        assert(cat);
        return cat;
    }

    IConcatenationLayer* mixed_5a(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        std::cout<<"mixed_5a"<<std::endl;
        //branch 0
        IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 192, DimsHW{ 3, 3 }, 2, DimsHW{ 0, 0 }, lname + ".conv");

        //branch 1
        IPoolingLayer* pool1 = network -> addPoolingNd(input, PoolingType::kMAX, DimsHW{ 3, 3 });
        assert(pool1);
        pool1 -> setStrideNd(DimsHW{ 2, 2 });

        // concatenate branches
        ITensor* inputTensors[] = { relu1 -> getOutput(0), pool1 -> getOutput(0)};
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 2);
        assert(cat);
        std::cout<<"mixed_5a done"<<std::endl;
        return cat;
    }

    IConcatenationLayer* inceptionA(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        // branch 0
        IActivationLayer* relu0 = basicConv2d(network, weightMap, input, 96, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch0");

        // branch 1
        IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 64, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname +".branch1.0");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 96, DimsHW{ 3, 3 }, 1, DimsHW{ 1, 1 }, lname+".branch1.1");
        
        // branch 2
        IActivationLayer* relu2 = basicConv2d(network, weightMap, input, 64, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname+".branch2.0");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 96, DimsHW{ 3, 3 }, 1, DimsHW{ 1, 1 }, lname+".branch2.1");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 96, DimsHW{ 3, 3 }, 1, DimsHW{ 1, 1 }, lname+".branch2.2");

        // branch 3
        IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{3, 3});
        assert(pool1);
        pool1->setStrideNd(DimsHW{1, 1});
        pool1->setPaddingNd(DimsHW{1, 1});
        pool1->setAverageCountExcludesPadding(false);
        IActivationLayer* relu3 = basicConv2d(network, weightMap, *pool1 -> getOutput(0), 96, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname+".branch3.1");

        // concatenate all branches outputs
        ITensor* inputTensors[] = { relu0 -> getOutput(0), relu1 -> getOutput(0), relu2 -> getOutput(0), relu3 -> getOutput(0)};
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 4);
        assert(cat);
        return cat;

    }

    IConcatenationLayer* reductionA(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        // features 10 branch 0
        IActivationLayer* relu0 = basicConv2d(network, weightMap, input, 384, DimsHW{ 3, 3 }, 2, DimsHW{ 0, 0 }, lname + ".branch0");

        // branch 1
        IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 192, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch1.0");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 224, DimsHW{ 3, 3 }, 1, DimsHW{ 1, 1 }, lname + ".branch1.1");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 256, DimsHW{ 3, 3 }, 2, DimsHW{ 0, 0 }, lname + ".branch1.2");

        // branch 2
        IPoolingLayer* pool1 = network -> addPoolingNd(input, PoolingType::kMAX, DimsHW{ 3, 3 });
        assert(pool1);
        pool1 -> setStrideNd(DimsHW{ 2, 2 });

        // concatenate
        ITensor* inputTensors[] = { relu0 -> getOutput(0), relu1 -> getOutput(0), pool1 -> getOutput(0) };
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 3);
        assert(cat);
        return cat;
    }

    IConcatenationLayer* inceptionB(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        // features 11 branch 0
        IActivationLayer* relu0 = basicConv2d(network, weightMap, input, 384, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch0");

        // branch 1
        IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 192, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch1.0");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 224, DimsHW{ 1, 7 }, 1, DimsHW{ 0, 3 }, lname + ".branch1.1");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 256, DimsHW{ 7, 1 }, 1, DimsHW{ 3, 0 }, lname + ".branch1.2");
        
        // branch 2
        IActivationLayer* relu2 = basicConv2d(network, weightMap, input, 192, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch2.0");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 192, DimsHW{ 7, 1 }, 1, DimsHW{ 3, 0 }, lname + ".branch2.1");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 224, DimsHW{ 1, 7 }, 1, DimsHW{ 0, 3 }, lname + ".branch2.2");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 224, DimsHW{ 7, 1 }, 1, DimsHW{ 3, 0 }, lname + ".branch2.3");
        relu2 = basicConv2d(network, weightMap, *relu2 -> getOutput(0), 256, DimsHW{ 1, 7 }, 1, DimsHW{ 0, 3 }, lname + ".branch2.4");

        // branch 3
        IPoolingLayer* pool0 = network -> addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 3, 3 });
        assert(pool0);
        pool0 -> setStrideNd(DimsHW{ 1, 1 });
        pool0 -> setPaddingNd(DimsHW{ 1, 1 });
        pool0 -> setAverageCountExcludesPadding(false);
        IActivationLayer* relu3 = basicConv2d(network, weightMap, *pool0 -> getOutput(0), 128, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch3.1");

        // concatenate branches
        ITensor* inputTensors[] = { relu0 -> getOutput(0), relu1 -> getOutput(0), relu2 -> getOutput(0), relu3 -> getOutput(0) };
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 4);
        assert(cat);

        return cat;
    }

    IConcatenationLayer* reductionB(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {
        // features 18 branch 0
        IActivationLayer* relu0 = basicConv2d(network, weightMap, input, 192, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch0.0");
        relu0 = basicConv2d(network, weightMap, *relu0 -> getOutput(0), 192, DimsHW{ 3, 3 }, 2, DimsHW{ 0, 0 }, lname + ".branch0.1");

        // branch 1
        IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 256, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch1.0");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 256, DimsHW{ 1, 7 }, 1, DimsHW{ 0, 3 }, lname + ".branch1.1");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 320, DimsHW{ 7, 1 }, 1, DimsHW{ 3, 0 }, lname + ".branch1.2");
        relu1 = basicConv2d(network, weightMap, *relu1 -> getOutput(0), 320, DimsHW{ 3, 3 }, 2, DimsHW{ 0, 0 }, lname + ".branch1.3");

        // branch 2
        IPoolingLayer* pool1 = network -> addPoolingNd(input, PoolingType::kMAX, DimsHW{ 3, 3 });
        assert(pool1);
        pool1 -> setStrideNd(DimsHW{ 2, 2 });

        // concatenate
        ITensor* inputTensors[] = { relu0 -> getOutput(0), relu1 -> getOutput(0), pool1 -> getOutput(0) };
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 3);
        assert(cat);

        return cat;
    }

    IConcatenationLayer* inceptionC(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    )
    {

        // features 19 branch 0
        IActivationLayer* relu0 = basicConv2d(network, weightMap, input, 256, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch0");

        // branch 1
        IActivationLayer* relu1_0 = basicConv2d(network, weightMap, input, 384, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch1_0");
        IActivationLayer* relu1_1a = basicConv2d(network, weightMap, *relu1_0 -> getOutput(0), 256, DimsHW{ 1, 3 }, 1, DimsHW{ 0, 1 }, lname + ".branch1_1a");
        IActivationLayer* relu1_1b = basicConv2d(network, weightMap, *relu1_0 -> getOutput(0), 256, DimsHW{ 3, 1 }, 1, DimsHW{ 1, 0 }, lname + ".branch1_1b");
        ITensor* inputTensors1[] = { relu1_1a -> getOutput(0), relu1_1b -> getOutput(0) };
        IConcatenationLayer* cat1 = network -> addConcatenation(inputTensors1, 2);
        assert(cat1);

        // branch 2
        IActivationLayer* relu2_0 = basicConv2d(network, weightMap, input, 384, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch2_0");
        IActivationLayer* relu2_1 = basicConv2d(network, weightMap, *relu2_0 -> getOutput(0), 448, DimsHW{ 3, 1 }, 1, DimsHW{ 1, 0 }, lname + ".branch2_1");
        IActivationLayer* relu2_2 = basicConv2d(network, weightMap, *relu2_1 -> getOutput(0), 512, DimsHW{ 1, 3 }, 1, DimsHW{ 0, 1 }, lname + ".branch2_2");
        IActivationLayer* relu2_3a = basicConv2d(network, weightMap, *relu2_2 -> getOutput(0), 256, DimsHW{ 1, 3 }, 1, DimsHW{ 0, 1 }, lname + ".branch2_3a");
        IActivationLayer* relu2_3b = basicConv2d(network, weightMap, *relu2_2 -> getOutput(0), 256, DimsHW{ 3, 1 }, 1, DimsHW{ 1, 0 }, lname + ".branch2_3b");
        ITensor* inputTensors2[] = { relu2_3a -> getOutput(0), relu2_3b -> getOutput(0) };
        IConcatenationLayer* cat2 = network -> addConcatenation(inputTensors2, 2);
        assert(cat2);

        // branch 3
        IPoolingLayer* pool3 = network -> addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 3, 3 });
        assert(pool3);
        pool3 -> setStrideNd(DimsHW{ 1, 1 });
        pool3 -> setPaddingNd(DimsHW{ 1, 1 });
        pool3 -> setAverageCountExcludesPadding(false);
        IActivationLayer* relu3 = basicConv2d(network, weightMap, *pool3 -> getOutput(0), 256, DimsHW{ 1, 1 }, 1, DimsHW{ 0, 0 }, lname + ".branch3.1");

        // concatenate
        ITensor* inputTensors[] = { relu0 -> getOutput(0), cat1 -> getOutput(0), cat2 -> getOutput(0), relu3 -> getOutput(0) };
        IConcatenationLayer* cat = network -> addConcatenation(inputTensors, 4);
        assert(cat);
        return cat;
    }
}