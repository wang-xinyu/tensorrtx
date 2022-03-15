#include "common.hpp"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>

#define USE_FP32

static Logger gLogger;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "output";
static const int bs = 1;
static const int channels = 96;
static const int ch = 3;
static const int INPUT_H = 576;
static const int INPUT_W = 576;
static const int NUM_CLASSES = 15;
static const int outputSize = 576 * 576;
cudaStream_t m_cudaStream;
vector<void *> m_bindings;
IExecutionContext *m_context;

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt,std::string wtsPath)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ch, INPUT_H, INPUT_W});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    ITensor* conv1 = conv(network, weightMap, data, "backbone.patch_embed.proj", channels);
    ITensor* shuffle1 = shuffle_reshapeApermute(network, conv1, Dims2{channels, -1}, Permutation{1, 0}, true);
    ITensor *ln = m_layerNorm(network, weightMap, shuffle1, "backbone.patch_embed.norm");
    debug_print(ln, "ln");
    //layer0

    ITensor *mask0 = trt_transform_imgMask(network, 147, 7, 3);
    ITensor *blk00 = blk(network, weightMap, ln, mask0, "backbone.layers.0.blocks.0", INPUT_H / 4, channels, 3, 7, 0);
    debug_print(blk00, "blk00");
    ITensor *blk01 = blk(network, weightMap, blk00, mask0, "backbone.layers.0.blocks.1", INPUT_H / 4, channels, 3, 7, 3);
    debug_print(blk01, "blk01");
    ITensor* out0 = m_layerNorm(network, weightMap, blk01, "backbone.norm0");
    out0 = shuffle_reshapeApermute(network, out0, Dims3{INPUT_H / 4, INPUT_H / 4, channels}, Permutation{2, 0, 1}, true);
    ITensor *down_layer0 = downsample(network, weightMap, blk01, "backbone.layers.0.downsample", INPUT_H / 4);
    debug_print(down_layer0, "down_blk1");
    //layer1
    ITensor *mask1 = trt_transform_imgMask(network, 77, 7, 3);
    ITensor *blk10 = blk(network, weightMap, down_layer0, mask1, "backbone.layers.1.blocks.0", INPUT_H / 8, channels * 2, 6, 7, 0);
    debug_print(blk10, "blk10");
    ITensor *blk11 = blk(network, weightMap, blk10, mask1, "backbone.layers.1.blocks.1", INPUT_H / 8, channels * 2, 6, 7, 3);
    debug_print(blk11, "blk11");
    ITensor* out1 = m_layerNorm(network, weightMap, blk11, "backbone.norm1");
    out1 = shuffle_reshapeApermute(network, out1, Dims3{INPUT_H / 8, INPUT_H / 8, channels * 2}, Permutation{2, 0, 1}, true);
    ITensor *down_layer1 = downsample(network, weightMap, blk11, "backbone.layers.1.downsample", INPUT_H / 8);
    debug_print(down_layer1, "down_layer1");
    //layer2
    ITensor *mask2 = trt_transform_imgMask(network, 42, 7, 3);
    ITensor *blk20 = blk(network, weightMap, down_layer1, mask2, "backbone.layers.2.blocks.0", INPUT_H / 16, channels * 4, 12, 7, 0);
    debug_print(blk20, "blk20");
    ITensor *blk21 = blk(network, weightMap, blk20, mask2, "backbone.layers.2.blocks.1", INPUT_H / 16, channels * 4, 12, 7, 3);
    debug_print(blk21, "blk21");
    ITensor *blk22 = blk(network, weightMap, blk21, mask2, "backbone.layers.2.blocks.2", INPUT_H / 16,channels * 4, 12, 7, 0);
    debug_print(blk22, "blk22");
    ITensor *blk23 = blk(network, weightMap, blk22, mask2, "backbone.layers.2.blocks.3", INPUT_H / 16, channels * 4, 12, 7, 3);
    debug_print(blk23, "blk23");
    ITensor *blk24 = blk(network, weightMap, blk23, mask2, "backbone.layers.2.blocks.4", INPUT_H / 16, channels * 4, 12, 7, 0);
    debug_print(blk24, "blk24");
    ITensor *blk25 = blk(network, weightMap, blk24, mask2, "backbone.layers.2.blocks.5", INPUT_H / 16, channels * 4, 12, 7, 3);
    debug_print(blk25, "blk25");
    ITensor* out2 = m_layerNorm(network, weightMap, blk25, "backbone.norm2");
    out2 = shuffle_reshapeApermute(network, out2, Dims3{INPUT_H / 16, INPUT_H / 16, channels * 4}, Permutation{2, 0, 1}, true);
    ITensor *down_layer2 = downsample(network, weightMap, blk25, "backbone.layers.2.downsample", INPUT_H / 16);
    debug_print(down_layer2, "down_layer2");
    //layer3
    ITensor *mask3 = trt_transform_imgMask(network, 21, 7, 3);
    ITensor *blk30 = blk(network, weightMap, down_layer2, mask3, "backbone.layers.3.blocks.0", INPUT_H / 32, channels * 8, 24, 7, 0);
    debug_print(blk30, "blk30");
    ITensor *blk31 = blk(network, weightMap, blk30, mask3, "backbone.layers.3.blocks.1", INPUT_H / 32, channels * 8, 24, 7, 3);
    debug_print(blk31, "blk31");
    ITensor* out3 = m_layerNorm(network, weightMap, blk31, "backbone.norm3");
    out3 = shuffle_reshapeApermute(network, out3, Dims3{INPUT_H / 32, INPUT_H / 32, channels * 8}, Permutation{2, 0, 1}, true);
    ITensor* out[4] = {out0, out1, out2, out3};
    out0 = transform_lateral_conv(network, weightMap, out0, "decode_head.lateral_convs.0");  // 512,INPUT_H/4,INPUT_H/4
    out1 = transform_lateral_conv(network, weightMap, out1, "decode_head.lateral_convs.1");  // 512,INPUT_H/8,INPUT_H/8
    out2 = transform_lateral_conv(network, weightMap, out2, "decode_head.lateral_convs.2");  // 512,INPUT_H/16,INPUT_H/16
    auto psp_out_0 = transform_psp(network, weightMap, out3, "decode_head.psp_modules.0.1", 1);
    auto psp_out_1 = transform_psp(network, weightMap, out3, "decode_head.psp_modules.1.1", 2);
    auto psp_out_2 = transform_psp(network, weightMap, out3, "decode_head.psp_modules.2.1", 3);
    auto psp_out_3 = transform_psp(network, weightMap, out3, "decode_head.psp_modules.3.1", 6);
    ITensor* psp_outs[5] = {out3, psp_out_0, psp_out_1, psp_out_2, psp_out_3};
    auto PSP_outs = network->addConcatenation(psp_outs, 5);
    PSP_outs->setAxis(0);
    debug_print(PSP_outs->getOutput(0), "PSP_outs");
    out3 = transform_lateral_conv(network, weightMap, PSP_outs->getOutput(0), "decode_head.bottleneck", 3, 1, 512);  // 512,INPUT_H/32,INPUT_H/32
    debug_print(out3, "out3");
    auto laterals2 = up_Add(network, out3, out2);
    auto laterals1 = up_Add(network, laterals2, out1);
    auto laterals0 = up_Add(network, laterals1, out0);
    auto fpn0 = transform_lateral_conv(network, weightMap, laterals0, "decode_head.fpn_convs.0", 3, 1, 512);
    auto fpn1 = transform_lateral_conv(network, weightMap, laterals1, "decode_head.fpn_convs.1", 3, 1, 512);
    auto fpn2 = transform_lateral_conv(network, weightMap, laterals2, "decode_head.fpn_convs.2", 3, 1, 512);
    fpn1 = resize(network, fpn1,fpn0->getDimensions().d[1]);
    fpn2 = resize(network, fpn2,fpn0->getDimensions().d[1]);
    auto fpn3 = resize(network, out3, fpn0->getDimensions().d[1]);
    ITensor* fpn_outs[4] = {fpn0, fpn1, fpn2, fpn3};
    auto FPN_outs = network->addConcatenation(fpn_outs, 4);
    FPN_outs->setAxis(0);
    debug_print(FPN_outs->getOutput(0), "FPN_outs");
    auto fpn_output = transform_lateral_conv(network, weightMap, FPN_outs->getOutput(0), "decode_head.fpn_bottleneck", 3, 1, 512);
    debug_print(fpn_output, "fpn_output");
    auto seg = network->addConvolutionNd(*fpn_output, NUM_CLASSES, Dims2{1, 1}, weightMap["decode_head.conv_seg.weight"], weightMap["decode_head.conv_seg.bias"]);
    seg->setStrideNd(Dims2{1, 1});
    debug_print(seg->getOutput(0), "seg");
    auto seg_resize = resize(network, seg->getOutput(0), INPUT_H);
    debug_print(seg_resize, "seg_resize");
    auto output = network->addTopK(*seg_resize, TopKOperation::kMAX, 1, 0X01)->getOutput(1);
    debug_print(output, "output");

    std::cout << "set name out" << std::endl;
    output->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*output);
    builder->setMaxBatchSize(12);
    config->setMaxWorkspaceSize((1 << 30)); // 1G
#ifdef USE_FP16
    std::cout<< "use fp16"<<std::endl;
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build success!" << std::endl;
    network->destroy();

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream,std::string wtsPath)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wtsPath);
    assert(engine != nullptr);
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void createEng(std::string wtsPath, std::string engine_name)
{
    char *trtModelStream{nullptr};
    size_t size{0};

    IHostMemory *modelStream{nullptr};
    APIToModel(bs, &modelStream, wtsPath);
    assert(modelStream != nullptr);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
}

void inference_init(string ENGPath,ICudaEngine *m_engine)
{
    ifstream cache(ENGPath, ios::binary);
    cache.seekg(0, ios::end);
    const int engSize = cache.tellg();
    cache.seekg(0, ios::beg);
    void *modelMem = malloc(engSize);
    cache.read((char*)modelMem, engSize);
    cache.close();
    IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    m_engine = runtime->deserializeCudaEngine(modelMem, engSize);
    runtime->destroy();
    free(modelMem);
    if (!m_engine) {
        cout << "deserialize eng error!" << endl;
        return;
    }
    m_context = m_engine->createExecutionContext();
    if (cudaStreamCreate(&m_cudaStream) != 0) return;
    int bindings = m_engine->getNbBindings();
    if (bindings < 2)
    {
        cout << "Error! the network have one input and one output at least!" << endl;
        return;
    }
    cout << "1111111111111" << endl;
    m_bindings.resize(bindings, nullptr);
    CHECK(cudaMalloc(&m_bindings.at(0), bs * ch * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&m_bindings.at(1), bs * outputSize * 4));
}

void doInference(const float *input, int *output)
{
    cout << "do infer:" << endl;
    CHECK(cudaMemcpyAsync(m_bindings.at(0), input, bs * ch * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, m_cudaStream));

    m_context->enqueue(bs, m_bindings.data(), m_cudaStream, nullptr);

    CHECK(cudaMemcpyAsync(output, m_bindings.at(1), bs * outputSize * 4,
                          cudaMemcpyDeviceToHost, m_cudaStream));

    cudaStreamSynchronize(m_cudaStream);
}


int main(int argc, char** argv)
{
    cout << "begin" << endl;
    //string wts = "G:/shaj/trainsform/ktn5n6_29.511.21.8.wts";
    //string eng = "G:/shaj/trainsform/trainsform.eng";
    if (argv[1] = "-s") {
        string wts = argv[2];
        string eng = argv[3];
        createEng(wts,eng);
    } else {
        string eng = argv[2];

        ICudaEngine *m_engine;

        inference_init(eng,m_engine);

        vector<cv::Mat> testVal;
        map<string,cv::Mat> dataProb;
        vector<string> imgs;
        cv::Mat img;
        //string pattern_dir = "G:/shaj/trainsform";
        string pattern_dir = argv[3];
        string pattern = pattern_dir+ "/*.bmp";
        vector<cv::String> images_names;
        cv::glob(pattern, images_names, false);
        int i = 0;
        cv::Scalar Mean = cv::Scalar(123.675, 116.28, 103.53);
        cv::Scalar Std = cv::Scalar(58.395, 57.12, 57.375);
        cv::Size size = {INPUT_H,INPUT_W};

        for (auto image_name: images_names)
        {
            if (i < bs)
            {
                cv::Mat Img = cv::imread(image_name, 1);

                testVal.push_back(Img);
                cout << image_name << endl;
                imgs.push_back(image_name);
            }
        }
        float *data = new float[bs * ch * INPUT_H * INPUT_W];
        int *output = new int[bs * outputSize];

        cv::Mat Transed_t = BlobFromImages(testVal, cv::Size{INPUT_H, INPUT_W}, Mean, Std, true, false);

        memcpy(data, Transed_t.data, bs * ch * INPUT_H * INPUT_W * sizeof(float));

        //for(int i = 0 ; i< 20; i++){

        auto start_time = std::chrono::system_clock::now();
        doInference(data, output);

        auto end_time = std::chrono::system_clock::now();
        float duration;
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        cout << "time:" << duration << endl;
        //}
        //    for(int i = 0; i < 100; i++)
        //        cout<<i<<":"<<output[i]<<endl;

        int n = 0;
        int *out = new int[outputSize];
        string outPath = pattern_dir + "/output";
        for (int i = 0; i < testVal.size(); i++)
        {
            cv::Mat img = cv::imread(imgs[i], 1);
            cv::Mat dst;
            cv::resize(img,dst,cv::Size{INPUT_H, INPUT_W});
            //string outPath_n = outPath + "/"+to_string(n) + ".jpg";
            n += 1;
            out = output + i * outputSize;
            for (int i = 0; i < outputSize; i++)
            {
                if (out[i] != 0)
                {
                    int w = i % (INPUT_H);
                    int h = i / (INPUT_W);
                    dst.at<cv::Vec3b>(h, w)[0] =  out[i] * 10;
                    dst.at<cv::Vec3b>(h, w)[1] =  out[i] * 30;
                    dst.at<cv::Vec3b>(h, w)[2] =  out[i] * 40;
                }
            }
            //cout<<outPath_n<<endl;
            string outPath_result = imgs[i].replace(0, pattern_dir.size(), outPath);
            cout << outPath_result << endl;
            cv::imwrite(outPath_result, dst);
        }
        testVal.clear();
        imgs.clear();
    }

    m_context->destroy();
    m_engine->destroy();
    for (auto bindings: m_bindings) {
        cudaFree(bindings);
    }
    cudaFree(m_cudaStream);

    cout << "swin_transform" << endl;
    return 0;
}

