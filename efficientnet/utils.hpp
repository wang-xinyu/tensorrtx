#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <math.h>
#include <string>
#include <algorithm>
using namespace nvinfer1;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

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

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
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

struct BlockArgs
{
    int num_repeat;
    int kernel_size;
    int stride;
    float expand_ratio;
    int input_filters;
    int output_filters;
    float se_ratio;
    bool id_skip;
};

struct GlobalParams
{
    int input_h;
    int input_w;
    int num_classes;
    float batch_norm_epsilon;
    float width_coefficient;
    float depth_coefficient;
    int depth_divisor;
    int min_depth;
};

int roundFilters(int filters, GlobalParams global_params)
{
    float multiplier = global_params.width_coefficient;
    int divisor = global_params.depth_divisor;
    int min_depth = global_params.min_depth;
    filters = int(filters * multiplier);
    if (min_depth < 0)
    {
        min_depth = divisor;
    }
    // follow the formula transferred from official TensorFlow implementation
    int new_filters = std::max(min_depth, int(int(filters + divisor / 2) / divisor) * divisor);
    if (new_filters < 0.9 * filters) // prevent rounding by more than 10%
        new_filters += divisor;
    return int(new_filters);
}

DimsHW calculateOutputImageSize(DimsHW image_size, int stride)
{
    int image_h = int(ceil(float(image_size.h()) / float(stride)));
    int image_w = int(ceil(float(image_size.w()) / float(stride)));
    return DimsHW{image_h, image_w};
}

int roundRepeats(int repeats, GlobalParams global_params)
{
    float multiplier = global_params.depth_coefficient;
    // follow the formula transferred from official TensorFlow implementation
    int new_repeats = int(ceil(multiplier * repeats));
    return new_repeats;
}

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
    float *gamma = (float *)weightMap[lname + ".weight"].values;
    float *beta = (float *)weightMap[lname + ".bias"].values;
    float *mean = (float *)weightMap[lname + ".running_mean"].values;
    float *var = (float *)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IConvolutionLayer *addSamePaddingConv2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int kernel_size, int stride, int dilation, int groups, DimsHW image_size, std::string lname, bool bias = true)
{
    int ih = image_size.h();
    int iw = image_size.w();
    int kh = kernel_size;
    int kw = kernel_size;
    int sh = stride;
    int sw = stride;
    int oh = ceil(float(ih) / float(sh));
    int ow = ceil(float(iw) / float(sw));
    int pad_h = std::max((oh - 1) * stride + (kh - 1) * dilation + 1 - ih, 0);
    int pad_w = std::max((ow - 1) * stride + (kw - 1) * dilation + 1 - iw, 0);
    int pad_left = 0;
    int pad_right = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    if (pad_h > 0 || pad_w > 0)
    {
        pad_left = int(pad_w / 2);
        pad_right = pad_w - int(pad_w / 2);
        pad_top = int(pad_h / 2);
        pad_bottom = pad_h - int(pad_h / 2);
    }
    Weights bias_wt{DataType::kFLOAT, nullptr, 0};
    if (bias)
    {
        bias_wt = weightMap[lname + ".bias"];
    }
    IConvolutionLayer *conv = network->addConvolutionNd(input, outch, DimsHW{kh, kw}, weightMap[lname + ".weight"], bias_wt);
    conv->setPrePadding(DimsHW{pad_top, pad_left});
    conv->setPostPadding(DimsHW{pad_bottom, pad_right});
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setDilationNd(DimsHW{dilation, dilation});
    conv->setNbGroups(groups);
    return conv;
}

ILayer *addSwish(INetworkDefinition *network, ITensor &input)
{
    //swish
    auto *sigmoid = network->addActivation(input, ActivationType::kSIGMOID);
    auto *ew = network->addElementWise(input, *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    return ew;
}

ITensor *MBConvBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, BlockArgs block_args, GlobalParams global_params, DimsHW image_size)
{
    bool has_se = block_args.se_ratio > 0 && block_args.se_ratio <= 1;
    bool id_skip = block_args.id_skip;
    float bn_eps = global_params.batch_norm_epsilon;
    int input_filters = block_args.input_filters;
    int output_filters = block_args.output_filters;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    ITensor *x = &input;
    int inp = block_args.input_filters;
    int oup = int(block_args.input_filters * block_args.expand_ratio);
    // expand_ratio != 1
    if (fabs(block_args.expand_ratio - 1) > 1e-5)
    {
        auto expand_conv = addSamePaddingConv2d(network, weightMap, input, oup, 1, 1, 1, 1, image_size, lname + "._expand_conv");
        auto bn0 = addBatchNorm2d(network, weightMap, *expand_conv->getOutput(0), lname + "._bn0", bn_eps);
        auto swish0 = addSwish(network, *bn0->getOutput(0));
        x = swish0->getOutput(0);
    }
    int k = block_args.kernel_size;
    int s = block_args.stride;
    auto depthwise_conv = addSamePaddingConv2d(network, weightMap, *x, oup, k, s, 1, oup, image_size, lname + "._depthwise_conv", false);
    auto bn1 = addBatchNorm2d(network, weightMap, *depthwise_conv->getOutput(0), lname + "._bn1", bn_eps);
    //swish
    auto swish1 = addSwish(network, *bn1->getOutput(0));
    x = swish1->getOutput(0);
    image_size = calculateOutputImageSize(image_size, s);
    if (has_se)
    {
        auto avg_pool = network->addPoolingNd(*x, PoolingType::kAVERAGE, image_size);
        int num_squeezed_channels = std::max(1, int(input_filters * block_args.se_ratio));
        auto se_reduce = addSamePaddingConv2d(network, weightMap, *avg_pool->getOutput(0), num_squeezed_channels, 1, 1, 1, 1, DimsHW{1, 1}, lname + "._se_reduce");

        auto swish2 = addSwish(network, *se_reduce->getOutput(0));
        auto se_expand = addSamePaddingConv2d(network, weightMap, *swish2->getOutput(0), oup, 1, 1, 1, 1, DimsHW{1, 1}, lname + "._se_expand");

        auto *sigmoid = network->addActivation(*se_expand->getOutput(0), ActivationType::kSIGMOID);
        auto *ew = network->addElementWise(*x, *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
        x = ew->getOutput(0);
    }
    int final_oup = block_args.output_filters;
    auto project_conv = addSamePaddingConv2d(network, weightMap, *x, final_oup, 1, 1, 1, 1, image_size, lname + "._project_conv");

    auto bn2 = addBatchNorm2d(network, weightMap, *project_conv->getOutput(0), lname + "._bn2", bn_eps);
    x = bn2->getOutput(0);

    if (id_skip && block_args.stride == 1 && input_filters == output_filters)
    {
        auto *ew = network->addElementWise(input, *x, ElementWiseOperation::kSUM);
        x = ew->getOutput(0);
    }
    return x;
}
