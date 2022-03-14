#ifndef COMMON_HPP
#define COMMON_HPP

#include "layerNorm.h"
#include "NvInfer.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include <assert.h>
#include <map>
#include <fstream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<opencv2/dnn/dnn.hpp>

using namespace nvinfer1;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

void mblobFromImages(cv::InputArrayOfArrays images_, cv::OutputArray blob_,
    cv::Size size, const cv::Scalar& mean_, const cv::Scalar& std_, bool swapRB, bool crop)
{
    //CV_TRACE_FUNCTION();
    std::vector<cv::Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    for (int i = 0; i < images.size(); i++)
    {
        cv::Size imgSize = images[i].size();
        if (size == cv::Size())
            size = imgSize;
        if (size != imgSize)
        {
            if (crop)
            {
                float resizeFactor = std::max(size.width / (float)imgSize.width,
                    size.height / (float)imgSize.height);
                resize(images[i], images[i], cv::Size(), resizeFactor, resizeFactor, cv::INTER_LINEAR);
                cv::Rect crop(cv::Point(0.5 * (images[i].cols - size.width),
                    0.5 * (images[i].rows - size.height)),
                    size);
                images[i] = images[i](crop);
            }
            else
                resize(images[i], images[i], size, 0, 0, cv::INTER_LINEAR);
        }
        if (images[i].depth() == CV_8U)
            images[i].convertTo(images[i], CV_32F);
        cv::Scalar mean = mean_;
        cv::Scalar std_num = std_;
        if (swapRB)
        {
            std::swap(mean[0], mean[2]);
            std::swap(std_num[0], std_num[2]);
        }

        images[i] -= mean;
        images[i] /= std_num;
    }

    size_t i, nimages = images.size();
    cv::Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    cv::Mat image;
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob_.create(4, sz, CV_32F);
        cv::Mat blob = blob_.getMat();
        cv::Mat ch[4];

        for (i = 0; i < nimages; i++)
        {
            image = images[i];
            CV_Assert(image.depth() == CV_32F);
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for (int j = 0; j < nch; j++)
                ch[j] = cv::Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, j));
            if (swapRB)
                std::swap(ch[0], ch[2]);
            split(image, ch);
        }
    }
    else
    {
        CV_Assert(nch == 1);
        int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
        blob_.create(4, sz, CV_32F);
        cv::Mat blob = blob_.getMat();

        for (i = 0; i < nimages; i++)
        {
            cv::Mat image = images[i];
            CV_Assert(image.depth() == CV_32F);
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 1));
            CV_Assert(image.size() == image0.size());

            image.copyTo(cv::Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, 0)));
        }
    }
}
cv::Mat BlobFromImages(cv::InputArrayOfArrays images, cv::Size size,
    const cv::Scalar& mean, const cv::Scalar& std_num, bool swapRB, bool crop)
{
    //CV_TRACE_FUNCTION();
    cv::Mat blob;
    mblobFromImages(images, blob, size, mean, std_num, swapRB, crop);
    return blob;
}
void debug_print(ITensor *input_tensor,std::string head)
{
    std::cout << head<< " : ";

       for (int i = 0; i < input_tensor->getDimensions().nbDims; i++)
       {
           std::cout << input_tensor->getDimensions().d[i] << " ";
       }
       std::cout<<std::endl;

}
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

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
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

ITensor* m_layerNorm(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,ITensor *input, string lname)
{
    auto creator = getPluginRegistry()->getPluginCreator("layerNorm_trt","1");

    PluginField pluginMultidata[2];

    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin(lname.c_str(), pluginData);
    ITensor* inputTensors[] = {input};
    auto ln_ms = m_Network->addPluginV2(inputTensors, 1, *pluginObj);
    auto ln_m = m_Network->addElementWise(*input,*ln_ms->getOutput(0),ElementWiseOperation::kSUB);
    auto ln = m_Network->addElementWise(*ln_m->getOutput(0),*ln_ms->getOutput(1),ElementWiseOperation::kDIV);
    Weights W = weightMap[lname + ".weight"];
    int len = W.count;
    Dims wb ;
    wb.nbDims = ln->getOutput(0)->getDimensions().nbDims;
    for (int i = 0 ; i < wb.nbDims; i++)
    {
        if (i != wb.nbDims -1)
            wb.d[i] = 1;
        else{
            wb.d[i] = len;
        }
    }
    auto wgts = m_Network->addConstant(wb,W);
    auto p_w = m_Network->addElementWise(*ln->getOutput(0),*wgts->getOutput(0),ElementWiseOperation::kPROD);
    Weights B = weightMap[lname + ".bias"];
    auto bias = m_Network->addConstant(wb,B);
    auto sum_bias = m_Network->addElementWise(*p_w->getOutput(0),*bias->getOutput(0),ElementWiseOperation::kSUM);
    debug_print(sum_bias->getOutput(0),lname);
    return sum_bias->getOutput(0);
}
ITensor* layerNorm(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,ITensor *input, string lname)
{
    auto mean = m_Network->addReduce(*input, ReduceOperation::kAVG, 2, true);
    assert(mean);

    auto sub_mean = m_Network->addElementWise(*input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(sub_mean);
//    float SCALING_ONE = 1.0;
//    float SHIFT_ZERO = 0.0;
//    float POWER_TWO = 2.0;
//    // implement pow2 with scale
//    Weights scale{ DataType::kFLOAT, &SCALING_ONE, 1 };
//    Weights shift{ DataType::kFLOAT, &SHIFT_ZERO, 1 };
//    Weights power{ DataType::kFLOAT, &POWER_TWO, 1 };
//    auto pow2 = m_Network->addScaleNd(*sub_mean->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power,0);
//    assert(pow2);
    auto pow2 = m_Network->addElementWise(*sub_mean->getOutput(0), *sub_mean->getOutput(0), ElementWiseOperation::kPROD);
    assert(pow2);
    debug_print(pow2->getOutput(0),"pow2");
    auto pow_mean = m_Network->addReduce(*pow2->getOutput(0), ReduceOperation::kAVG, 2, true);
    assert(pow_mean);
    debug_print(pow_mean->getOutput(0),"pow_mean");
    float E = 1e-5;
    Weights EPS{DataType::kFLOAT,nullptr,1};
    EPS.values = &E;
    auto eps = m_Network->addConstant(Dims2{1,1}, EPS);
    assert(eps);

    auto add_eps = m_Network->addElementWise(*pow_mean->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
    assert(add_eps);

    auto sqrt = m_Network->addUnary(*add_eps->getOutput(0), UnaryOperation::kSQRT);
    assert(sqrt);

    auto div = m_Network->addElementWise(*sub_mean->getOutput(0), *sqrt->getOutput(0), ElementWiseOperation::kDIV);
    assert(div);
    debug_print(div->getOutput(0),"div");

    string weightsFile = lname + ".weight";
    string biasFile = lname + ".bias";

    int d_model = input->getDimensions().d[input->getDimensions().nbDims - 1];
    cout<<"d_model = "<<d_model<<endl;
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * d_model));
    for (int i = 0; i < d_model; i++) {
        pval[i] = 1.0;
    }
    Weights norm1_power{ DataType::kFLOAT, pval, d_model };
    auto affine = m_Network->addScaleNd(
        *div->getOutput(0),
        ScaleMode::kELEMENTWISE,
        weightMap[biasFile],
        weightMap[weightsFile],
        norm1_power,1);
    assert(affine);
    return affine->getOutput(0);
}
ITensor* conv(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,ITensor *input, string lname,
              int c_out,bool bias = true,int k = 4 , int s = 4, int p = 0)
{
    Weights Bias{ DataType::kFLOAT, nullptr, 0 };
    if(bias)
        Bias = weightMap[lname + ".bias"];
    auto out = m_Network->addConvolutionNd(*input,c_out,Dims2{k,k},weightMap[lname + ".weight"],Bias);
    out->setStrideNd(Dims2{s,s});
    out->setPaddingNd(Dims2{p,p});
    out->setNbGroups(1);
    debug_print(out->getOutput(0),lname);
    return out->getOutput(0);
}
ITensor* shuffle_reshape(INetworkDefinition *m_Network,ITensor *input,Dims reshapeDims)
{
    auto out = m_Network->addShuffle(*input);
    out->setReshapeDimensions(reshapeDims);
    debug_print(out->getOutput(0),"reshape");
    return out->getOutput(0);
}
ITensor* shuffle_permute(INetworkDefinition *m_Network,ITensor *input,Permutation permutation)
{
    auto out = m_Network->addShuffle(*input);
    out->setFirstTranspose(permutation);
    debug_print(out->getOutput(0),"permute");
    return out->getOutput(0);
}
ITensor* shuffle_reshapeApermute(INetworkDefinition *m_Network,ITensor *input,Dims reshapeDims,
                                 Permutation permutation,bool firstReshape)
{
    auto out = m_Network->addShuffle(*input);
    out->setReshapeDimensions(reshapeDims);
    if(firstReshape)
        out->setSecondTranspose(permutation);
    else
        out->setFirstTranspose(permutation);
    debug_print(out->getOutput(0),"shuffle");
    return out->getOutput(0);
}
ITensor* trt_transform_imgMask(INetworkDefinition *m_Network,int hw, int window_size, int shift_size)
{
    int Hp = hw;
    int Wp = hw;
    Weights Mask_param{DataType::kFLOAT,nullptr,Hp*Wp};
    float *mask_param = new float[Hp*Wp];
    for(int i = 0; i < Hp ; i++)
    {
        for(int j = 0; j < Wp; j++)
        {
            if(i<Hp-window_size && j<Wp-window_size)
                mask_param[i*Wp + j] = 0.0;
            else if(i<Hp-window_size && j>=Wp-window_size && j < Wp-shift_size)
                mask_param[i*Wp + j] = 1.0;
            else if(i<Hp-window_size &&  j >= Wp-shift_size)
                mask_param[i*Wp + j] = 2.0;

            else if(i >= Hp-window_size && i < Hp-shift_size && j<Wp-window_size)
                mask_param[i*Wp + j] = 3.0;
            else if(i >= Hp-window_size && i < Hp-shift_size && j>=Wp-window_size && j < Wp-shift_size)
                mask_param[i*Wp + j] = 4.0;
            else if(i >= Hp-window_size && i < Hp-shift_size && j >= Wp-shift_size)
                mask_param[i*Wp + j] = 5.0;

            else if(i >=  Hp-shift_size && j<Wp-window_size)
                mask_param[i*Wp + j] = 6.0;
            else if(i >=  Hp-shift_size && j>=Wp-window_size && j < Wp-shift_size)
                mask_param[i*Wp + j] = 7.0;
            else if(i >=  Hp-shift_size && j >= Wp-shift_size)
                mask_param[i*Wp + j] = 8.0;
            else{
                cout<<" i && j not limit"<<endl;
                return nullptr;
            }
        }
    }
    Mask_param.values = mask_param;
    auto img_mask = m_Network->addConstant(Dims4{1,Hp,Wp,1},Mask_param);
    auto img_mask_shuffle = m_Network->addShuffle(*img_mask->getOutput(0));
    Dims shuffle1_dims;
    shuffle1_dims.nbDims = 6;
    int dims[] = {1,Hp/window_size,window_size,Wp/window_size,window_size,1};
    for(int i = 0 ; i < 6; i++)
        shuffle1_dims.d[i] = dims[i];
    img_mask_shuffle->setReshapeDimensions(shuffle1_dims);
    img_mask_shuffle->setSecondTranspose(Permutation{0,1,3,2,4,5});
    auto img_mask_shuffle2 = m_Network->addShuffle(*img_mask_shuffle->getOutput(0));
    img_mask_shuffle2->setReshapeDimensions(Dims3{-1,1,window_size*window_size});
    auto img_mask_shuffle3 = m_Network->addShuffle(*img_mask_shuffle->getOutput(0)) ;
    img_mask_shuffle3->setReshapeDimensions(Dims3{-1,window_size*window_size,1});
    auto atten_mask = m_Network->addElementWise(*img_mask_shuffle2->getOutput(0),*img_mask_shuffle3->getOutput(0),ElementWiseOperation::kSUB);

    auto creator = getPluginRegistry()->getPluginCreator("fillmaskLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("fillmask", pluginData);
    ITensor* inputTensors[] = {atten_mask->getOutput(0)};
    auto fillmask = m_Network->addPluginV2(inputTensors, 1, *pluginObj);

    debug_print(fillmask->getOutput(0),"imgMask");
    return fillmask->getOutput(0);
}
ITensor* trt_transform_pad(INetworkDefinition *m_Network,ITensor *input,int window_size)
{
    int h = input->getDimensions().d[0];
    int w = input->getDimensions().d[1];
    int c = input->getDimensions().d[2];
    int pad_h = (window_size - h%window_size)%window_size;
    int pad_w = (window_size - w%window_size)%window_size;

    ITensor* temp = input;
    if(pad_h != 0)
    {
        Weights pad1{DataType::kFLOAT,nullptr,pad_h*w*c};
        cout<<pad_h*w*c<<endl;
        float *p1 = new float[pad_h*w*c];
        for(int i = 0 ; i < pad_h*w*c; i++)
            p1[i] = 0.f;
        pad1.values = p1;
        auto Pad1 = m_Network->addConstant(Dims3{pad_h,w,c},pad1);
        ITensor *cat1[2] = {temp,Pad1->getOutput(0)};
        auto xp1 = m_Network->addConcatenation(cat1,2);
        xp1->setAxis(0);
        temp = xp1->getOutput(0);
    }
    if(pad_w != 0)
    {
        Weights pad2{DataType::kFLOAT,nullptr,pad_w*(h+pad_h)*c};
        cout<<pad_w*(h+pad_h)*c<<endl;
        float *p2 = new float[pad_w*(h+pad_h)*c];
        for(int i = 0 ; i < pad_w*(h+pad_h)*c; i++)
            p2[i] = 0.0f;
        pad2.values = p2;
        auto Pad2 = m_Network->addConstant(Dims3{(h+pad_h),pad_w,c},pad2);
        ITensor *cat2[] = {temp,Pad2->getOutput(0)};
        auto xp2 = m_Network->addConcatenation(cat2,2);
        xp2->setAxis(1);
        temp = xp2->getOutput(0);
    }
    debug_print(temp, "pad");
    return  temp;
}
ITensor* trt_swinRoll(INetworkDefinition *m_Network,ITensor *input,vector<int> shifts, vector<int> dims)
{
    int len = shifts.size();
    Dims input_dim = input->getDimensions();
    int nbdims = input_dim.nbDims;
    ITensor *temp = input;
    for(int i = 0 ; i < len; i++)
    {
        Dims start, size,stride;
        start.nbDims = nbdims;
        size.nbDims = nbdims;
        stride.nbDims = nbdims;
        if(shifts[i] > 0)
        {
            for(int j = 0 ; j < nbdims; j++)
            {
                if(j != (dims[i] -1 ))
                {
                    start.d[j] = 0;
                    size.d[j] = input_dim.d[j];
                    stride.d[j] = 1;
                }
                else{
                    start.d[j] = 0;
                    size.d[j] = input_dim.d[j] - shifts[i];
                    stride.d[j] = 1;
                }
            }

            auto cat1 = m_Network->addSlice(*temp,start,size,stride);

            for(int j = 0 ; j < nbdims; j++)
            {
                if(j != (dims[i] - 1))
                {
                    start.d[j] = 0;
                    size.d[j] = input_dim.d[j];
                    stride.d[j] = 1;
                }
                else{
                    start.d[j] = input_dim.d[j] - shifts[i];
                    size.d[j] = shifts[i];
                    stride.d[j] = 1;
                }
            }
            auto cat2 = m_Network->addSlice(*temp,start,size,stride);
            ITensor *cat[] ={cat2->getOutput(0),cat1->getOutput(0)};
            auto Cat = m_Network->addConcatenation(cat,2);
            Cat->setAxis(dims[i] - 1);
            temp = Cat->getOutput(0);
        }
        if(shifts[i] < 0)
        {
            for(int j = 0 ; j < nbdims; j++)
            {
                if(j != (dims[i] - 1))
                {
                    start.d[j] = 0;
                    size.d[j] = input_dim.d[j];
                    stride.d[j] = 1;
                }
                else{
                    start.d[j] = 0;
                    size.d[j] = abs(shifts[i]);
                    stride.d[j] = 1;
                }
            }
            auto cat1 = m_Network->addSlice(*temp,start,size,stride);
            debug_print(cat1->getOutput(0), "cat1 dims : ");
            for(int j = 0 ; j < nbdims; j++)
            {
                if(j != (dims[i] - 1))
                {
                    start.d[j] = 0;
                    size.d[j] = input_dim.d[j];
                    stride.d[j] = 1;
                }
                else{
                    start.d[j] =  abs(shifts[i]);
                    size.d[j] = input_dim.d[j] - abs(shifts[i]);
                    stride.d[j] = 1;
                }
            }
            auto cat2 = m_Network->addSlice(*temp,start,size,stride);
            debug_print(cat2->getOutput(0), "cat2 dims : ");
            ITensor *cat[] ={cat2->getOutput(0),cat1->getOutput(0)};
            auto Cat = m_Network->addConcatenation(cat,2);
            Cat->setAxis(dims[i] - 1);
            temp = Cat->getOutput(0);
        }
    }
    return temp;
}
ITensor* trt_transform_window_partition(INetworkDefinition *m_Network,ITensor *input,int window_size)
{
    auto shuffle1 = m_Network->addShuffle(*input);
    Dims shuffle1_dims;
    shuffle1_dims.nbDims = 5;
    int h = input->getDimensions().d[0];
    int w = input->getDimensions().d[1];
    int c = input->getDimensions().d[2];

    int dims[] = {h/window_size,window_size,w/window_size,window_size,c};
    for(int i = 0 ; i < shuffle1_dims.nbDims; i++)
        shuffle1_dims.d[i] = dims[i];
    shuffle1->setReshapeDimensions(shuffle1_dims);
    shuffle1->setSecondTranspose(Permutation{0,2,1,3,4});
    debug_print(shuffle1->getOutput(0)," shuffle1 dims : ");
    auto shuffle2 = m_Network->addShuffle(*shuffle1->getOutput(0));
    shuffle2->setReshapeDimensions(Dims3{-1,window_size*window_size,c});

    debug_print(shuffle2->getOutput(0), "window partition");
    return shuffle2->getOutput(0);
}
ITensor* trt_swinLinear(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,
                        ITensor *input, string lname, bool bias = true)
{
    int c = input->getDimensions().d[input->getDimensions().nbDims-1];
    string fc_wpath = lname + ".weight";
    Weights fcW = weightMap[fc_wpath];
    int len_fcw = fcW.count;
    if(len_fcw == 0)
    {
        cout<<"file is not open,please check it's path: "<<fc_wpath<<endl;
        assert(0);
    }
    Dims fcWdims;
    fcWdims.nbDims = input->getDimensions().nbDims;
    if(fcWdims.nbDims == 2)
    {
        fcWdims.d[0] = len_fcw/c;
        fcWdims.d[1] = c;
    }
    else {
        fcWdims.d[0] = 1;
        fcWdims.d[1] = len_fcw/c;
        fcWdims.d[2] = c;
    }
    auto fc_w_constant = m_Network->addConstant(fcWdims,fcW);
    auto fc_w_mm = m_Network->addMatrixMultiply(*input,MatrixOperation::kNONE,
                                                *fc_w_constant->getOutput(0),MatrixOperation::kTRANSPOSE);

    string fc_bpath = lname +".bias";
    Weights fcB = weightMap[fc_bpath];
    int len_fcb = fcB.count;
    if(!bias)
    {
        cout<<lname<<" bias is Null!"<<endl;
        debug_print(fc_w_mm->getOutput(0),lname);
        return fc_w_mm->getOutput(0);
    }
    Dims fcBdims;
    fcBdims.nbDims = input->getDimensions().nbDims;
    if(fcBdims.nbDims == 2)
    {
        fcBdims.d[0] = 1;
        fcBdims.d[1] = len_fcb;
    }
    else {
        fcBdims.d[0] = 1;
        fcBdims.d[1] = 1;
        fcBdims.d[2] = len_fcb;
    }
    auto fc_b_constant = m_Network->addConstant(fcBdims,fcB);
    auto fc = m_Network->addElementWise(*fc_w_mm->getOutput(0),*fc_b_constant->getOutput(0),ElementWiseOperation::kSUM);
    debug_print(fc->getOutput(0),lname);
    return fc->getOutput(0);
}
ITensor* trt_trainsform_WindowAttention(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,ITensor *input,
                                        ITensor* mask,string lname,int dim, int num_heads,int window_size, int shift_size)
{

    int b = input->getDimensions().d[0];
    int n = input->getDimensions().d[1];
    int c = input->getDimensions().d[2];

    auto qkv = trt_swinLinear(m_Network,weightMap,input,lname+".qkv");

    Dims qkv_dim;
    qkv_dim.nbDims = 5;
    int d[5] = {b,n,3,num_heads,c/num_heads};
    for(int i = 0; i < 5; i++)
        qkv_dim.d[i] = d[i];
    Permutation qkv_p;
    int p[5] = {2, 0, 3, 1, 4};
    for(int i = 0; i < 5; i++)
        qkv_p.order[i] = p[i];
    auto qkv_shuffle = shuffle_reshapeApermute(m_Network,qkv,qkv_dim,qkv_p,true);

    Dims qkvDims = qkv_shuffle->getDimensions();
    Dims qstart,kstart,vstart,sizes,stride;
    qstart.nbDims = 5;
    kstart.nbDims = 5;
    vstart.nbDims = 5;
    sizes.nbDims = 5;
    stride.nbDims = 5;
    for(int i = 0; i < 5; i++)
    {
        if(i == 0)
        {
            qstart.d[0] = 0;
            kstart.d[0] = 1;
            vstart.d[0] = 2;
            sizes.d[0] = 1;
            stride.d[0] =1;
        }
        else{
            qstart.d[i] = 0;
            kstart.d[i] = 0;
            vstart.d[i] = 0;
            sizes.d[i] = qkvDims.d[i];
            stride.d[i] =1;
        }
    }
    auto q = m_Network->addSlice(*qkv_shuffle,qstart,sizes,stride);
    auto k = m_Network->addSlice(*qkv_shuffle,kstart,sizes,stride);
    auto v = m_Network->addSlice(*qkv_shuffle,vstart,sizes,stride);

    // q * s
    int len = 1;
    Weights scale_w{DataType::kFLOAT,nullptr,len};
    float *scale = new float[len];
    for(int i = 0 ; i < len; i++)
        scale[i] = 1 / sqrt(dim/num_heads);
    scale_w.values = scale;
    Dims scale_dim;
    scale_dim.nbDims = 5;

    for(int i = 0 ; i < 5; i++)
        scale_dim.d[i] = 1;
    auto Scale = m_Network->addConstant(scale_dim,scale_w);
    auto qs = m_Network->addElementWise(*q->getOutput(0),*Scale->getOutput(0),ElementWiseOperation::kPROD);
    auto qs_ = m_Network->addShuffle(*qs->getOutput(0));
    qs_->setReshapeDimensions(Dims4{qkvDims.d[1],qkvDims.d[2],qkvDims.d[3],qkvDims.d[4]});
    auto k_ = m_Network->addShuffle(*k->getOutput(0));
    k_->setReshapeDimensions(Dims4{qkvDims.d[1],qkvDims.d[2],qkvDims.d[3],qkvDims.d[4]});
    auto attn = m_Network->addMatrixMultiply(*qs_->getOutput(0),MatrixOperation::kNONE,
                                             *k_->getOutput(0),MatrixOperation::kTRANSPOSE);
    auto relatbias = m_Network->addConstant(Dims2{(2*window_size -1)*(2*window_size -1),num_heads},weightMap[lname + ".relative_position_bias_table"]);
    Dims r_i_dims;
    r_i_dims.nbDims = 1;
    r_i_dims.d[0] = window_size*window_size * window_size*window_size;
    Weights index{DataType::kINT32,nullptr,r_i_dims.d[0]};
    int* idx = new int[r_i_dims.d[0]];
    for (int i = 0; i < r_i_dims.d[0]; i++) {
        idx[i] =(int)((float*)weightMap[lname+".relative_position_index"].values)[i];
    }
    //idx = (int*)weightMap[lname+".relative_position_index"].values;
    //cout<<"idx = "<<((float*)weightMap[lname+".relative_position_index"].values)[0]<<endl;
    index.values = idx;
    auto relatidx = m_Network->addConstant(r_i_dims,index);
    auto relat = m_Network->addGather(*relatbias->getOutput(0),*relatidx->getOutput(0),0);
    auto relat_view = shuffle_reshapeApermute(m_Network,relat->getOutput(0),
                                              Dims4{1,window_size*window_size,window_size*window_size,-1},
                                              Permutation{0,3,1,2},true);
    auto attn_rv = m_Network->addElementWise(*attn->getOutput(0),*relat_view,ElementWiseOperation::kSUM);
    ITensor *Attn_rv = attn_rv->getOutput(0);
    if (mask != nullptr)
    {
        Dims maskdims;
        maskdims.nbDims = mask->getDimensions().nbDims +1;
        maskdims.d[0] = mask->getDimensions().d[0];
        maskdims.d[1] = 1;
        for(int i = 2; i< maskdims.nbDims; i++)
        {
            maskdims.d[i] = mask->getDimensions().d[i-1];
        }
        auto maskshuffle = m_Network->addShuffle(*mask);
        maskshuffle->setReshapeDimensions(maskdims);
        auto attn_rnM = m_Network->addElementWise(*attn_rv->getOutput(0),*maskshuffle->getOutput(0),ElementWiseOperation::kSUM);
        Attn_rv = attn_rnM->getOutput(0);
    }
    auto attn_rv_s = m_Network->addSoftMax(*Attn_rv);
    attn_rv_s->setAxes(8);
    auto v_ = m_Network->addShuffle(*v->getOutput(0));
    v_->setReshapeDimensions(Dims4{qkvDims.d[1],qkvDims.d[2],qkvDims.d[3],qkvDims.d[4]});
    auto attn_v = m_Network->addMatrixMultiply(*attn_rv_s->getOutput(0),MatrixOperation::kNONE,
                                               *v_->getOutput(0),MatrixOperation::kNONE);
    auto x_reshape = shuffle_reshapeApermute(m_Network,attn_v->getOutput(0),Dims3{b,n,c},Permutation{0,2,1,3},false);
    auto x_linear = trt_swinLinear(m_Network,weightMap,x_reshape,lname+".proj");
    return x_linear;
}
ITensor* trt_window_reverse(INetworkDefinition *m_Network, ITensor *input, int window_size, int H, int W)
{
    Dims viewDims;
    viewDims.nbDims = 5;
    int d[5] = {H/window_size,W/window_size,window_size,window_size,-1};
    for(int i = 0; i < 5; i++)
        viewDims.d[i] = d[i];
    auto x_view = shuffle_reshape(m_Network,input,viewDims);
    auto output = shuffle_reshapeApermute(m_Network,x_view,Dims3{H,W,-1},Permutation{0,2,1,3,4},false);
    return output;
}
ITensor* gelu(INetworkDefinition *m_Network,ITensor *input)
{
    auto creator = getPluginRegistry()->getPluginCreator("geluLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("gelu", pluginData);
    ITensor* inputTensors[] = {input};
    auto g = m_Network->addPluginV2(inputTensors, 1, *pluginObj);
    return g->getOutput(0);
}
//ITensor* adaptiveAvgPool2d(INetworkDefinition *m_Network,ITensor *input)
//{
//    auto creator = getPluginRegistry()->getPluginCreator("adaptiveAvgPooling_TRT", "1");
//    const PluginFieldCollection* pluginData = creator->getFieldNames();
//    IPluginV2 *pluginObj = creator->createPlugin("apAvgPool", pluginData);
//    ITensor* inputTensors[] = {input};
//    auto g = m_Network->addPluginV2(inputTensors, 1, *pluginObj);
//    return g->getOutput(0);
//}
ITensor* trt_transform_mlp(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,ITensor *input,
                           string lname,int dim,int mlp_ratio = 4)
{
//    auto fc1 = m_Network->addFullyConnected(*input,dim * mlp_ratio,
//                                            weightMap[lname+".fc1.weight"],weightMap[lname+".fc1.bias"]);
    auto fc1 = trt_swinLinear(m_Network,weightMap,input,lname+".fc1");
    auto act = gelu(m_Network,fc1);
//    auto fc2 = m_Network->addFullyConnected(*act,dim ,
//                                            weightMap[lname+".fc2.weight"],weightMap[lname+".fc2.bias"]);
    auto fc2 = trt_swinLinear(m_Network,weightMap,act,lname+".fc2");
    return fc2;
}
ITensor* blk(INetworkDefinition *m_Network,std::map<std::string, Weights> weightMap,ITensor *input, ITensor* mask, string lname,
             int hw,int dim, int num_heads,int window_size,int shift_size,int mlp_ratio = 4)
{
    int c = input->getDimensions().d[input->getDimensions().nbDims - 1];
    auto x = input;
    auto norm1 = m_layerNorm(m_Network,weightMap,x,lname+".norm1");
    //auto norm1 = x;
    auto view1 = shuffle_reshape(m_Network,norm1,Dims3{hw,hw,c});
    auto pad = trt_transform_pad(m_Network,view1,window_size);
    int hp = pad->getDimensions().d[0];
    int wp = pad->getDimensions().d[1];
    ITensor* shifted_x;
    ITensor* atten_mask = nullptr;
    if(shift_size > 0)
    {
        shifted_x = trt_swinRoll(m_Network,pad,{-3,-3},{1,2});
        atten_mask = mask;
    }
    else
    {
        shifted_x = pad;
    }
    auto x_windows = trt_transform_window_partition(m_Network,shifted_x,window_size);
    auto x_atten_windows = trt_trainsform_WindowAttention(m_Network,weightMap,x_windows,atten_mask,lname+".attn",dim,num_heads,
                                                          window_size,shift_size);
    auto x_atten_windows_view = shuffle_reshape(m_Network,x_atten_windows,Dims4{-1,window_size,window_size,c});

    shifted_x = trt_window_reverse(m_Network,x_atten_windows_view,window_size,hp,wp);
    if(shift_size > 0)
    {
        x = trt_swinRoll(m_Network,shifted_x,{3,3},{1,2});
    }
    else {
        x = shifted_x;
    }
    if(hw < hp){
        auto sss = m_Network->addSlice(*x,Dims3{0,0,0},Dims3{hw,hw,c},Dims3{1,1,1});
        x = sss->getOutput(0);
    }
    x = shuffle_reshape(m_Network,x,Dims2{hw*hw,c});
    x = m_Network->addElementWise(*x,*input,ElementWiseOperation::kSUM)->getOutput(0);
    auto norm2 = m_layerNorm(m_Network,weightMap,x,lname+".norm2");
    //auto norm2 = x;
    auto mlp = trt_transform_mlp(m_Network,weightMap,norm2,lname+".mlp",dim);
    auto out= m_Network->addElementWise(*x,*mlp,ElementWiseOperation::kSUM)->getOutput(0);
    debug_print(out, "blk");
    return out;
}
ITensor* downsample(INetworkDefinition* m_Network,std::map<std::string, Weights> weightMap,ITensor *input,
                    string lname, int hw)
{
    int c = input->getDimensions().d[input->getDimensions().nbDims - 1];
    auto x = shuffle_reshape(m_Network,input,Dims3{hw,hw,c});
    auto x0 = m_Network->addSlice(*x,Dims3{0,0,0},Dims3{hw/2,hw/2,c},Dims3{2,2,1});
    auto x1 = m_Network->addSlice(*x,Dims3{1,0,0},Dims3{hw/2,hw/2,c},Dims3{2,2,1});
    auto x2 = m_Network->addSlice(*x,Dims3{0,1,0},Dims3{hw/2,hw/2,c},Dims3{2,2,1});
    auto x3 = m_Network->addSlice(*x,Dims3{1,1,0},Dims3{hw/2,hw/2,c},Dims3{2,2,1});
    ITensor* inputTensors[] = { x0->getOutput(0), x1->getOutput(0), x2->getOutput(0), x3->getOutput(0) };
    auto cat = m_Network->addConcatenation(inputTensors, 4);
    cat->setAxis(2);
    auto cat_view = shuffle_reshape(m_Network,cat->getOutput(0),Dims2{-1,4*c});
    auto norm = m_layerNorm(m_Network,weightMap,cat_view,lname+".norm");
    //auto norm = cat_view;
    auto reduction = trt_swinLinear(m_Network,weightMap,norm,lname+".reduction",false);
    return reduction;
}
ITensor* addBatchNorm2d(
INetworkDefinition *network,
std::map<std::string, Weights> weightMap,
ITensor* input,
const std::string& lname,
float eps = 1e-5
) {
    float *gamma = (float*)(weightMap[lname + ".weight"].values);
    float *beta = (float*)(weightMap[lname + ".bias"].values);
    float *mean = (float*)(weightMap[lname + ".running_mean"].values);
    float *var = (float*)(weightMap[lname + ".running_var"].values);
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(*input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1->getOutput(0);
}
ITensor* transform_lateral_conv(INetworkDefinition* m_Network,std::map<std::string, Weights> weightMap,ITensor* input,
                                string lname, int k = 1, int s = 1,int out_features = 512)
{
    Weights empty{DataType::kFLOAT,nullptr,0};
    auto conv = m_Network->addConvolutionNd(*input,out_features,Dims2{k,k},weightMap[lname+".conv.weight"],empty);
    conv->setStrideNd(Dims2{s,s});
    conv->setNbGroups(1);
    conv->setPaddingNd(Dims2{k/2,k/2});
    ITensor* bn = addBatchNorm2d(m_Network,weightMap,conv->getOutput(0),lname+".bn");
    auto act = m_Network->addActivation(*bn,ActivationType::kRELU);
    return act->getOutput(0);
}
ITensor* resize(INetworkDefinition* m_Network, ITensor* input, int grid)
{
    float scale_h = 2.0f;
    float scale_w = 2.0f;

    scale_h = 1.0*grid / input->getDimensions().d[1];
    scale_w = 1.0*grid / input->getDimensions().d[2];

    auto creator = getPluginRegistry()->getPluginCreator("UpsamplePlugin", "1");
    PluginField pField[1];
    float *s = new float[2];
    s[0] = scale_h;
    s[1] = scale_w;
    pField[0].data = s;
    pField[0].length = 2;
    pField[0].type = PluginFieldType::kFLOAT32;
    pField[0].name = "scaleFactor";

    PluginFieldCollection pluginData;
    pluginData.nbFields = 1;
    pluginData.fields = pField;
    IPluginV2 *pluginObj = creator->createPlugin("upSample", &pluginData);
    ITensor* inputTensors[] = {input};
    auto upS = m_Network->addPluginV2(inputTensors, 1, *pluginObj);
    return upS->getOutput(0);
}
ITensor* transform_psp(INetworkDefinition* m_Network,std::map<std::string, Weights> weightMap,ITensor* input,
                       string lname, int output_Avg_Size, int out_features = 512)
{
    int inH = input->getDimensions().d[1];
    int inW = input->getDimensions().d[2];
    int kH = inH / output_Avg_Size;
    int kW = inW / output_Avg_Size;
    auto avgPool = m_Network->addPoolingNd(*input,PoolingType::kAVERAGE,Dims2{kH,kW});
    avgPool->setStrideNd(Dims2{kH,kW});
    auto cba = transform_lateral_conv(m_Network,weightMap,avgPool->getOutput(0),lname,1,1,out_features);
    auto out = resize(m_Network,cba,inH);
    return out;
}
ITensor* up_Add(INetworkDefinition* m_Network,ITensor* input1,ITensor* input2)
{
    auto in1 = resize(m_Network,input1,input2->getDimensions().d[1]);
    auto out = m_Network->addElementWise(*in1,*input2,ElementWiseOperation::kSUM);
    return out->getOutput(0);
}


#endif // COMMON_HPP
