#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include "NvInfer.h"

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginExt
    {
        public:
            explicit YoloLayerPlugin(int class_num, int yolo_grid, int input_dim, int cuda_block, float anchors[6]);
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            bool supportsFormat(DataType type, PluginFormat format) const override { 
                return type == DataType::kFLOAT && format == PluginFormat::kNCHW; 
            }

            void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {};

            int initialize() override;

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() override;

            virtual void serialize(void* buffer) override;

            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);


        private:
            int class_num_;
            int yolo_grid_;
            int input_dim_;
            int cuda_block_;
            float anchors_[6];
    };
};

#endif 
