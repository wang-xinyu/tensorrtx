#include <assert.h>
#include "yololayer.h"
#include "utils.h"

using namespace Yolo;

namespace nvinfer1
{
    /**
    * @brief Default constructor for the YoloLayerPlugin.
    *
    * Initializes the YOLO layer plugin with default settings:
    * - mS: grid size
    * - mB: number of bounding boxes per grid cell
    * - mClasses: number of object classes
    * - mOutputSize: total size of the output tensor
    * - mThreadCount: number of threads to use in CUDA kernel
    */
    YoloLayerPlugin::YoloLayerPlugin()
    {
        mS = S;
        mB = B;
        mClasses = CLASSES;

        mOutputSize = OUTPUT_SIZE;

        mThreadCount = 256;
    }

    /**
    * @brief Deserialize constructor for the YoloLayerPlugin.
    *
    * This constructor is used to reconstruct the plugin from serialized data.
    * It reads the plugin parameters from the given memory buffer.
    *
    * @param data Pointer to the serialized plugin data.
    * @param length Length of the serialized data in bytes.
    *
    * @note The serialized data must contain values for mS, mB, mClasses, mOutputSize,
    * and mThreadCount, in that order.
    */
    YoloLayerPlugin::YoloLayerPlugin(const void * data, size_t length)
    {
        using namespace Tn;

        const char * d = reinterpret_cast<const char*>(data);
        const char * start = d;

        read(d, mS);
        read(d, mB);
        read(d, mClasses);
        read(d, mOutputSize);
        read(d, mThreadCount);

        assert(d == start + length);
    }

    /**
    * @brief Destroy the Yolo Layer Plugin:: Yolo Layer Plugin object
    * 
    */
    YoloLayerPlugin::~YoloLayerPlugin()
    {

    }

    /**
    * @brief Serialize the plugin parameters to a memory buffer.
    *
    * This function writes all internal parameters of the YOLO layer plugin
    * into the provided memory buffer. The serialized data can later be
    * used to reconstruct the plugin using the deserialization constructor.
    *
    * @param buffer Pointer to the memory buffer where serialized data will be stored.
    *
    * @note The buffer must have at least getSerializationSize() bytes allocated.
    */
    void YoloLayerPlugin::serialize(void * buffer) const TRT_NOEXCEPT
    {
        using namespace Tn;

        char * d = reinterpret_cast<char *>(buffer);
        char * start = d;

        write(d, mS);
        write(d, mB);
        write(d, mClasses);
        write(d, mOutputSize);
        write(d, mThreadCount);

        assert(d == start + getSerializationSize());
    }

    /**
    * @brief Get the size in bytes required to serialize the plugin.
    *
    * This function returns the total number of bytes needed to store all
    * internal parameters of the YOLO layer plugin when calling serialize().
    *
    * @return size_t The size in bytes required for serialization.
    */
    size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return sizeof(mS) + sizeof(mB) + sizeof(mClasses) + sizeof(mOutputSize) + sizeof(mThreadCount);
    }

    /**
    * @brief Initialize the YOLO layer plugin.
    *
    * This function is called before the plugin is used for inference.
    * It can be used to allocate resources, initialize variables, or
    * perform any setup required for the plugin to run. 
    *
    * @return int Returns 0 if initialization is successful.
    */
    int YoloLayerPlugin::initialize() TRT_NOEXCEPT
    { 
        return 0;
    }

    /**
    * @brief Get the output dimensions of the YOLO layer plugin.
    *
    * This function calculates the dimensions of the output tensor based
    * on the YOLOv1 layer configuration:
    * - For YOLOv1, the output tensor shape is typically (C, H, W)
    *   where:
    *     - C = 5 * B + number of classes (each grid cell predicts B boxes + class scores)
    *     - H = S (grid size)
    *     - W = S (grid size)
    *
    * In this implementation, the output is returned as Dims2 with:
    * - H*H rows (number of grid cells)
    * - 5 + number of classes columns (per-cell predictions)
    *
    * @param index The index of the output tensor (usually 0 for a single-output plugin).
    * @param inputs Pointer to the input tensor dimensions.
    * @param nbInputDims Number of input dimensions.
    * @return Dims The dimensions of the plugin output tensor.
    */
    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        // YOLOv1 output dims: (C, H, W)
        // C = (5B + C)
        // H = S
        // W = S
        int C = (5 * mB + mClasses);
        int H = mS;
        int W = mS;

        // return Dims3(C, H, W);
        return Dims2(H * H, 5 + mClasses);
    }

    /**
    * @brief Set the namespace for the plugin.
    *
    * TensorRT plugins can be assigned a namespace to avoid naming conflicts
    * when multiple plugins are used in the same network. This function
    * sets the namespace for the current YOLO layer plugin.
    *
    * @param pluginNamespace Pointer to a null-terminated string representing the plugin namespace.
    */
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    /**
    * @brief Get the namespace of the plugin.
    *
    * Returns the namespace assigned to this YOLO layer plugin. Plugin namespaces
    * are used in TensorRT to avoid naming conflicts when multiple plugins are
    * used in the same network.
    *
    * @return const char* Pointer to the null-terminated string representing the plugin namespace.
    */
    const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    /**
    * @brief Get the data type of the plugin output tensor.
    *
    * This function specifies the data type of the plugin's output.
    * For the YOLO layer plugin, the output tensor is always of type float.
    *
    * @param index The index of the output tensor (usually 0 for a single-output plugin).
    * @param inputTypes Pointer to an array of input tensor data types.
    * @param nbInputs Number of input tensors.
    * @return DataType The data type of the plugin output tensor (here, always kFLOAT).
    */
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    /**
    * @brief Check if the plugin output is broadcast across the batch.
    *
    * This function indicates whether the plugin's output tensor is identical
    * for all batch elements. For the YOLO layer plugin, the output is unique
    * per batch element, so this function always returns false.
    *
    * @param outputIndex The index of the output tensor (usually 0 for single-output plugin).
    * @param inputIsBroadcasted Pointer to an array indicating which inputs are broadcasted.
    * @param nbInputs Number of input tensors.
    * @return true If the output is broadcast across the batch.
    * @return false If the output is not broadcast across the batch (here, always false).
    */
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    /**
    * @brief Check if the plugin can broadcast a specific input across the batch.
    *
    * This function indicates whether the plugin is able to treat the input tensor
    * as the same for all batch elements. For the YOLO layer plugin, each input
    * is unique per batch element, so this function always returns false.
    *
    * @param inputIndex The index of the input tensor to check.
    * @return true If the input can be broadcast across the batch.
    * @return false If the input cannot be broadcast across the batch (here, always false).
    */
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    /**
    * @brief Configure the plugin with input and output tensor information.
    *
    * This function is called by TensorRT during network construction to provide
    * the plugin with information about input and output tensors, such as
    * their data types, dimensions, and formats. The plugin can use this
    * information to set up internal parameters or allocate resources.
    *
    * @param in Pointer to an array of input tensor descriptions.
    * @param nbInput Number of input tensors.
    * @param out Pointer to an array of output tensor descriptions.
    * @param nbOutput Number of output tensors.
    */
    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    /**
    * @brief Attach the plugin to GPU contexts and allocator.
    *
    * This function is called by TensorRT when the plugin is associated with
    * a CUDA context. It provides the plugin with handles to cuDNN, cuBLAS,
    * and the GPU memory allocator, which can be used during inference.
    *
    * @param cudnnContext Pointer to the cuDNN context.
    * @param cublasContext Pointer to the cuBLAS context.
    * @param gpuAllocator Pointer to the GPU memory allocator.
    */
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    /**
    * @brief Detach the plugin from GPU contexts.
    *
    * This function is called by TensorRT when the plugin is being removed
    * from a CUDA context. It allows the plugin to release any resources
    * associated with cuDNN, cuBLAS, or GPU memory that were acquired in
    * attachToContext().
    */
    void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

    /**
    * @brief Get the type name of the plugin.
    *
    * This function returns a unique string identifying the plugin type.
    * TensorRT uses this string to match plugins during deserialization
    * and when creating plugin instances.
    *
    * @return const char* Null-terminated string representing the plugin type.
    */
    const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    /**
    * @brief Get the version of the plugin.
    *
    * This function returns a string representing the version of the plugin.
    * The version can be used by TensorRT or users to ensure compatibility
    * between serialized engines and plugin implementations.
    *
    * @return const char* Null-terminated string representing the plugin version.
    */
    const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    /**
    * @brief Destroy the plugin object.
    *
    * This function is called by TensorRT to release the plugin object.
    * It deletes the current instance of the plugin, freeing all associated resources.
    *
    * @note After calling this function, the plugin instance is no longer valid.
    */
    void YoloLayerPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    /**
    * @brief Create a deep copy of the plugin.
    *
    * This function creates a new instance of the YOLO layer plugin with
    * the same configuration as the current instance. The cloned plugin
    * can be used independently in another network or context.
    *
    * @return IPluginV2IOExt* Pointer to the newly cloned plugin instance.
    */
    IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT
    {
        YoloLayerPlugin *p = new YoloLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    /**
    * @brief CUDA kernel to decode YOLOv1 predictions.
    *
    * This kernel selects the bounding box with the higher confidence for each grid cell
    * and copies the class probabilities. It processes one grid cell per thread.
    *
    * @param input Pointer to the input tensor of shape (1, 30, S, S) containing YOLO predictions.
    *              The 30 channels correspond to 2 bounding boxes (5 each) + C class probabilities.
    * @param output Pointer to the output tensor of shape (1, 5 + C, S, S) containing the decoded predictions.
    *               For each grid cell, only the bounding box with higher confidence is kept.
    * @param S Grid size (number of cells along width/height, e.g., 7 for 7x7 grid).
    * @param C Number of object classes.
    *
    * @note This kernel assumes the input tensor follows YOLOv1 format: 2 bounding boxes and C class scores.
    *       Each thread handles one grid cell. The printf is used to verify GPU execution.
    */
    __global__ void decodeYoloKernel(
        const float* input,  // 1x30xSxS
        float* output,       // 1x(5+C)xSxS
        int S,
        int C
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0) printf("GPU decodeYoloKernel is running!\n");
        int totalGrid = S * S;
        if (idx >= totalGrid) return;

        int inStride = 30;
        int outStride = 5 + C;

        int inIdx = idx * inStride;
        int outIdx = idx * outStride;

        // B1
        float b1X    = input[inIdx + 0];
        float b1Y    = input[inIdx + 1];
        float b1W    = input[inIdx + 2];
        float b1H    = input[inIdx + 3];
        float b1Conf = input[inIdx + 4];

        // B2
        float b2X    = input[inIdx + 5];
        float b2Y    = input[inIdx + 6];
        float b2W    = input[inIdx + 7];
        float b2H    = input[inIdx + 8];
        float b2Conf = input[inIdx + 9];

        if (b1Conf > b2Conf)
        {
            output[outIdx + 0] = b1X;
            output[outIdx + 1] = b1Y;
            output[outIdx + 2] = b1W;
            output[outIdx + 3] = b1H;
            output[outIdx + 4] = b1Conf;
        }
        else
        {
            output[outIdx + 0] = b2X;
            output[outIdx + 1] = b2Y;
            output[outIdx + 2] = b2W;
            output[outIdx + 3] = b2H;
            output[outIdx + 4] = b2Conf;
        }

        //  class prob
        for (int c = 0; c < C; c++)
        {
            output[outIdx + 5 + c] = input[inIdx + 10 + c];
        }
    }

    /**
    * @brief Execute the YOLO layer plugin on the GPU.
    *
    * This function is called during inference to perform the YOLOv1 decoding.
    * For each batch element, it launches a CUDA kernel to:
    * - Select the bounding box with the higher confidence per grid cell.
    * - Copy the class probabilities for the selected bounding box.
    *
    * @param batchSize Number of elements in the current batch.
    * @param inputs Array of input tensor pointers. The input tensor is expected
    *               to be in YOLOv1 format: (1, 30, S, S) per batch element.
    * @param outputs Array of output tensor pointers. The output tensor has shape
    *                (1, 5 + C, S, S) per batch element.
    * @param workspace Pointer to temporary GPU memory (unused here, but provided by TensorRT).
    * @param stream CUDA stream to launch kernels on.
    * @return int Returns 0 if the kernel was successfully launched.
    *
    * @note Each thread in the kernel processes one grid cell. This implementation
    *       handles each batch element sequentially.
    *       The CUDA kernel `decodeYoloKernel` is responsible for selecting the
    *       highest-confidence bounding box and copying class probabilities.
    */
    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        const int S = mS;
        const int C = mClasses;
        const int gridSize = S * S;

        const float* inputDev = reinterpret_cast<const float*>(inputs[0]);
        float* outputDev = reinterpret_cast<float*>(outputs[0]);

        for (int b = 0; b < batchSize; ++b)
        {
            const float* inputPtr = inputDev + b * gridSize * 30;
            float* outputPtr      = outputDev + b * gridSize * (5 + C);

            int threads = 128;
            int blocks = (gridSize + threads - 1) / threads;

            decodeYoloKernel<<<blocks, threads, 0, stream>>>(
                inputPtr, outputPtr, S, C
            );
        }

        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    /**
    * @brief Default constructor for the YoloPluginCreator.
    *
    * This constructor initializes the plugin creator object. It clears
    * the plugin attribute list and sets up the PluginFieldCollection
    * structure used by TensorRT to describe plugin parameters.
    *
    * - mPluginAttributes: List of plugin parameters (initially empty).
    * - mFC: PluginFieldCollection structure pointing to mPluginAttributes.
    */
    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    /**
    * @brief Get the name of the plugin this creator generates.
    *
    * This function returns a unique string identifying the plugin type
    * that this plugin creator creates. TensorRT uses this name to match
    * plugins during network creation and deserialization.
    *
    * @return const char* Null-terminated string representing the plugin name.
    */
    const char * YoloPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    /**
    * @brief Get the version of the plugin this creator generates.
    *
    * This function returns a string representing the version of the plugin
    * that this plugin creator creates. TensorRT can use this version to
    * ensure compatibility between serialized engines and plugin implementations.
    *
    * @return const char* Null-terminated string representing the plugin version.
    */
    const char * YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    /**
    * @brief Get the collection of plugin fields for this creator.
    *
    * This function returns a pointer to a PluginFieldCollection structure,
    * which describes all the configurable parameters of the plugin. TensorRT
    * uses this information when creating a plugin instance via the plugin creator.
    *
    * @return const PluginFieldCollection* Pointer to the plugin field collection.
    */
    const PluginFieldCollection * YoloPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    /**
    * @brief Create a new YOLO layer plugin instance.
    *
    * This function constructs a new YOLO layer plugin with the configuration
    * provided by the PluginFieldCollection. It sets the plugin namespace
    * to match the creator's namespace.
    *
    * @param name Name of the plugin instance (can be used for identification, optional).
    * @param fc Pointer to a PluginFieldCollection containing plugin parameters.
    *           In this implementation, the fields are not actively used.
    * @return IPluginV2IOExt* Pointer to the newly created plugin instance.
    */
    IPluginV2IOExt * YoloPluginCreator::createPlugin(const char *name, const PluginFieldCollection * fc) TRT_NOEXCEPT
    {
        YoloLayerPlugin * obj = new YoloLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    /**
    * @brief Deserialize a YOLO layer plugin from serialized data.
    *
    * This function creates a new YOLO layer plugin instance by deserializing
    * its parameters from a previously serialized buffer. The plugin namespace
    * is set to match the creator's namespace.
    *
    * @param name Name of the plugin instance (can be used for identification, optional).
    * @param serialData Pointer to the serialized plugin data.
    * @param serialLength Length of the serialized data in bytes.
    * @return IPluginV2IOExt* Pointer to the newly deserialized plugin instance.
    */
    IPluginV2IOExt * YoloPluginCreator::deserializePlugin(const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}