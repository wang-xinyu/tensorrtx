# Migration Guide

## <u>Newest</u> Migration Guide

Please check this [Doc](https://docs.nvidia.com/deeplearning/tensorrt/pdf/TensorRT-Migration-Guide.pdf) or this [Page](https://docs.nvidia.com/deeplearning/tensorrt/migration-guide/index.html)

For any archives version, please check this [Page](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)

## (DEPRECATED) Migrating from TensorRT 4.x to 7.x

**NOTE**: Both TensorRT 4.x and 7.x are **DEPRECATED** by NVIDIA officially, so this part is **outdated**.

The following APIs are deprecated and replaced in TensorRT 7.
- `DimsCHW`, replaced by `Dims3`
- `addConvolution()`, replaced by `addConvolutionNd()`
- `addPooling()`, replaced by `addPoolingNd()`
- `addDeconvolution()`, replaced by `addDeconvolutionNd()`
- `createNetwork()`, replaced by `createNetworkV2()`
- `buildCudaEngine()`, replaced by `buildEngineWithConfig()`
- `createPReLUPlugin()`, replaced by `addActivation()` with `ActivationType::kLEAKY_RELU`
- `IPlugin` and `IPluginExt` class, replaced by `IPluginV2IOExt` or `IPluginV2DynamicExt`
- Use the new `Logger` class defined in `logging.h`
