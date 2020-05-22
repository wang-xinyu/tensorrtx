# Migrating from TensorRT 4 to 7

The following APIs are deprecated and replaced in TensorRT 7.

- `DimsCHW`, replaced by `Dims3`
- `addConvolution()`, replaced by `addConvolutionNd()`
- `addPooling()`, replaced by `addPoolingNd()`
- `addDeconvolution()`, replaced by `addDeconvolutionNd()`
- `createNetwork()`, replaced by `createNetworkV2()`
- `buildCudaEngine()`, replaced by `buildEngineWithConfig()`
- `createPReLUPlugin()`, replaced by `addActivation()` with `ActivationType::kLEAKY_RELU`
- `IPlugin` and `IPluginExt` class, replaced by `IPluginV2IOExt` or `IPluginV2DynamicExt`
- Use the new `Logger` class defined in logging.h
