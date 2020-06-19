# How to Implement Multi-GPU Processing

Maybe you hope to take advantage of multiple GPU to make inference even faster. Here are few tips to help you deal with it! Take **YOLO V4** as an example.

## 1. Make custom plugin (i.e. YOLO layer and Mish layer for YOLO V4) running asynchronically.

To do this, we need to use CudaStream parameter in the kernels of all custom layers and use asynchronous functions.
For example, in function ` forwardGpu()` of **yololayer.cu**, you need to do the following changes to make sure that the engine will be running on a specific CudaStream.

  1) Change `cudaMemset(output + idx*outputElem, 0, sizeof(float))` to `cudaMemsetAsync(output + idx*outputElem, 0, sizeof(float), stream)`
  2) Change `CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>(inputs[i],output, numElem, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount ,outputElem)` to `CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>(inputs[i],output, numElem, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount ,outputElem)`
  
  ## 2. Create an engine for each device you want to use.
  
  Maybe it is a good idea to create a struct to store the engine, context and buffer for each device individually. For example,
  ```
  struct Plan{
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void buffers[2];
    cudaStream_t stream;
  };
  ```
  And then use `cudaSetDevice()` to make each engine you create running on specific device. Moreover, to maximize performance, make sure that the engine file you are using to deserialize is the one tensor RT optimized for this device.
  
  ## 3. Use function wisely
  Here are some knowledge I learned when trying to parallelize the inference.
  1) Do not use synchronized function , like `cudaFree()`, during inference.
  2) Using `cudaMallocHost()` instead of `malloc()` when allocating memory on the host side.
