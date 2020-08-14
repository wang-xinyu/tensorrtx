# Check if Your GPU Supports FP16/INT8

## 1. check your GPU Compute Capability

visit https://developer.nvidia.com/cuda-gpus#compute and check your GPU compute capability.

For example, GTX1080 is 6.1, Tesla T4 is 7.5.

## 2. check the hardware-precision-matrix

visit https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix and check the matrix.

For example, compute capability 6.1 supports FP32 and INT8. 7.5 supports FP32, FP16, INT8, FP16 tensor core, etc.

