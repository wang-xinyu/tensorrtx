# Install the dependencies of tensorrtx

## Ubuntu

Ubuntu16.04 / cuda10.0 / cudnn7.6.5 / tensorrt7.0.0 / opencv3.3 would be the example, other versions might also work, just need you to try.

It is strongly recommended to use `apt` to manage software in Ubuntu.

### 1. Install CUDA

Go to [cuda-10.0-download](https://developer.nvidia.com/cuda-10.0-download-archive). Choose `Linux` -> `x86_64` -> `Ubuntu` -> `16.04` -> `deb(local)` and download the .deb package.

Then follow the installation instructions.

```
sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

### 2. Install TensorRT

Go to [nvidia-tensorrt-7x-download](https://developer.nvidia.com/nvidia-tensorrt-7x-download). You might need login.

Choose TensorRT 7.0 and `TensorRT 7.0.0.11 for Ubuntu 1604 and CUDA 10.0 DEB local repo packages`

Install with following commands, after `apt install tensorrt`, it will automatically install cudnn, nvinfer, nvinfer-plugin, etc.

```
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda10.0-trt7.0.0.11-ga-20191216_1-1_amd64.deb
sudo apt update
sudo apt install tensorrt
```

### 3. Install OpenCV

```
sudo add-apt-repository ppa:timsc/opencv-3.3
sudo apt-get update
sudo apt install libopencv-dev
```

### 4. Check your installation

```
dpkg -l | grep cuda
dpkg -l | grep nvinfer
dpkg -l | grep opencv
```

### 5. Run tensorrtx

It is recommended to go through the [getting started guide, lenet5 as a demo.](https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/getting_started.md) first.

But if you are proficient in tensorrt, please check the readme of the model you want directly.

