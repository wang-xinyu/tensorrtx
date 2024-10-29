# Install the dependencies of tensorrtx

Using docker as development environment is strongly recommended, you may check [HERE](../docker/README) for the deployment instructions of docker container and *ignore* the rest of this document.

While if this is not your case, we always recommend using major LTS version of your OS, Nvidia driver, CUDA, and so on.

## OS

Ubuntu-22.04 is recommended. It is strongly recommended to use `apt` to manage packages in Ubuntu.

## Nidia Related

### Driver

You should install the nvidia driver first before anything else, go to [Ubuntu Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu) for more details.

**NOTE**: Since version 560, the installation step is a little different than before, check [HERE](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#recent-updates) for more details.

### CUDA

Go to [NVIDIA CUDA Installation Guide for Linux](https://developer.nvidia.com/cuda-10.0-download-archive) for the detailed steps.

**NOTE**:
- Do not forget to check [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) to setup the environment correctly.
- Make your CUDA version comply with your driver version
- If you want multi-version CUDA, docker is strongly recommended.

### TensorRT

check [HERE](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading) to install TensorRT.

### (Optional) OpenCV

```
sudo apt-get update && sudo apt install libgtk-3-dev libopencv-dev
```

## Verify installation

```
dpkg -l | grep cuda
dpkg -l | grep nvinfer
dpkg -l | grep opencv
```
