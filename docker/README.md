# Tutorials

## Introduction

This folder contains the docker and docker-compose file to build the development environment without pain.

## Prerequisites

* OS: Linux or WSL2
* docker
* nvidia-container-toolkit
* (Optional but **recommended**) docker-compose

## Usage

1. (With docker-compose) configure the `.env` file, change `DATA_DIR` to your mount point, such as your code or data folder, etc, comment the `volumes` in docker compose file if not necessariy needed

2. Build image:
```bash
docker compose -f docker-compose.yml build
```

3. Run a container at background:
```bash
docker compose -f docker-compose.yml up -d
```

4. Attach to this container with your IDE and have fun!

## HowTos

### How to build and run with docker?

``` bash
docker build -f docker/x86_64.dockerfile -v .
docker run -it --gpus all --privileged --net=host --ipc=host -v  /bin/bash
```

### How to build image with other TensorRT version?

Change the `TAG` on top of the `.dockerfile`. Note: all images are officially owned by NVIDIA NGC, which requires a registration before pulling. For this repo, the mainly used `TAG` would be:

| Container Image | Container OS | Driver | CUDA | TensorRT | Torch | Recommended |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 20.12-py3 | Ubuntu 20.04 | 455 | 11.2 | 7.2.2 | 1.8.0 | ❌ |
| 24.01-py3 | Ubuntu 22.04 | 545 | 12.3 | 8.6.1 | 2.2.0 | ✅ |
| 24.04-py3 | Ubuntu 22.04 | 545 | 12.4 | 8.6.3 | 2.3.0 | ✅ |
| 24.09-py3 | Ubuntu 22.04 | 560 | 12.6 | 10.4.0 | 2.5.0 | ✅ |

For more detail of the support matrix, please check [HERE](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)

### How to customize opencv?

If prebuilt package from apt cannot meet your requirements, please refer to the demo code in `.dockerfile` to build opencv from source.

### How to solve image build fail issues?

For *443 timeout* or any similar network issues, a proxy may required. To make your host proxy work for building env of docker, please change the `build` node inside docker-compose file like this:
```YAML
    build:
      dockerfile: x86_64.dockerfile
      args:
        HTTP_PROXY: ${PROXY}
        HTTPS_PROXY: ${PROXY}
        ALL_PROXY: ${PROXY}
        http_proxy: ${PROXY}
        https_proxy: ${PROXY}
        all_proxy: ${PROXY}
```
then add `PROXY="http://xxx:xxx"` in `.env` file

## Note

The older version support, like TensorRT version **< 8**, may be deprecated in the future.
