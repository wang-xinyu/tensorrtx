ARG TAG=24.01-py3

FROM nvcr.io/nvidia/tensorrt:${TAG} AS tensorrtx

ENV DEBIAN_FRONTEND noninteractive

# basic tools
RUN apt update && apt-get install -y --fix-missing --no-install-recommends \
sudo wget curl git ca-certificates ninja-build tzdata pkg-config \
gdb libglib2.0-dev libmount-dev \
&& rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir yapf isort cmake-format pre-commit

## override older cmake
RUN find /usr/local/share -type d -name "cmake-*" -exec rm -rf {} + \
&& curl -fsSL "https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.sh" \
-o cmake.sh && bash cmake.sh --skip-license --exclude-subdir --prefix=/usr/local && rm cmake.sh

RUN apt update && apt-get install -y \
libopencv-dev \
&& rm -rf /var/lib/apt/lists/*

## a template to build opencv and opencv_contrib from source
# RUN git clone -b 4.x https://github.com/opencv/opencv_contrib.git \
# && git clone -b 4.x https://github.com/opencv/opencv.git opencv \
# && cmake -S opencv -B opencv/build -G Ninja \
# -DBUILD_LIST=core,calib3d,imgproc,imgcodecs,highgui \
# -DOPENCV_EXTRA_MODULES_PATH="/workspace/opencv_contrib/modules" \
# -DCMAKE_BUILD_TYPE=RELEASE \
# -DCMAKE_INSTALL_PREFIX=/usr/local \
# -DENABLE_FAST_MATH=ON \
# -DOPENCV_GENERATE_PKGCONFIG=ON \
# -DBUILD_opencv_python2=OFF \
# -DBUILD_opencv_python3=OFF \
# -DBUILD_JAVA=OFF \
# -DBUILD_DOCS=OFF \
# -DBUILD_PERF_TESTS=OFF \
# -DBUILD_TESTS=OFF \
# && ninja -C opencv/build install
