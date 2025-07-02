# Installing Environment on Jetson Nano

Installing the required environment on Jetson Nano can require quite a bit of effort. So, I've included this notes on how I installed, hoping to save anyone trying it a couple of hours.

Jetson Nano is tied to Python 3.6, CUDA 10.2 and TensorRT 7.

First thing I did was to create a virtual environment.

```shell

python3 -m venv py36
source py36/bin/activate

```

# Using tensorrt
it seems that tensorrt installs hardcoded to `/usr/lib`, so in order to use it from the virtual environment:

```shell
    cp -r /usr/lib/python3.6/dist-packages/tensorrt [PATH-to-venv]/lib/python3.6/site-packages/
```

# Installing pucuda
I tried many different packages versions but what worked for me was:

```shell
    sudo apt install python3.6-dev build-essential libboost-python-dev libboost-thread-dev \
    libboost-system-dev libboost-filesystem-dev libboost-program-options-dev \
    libboost-serialization-dev libboost-iostreams-dev libboost-regex-dev

    pip install wheel
    pip install mako MarkupSafe

    pip install --upgrade pip setuptools wheel
    pip install Cython==0.29.37
    pip install numpy==1.19.5
    pip install six


    pip install pycuda==2019.1
```

# Installing OpenCV

In my case I built OpenCV from source for enabling CUDA and Gstreamer support.
I cloned from github the 4.5.2 version, as well as the contrib module.
It takes about 4 hours to complete the building process, so go to do anything interesting and come back later.

```shell

    sudo apt install -y \
        build-essential \
        cmake \
        git \
        libgtk2.0-dev \
        pkg-config \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        python3-dev \
        python3-numpy \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libdc1394-22-dev

    cd [PATH-WHERE-CLONED]

    mkdir build
    cd build

    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
        -D OPENCV_EXTRA_MODULES_PATH=[PATH-WHERE-CLONED]/opencv_contrib/modules \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -D WITH_OPENCL=OFF \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN=5.3 \
        -D CUDA_ARCH_PTX="" \
        -D WITH_CUDNN=ON \
        -D WITH_CUBLAS=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_NEON=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENMP=ON \
        -D WITH_OPENGL=ON \
        -D BUILD_TIFF=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D BUILD_TESTS=OFF \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python3=TRUE \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF \
        -D PYTHON3_EXECUTABLE=$(which python) \
        ..

    make -j 4

    make install

```
