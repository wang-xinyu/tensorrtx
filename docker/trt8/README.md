# Tutorials

## Pull docker image And build opencv with dnn module from source
- docker pull hakuyyf/tensorrtx:trt8_cuda11
- sudo docker run -it --name={yourname} --gpus all --privileged --net=host --ipc=host --pid=host -v {localpath}:{dockerpath} hakuyyf/tensorrtx:trt8_cuda11 /bin/bash
- docker start {container_id}
- docker attach {container_id}
- apt update
- cd /home
- git clone https://github.com/opencv/opencv.git
- git clone https://github.com/opencv/opencv_contrib.git
- cd opencv
- git checkout 4.5.5
- cd ../
- cd opencv_contrib
- git checkout 4.5.5
- cd ../
- cd opencv
- mkdir build && cd build
- cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN={[Your GPU Compute Capability](https://developer.nvidia.com/zh-cn/cuda-gpus#compute)} -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib/modules/ -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=/usr/bin/python3 -D BUILD_EXAMPLES=ON ..
- make -j
- make install
- ldconfig
- ln -s /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-x86_64-linux-gnu.so /usr/lib/python3.8/cv2.so

## Test
- cd /home
- python3 -c "import cv2; print(cv2.__version__)"
- git clone https://github.com/wang-xinyu/tensorrtx.git

