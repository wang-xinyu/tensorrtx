ARG TENSORRT="7"
ARG CUDA="10"

FROM hakuyyf/tensorrtx:trt${TENSORRT}_cuda${CUDA}

# Get opencv 3.4 for bionic based images
RUN rm /etc/apt/sources.list.d/timsc-ubuntu-opencv-3_3-bionic.list
RUN rm /etc/apt/sources.list.d/timsc-ubuntu-opencv-3_3-bionic.list.save
RUN add-apt-repository -y ppa:timsc/opencv-3.4

RUN apt-get update
RUN apt-get install -y libopencv-dev libopencv-dnn-dev libopencv-shape3.4-dbg

# git clone tensorrtx
RUN git clone https://github.com/wang-xinyu/tensorrtx.git