ARG TENSORRT="7"
ARG CUDA="10"

FROM hakuyyf/tensorrtx:trt${TENSORRT}_cuda${CUDA}

RUN apt-get update

# git clone tensorrtx
RUN git clone https://github.com/wang-xinyu/tensorrtx.git