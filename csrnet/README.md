# csrnet

The Pytorch implementation is [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch).

This repo is a TensorRT implementation of CSRNet.

paper : [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062)

Dev environment:
- Ubuntu 22.04
- TensorRT 8.6
- OpenCV 4.5.4
- CMake 3.24
- GPU Driver 535.113.01
- CUDA 12.2
- RTX3080


# how to run

```bash
1. generate csrnet engine
git clone https://github.com/leeyeehoo/CSRNet-pytorch.git
git clone https://github.com/wang-xinyu/tensorrtx.git
// copy gen_wts.py to CSRNet-pytorch
// generate wts file
python gen_wts.py
// csrnet wts will be generated in CSRNet-pytorch

2. build csrnet.engine
// mv CSRNet-pytorch/csrnet.engine to tensorrtx/csrnet
mv CSRNet-pytorch/csrnet.wts tensorrtx/csrnet
// build
mkdir build
cmake ..
make
sudo ./csrnet -s  ./csrnet.wts

Loading weights: ./csrnet.wts
build engine successfully : ./csrnet.engine

sudo ./csrnet -d  ./data

// output e.g
enqueueV2 time: 0.0328306s
detect time:44ms
Sum: 22.9101
```