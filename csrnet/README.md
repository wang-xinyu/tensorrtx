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

sudo ./csrnet -d  ./images

// output e.g
// enqueueV2 time: 0.0323869s
// detect time:44ms
// people num :22.9101 write_path: ../images/result_test.jpg
```


# result 

inference people num: 22.9101

<img src="https://raw.githubusercontent.com/wang-xinyu/tensorrtx/e358a5dd798a308d8213e2c8ad45730b26188f27/result_test.jpg"  alt="result jpg">
