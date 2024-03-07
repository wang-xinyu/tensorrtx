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

// download images https://github.com/wang-xinyu/tensorrtx/assets/46584679/46bc4def-e573-44ae-996d-5d68927c78ff and copy to images
sudo ./csrnet -d  ./images

// output e.g
// enqueueV2 time: 0.0323869s
// detect time:44ms
// people num :22.9101 write_path: ../images/data.jpg
```


# result 

inference people num: 22.9101

<p align="center">
<img src= https://raw.githubusercontent.com/wang-xinyu/tensorrtx/dbf857d25f77bf64113fc99a745ccf4973bdd44e/Density_Plot.jpg>
</p>
