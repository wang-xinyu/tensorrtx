# IBN-Net

An implementation of IBN-Net, proposed in ["Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"](https://arxiv.org/abs/1807.09441), ECCV2018 by Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. 

For the Pytorch implementation, you can refer to [IBN-Net](https://github.com/XingangPan/IBN-Net)

## Features
- InstanceNorm2d
- bottleneck_ibn
- Resnet50-IBNA
- Resnet50-IBNB
- Multi-thread inference

## How to Run

* 1. generate .wts

  // for ibn-a
  ```
  python gen_wts.py a
  ```
  a file 'resnet50-ibna.wts' will be generated.

  // for ibn-b
  ```
  python gen_wts.py b
  ```
  a file 'resnet50-ibnb.wts' will be generated.
* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
* 3. build engine and run classification

  // put resnet50-ibna.wts/resnet50-ibnb.wts into tensorrtx/ibnnet
  
  // go to tensorrtx/ibnnet
  ```
  ./ibnnet -s  // serialize model to plan file
  ./ibnnet -d  // deserialize plan file and run inference
  ```
  