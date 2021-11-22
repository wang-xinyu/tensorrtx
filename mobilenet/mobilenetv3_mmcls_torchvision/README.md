# mobilenet v3

MobileNetV3 architecture from
     "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244?context=cs>.

For the Pytorch implementation, you can refer to [mmclassification](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/mobilenet_v3.py) or [torchvision](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html)

## Run

1. generate mbv3_small.wts/mbv3_large.wts

Example

```shell
# small
python gen_wts.py -w small.pth -o mbv3_small.wts -s small

# large
python gen_wts.py -w large.pth -o mbv3_large.wts -s large
```

2. put mbv3_small.wts/mbv3_large.wts into tensorrtx/mobilenet/mobilenetv3_mmcls_torchvision

3. build and run

```
cd tensorrtx/mobilenet/mobilenetv3_mmcls_torchvision
mkdir build
cd build
cmake ..
make
sudo ./mobilenetv3 -s small(or large) // serialize model to plan file i.e. 'mobilenetv3_small.engine'
sudo ./mobilenetv3 -d small(or large)  // deserialize plan file and run inference
```

4. see if the output is same as pytorch side

### TensorRT Python API

```
# 1. generate mobilenetv3.wts

# 2. put mobilenetv3.wts into tensorrtx/mobilenet/mobilenetv3_mmcls_torchvision

# 3. install Python dependencies (tensorrt/pycuda/numpy)

cd tensorrtx/mobilenet/mobilenetv3_mmcls_torchvision

python mobilenet_v3.py -s small(or large)  // serialize model to plan file i.e. 'mobilenetv3.engine'
python mobilenet_v3.py -d small(or large)  // deserialize plan file and run inference

```
