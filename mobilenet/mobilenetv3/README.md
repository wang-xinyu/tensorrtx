# mobilenet v3

MobileNetV3 architecture from
     "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244?context=cs>.

For the Pytorch implementation, you can refer to [mobilenetv3.pytorch](https://github.com/chufei1995/mobilenetv3.pytorch)

## Run

1. generate mbv3_small.wts/mbv3_large.wts from pytorch implementation

2. put mbv3_small.wts/mbv3_large.wts into tensorrtx/mobilenet/mobilenetv3

3. build and run

```
cd tensorrtx/mobilenet/mobilenetv3
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
# 1. generate mobilenetv3.wts from [mobilenetv3.pytorch](https://github.com/chufei1995/mobilenetv3.pytorch)

# 2. put mobilenetv3.wts into tensorrtx/mobilenet/mobilenetv3

# 3. install Python dependencies (tensorrt/pycuda/numpy)

cd tensorrtx/mobilenet/mobilenetv3

python mobilenet_v2.py -s small(or large)  // serialize model to plan file i.e. 'mobilenetv2.engine'
python mobilenet_v2.py -d small(or large)  // deserialize plan file and run inference

```
