# mobilenet v2

MobileNetV2 architecture from
     "MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>.

For the Pytorch implementation, you can refer to [pytorchx/mobilenet](https://github.com/wang-xinyu/pytorchx/tree/master/mobilenet)

Following tricks are used in this mobilenet,

- Relu6 is used in mobilenet v2. We use `Relu6(x) = Relu(x) - Relu(x-6)` in tensorrt.
- Batchnorm layer, implemented by scale layer.

```
// 1. generate mobilenet.wts from [pytorchx/mobilenet](https://github.com/wang-xinyu/pytorchx/tree/master/mobilenet)

// 2. put mobilenet.wts into tensorrtx/mobilenet

// 3. build and run

cd tensorrtx/mobilenet/mobilenetv2

mkdir build

cd build

cmake ..

make

sudo ./mobilenet -s   // serialize model to plan file i.e. 'mobilenet.engine'

sudo ./mobilenet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/mobilenet
```

### TensorRT Python API

```
# 1. generate mobilenetv2.wts from [pytorchx/mobilenet](https://github.com/wang-xinyu/pytorchx/tree/master/mobilenet)

# 2. put mobilenetv2.wts into tensorrtx/mobilenet/mobilenetv2

# 3. install Python dependencies (tensorrt/pycuda/numpy)

cd tensorrtx/mobilenet/mobilenetv2

python mobilenet_v2.py -s   // serialize model to plan file i.e. 'mobilenetv2.engine'
python mobilenet_v2.py -d   // deserialize plan file and run inference

# 4. see if the output is same as pytorchx/mobilenet
```
