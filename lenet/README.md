# lenet5

lenet5 is one of the simplest net in this repo. You can learn the basic procedures of building CNN from TensorRT API. This demo includes 2 major steps:

1. Build engine
    * define network
    * set input/output
    * serialize model to `.engine` file
2. Do inference
    * load and deserialize model from `.engine` file
    * run inference

## TensorRT C++ API

see [HERE](../README.md#how-to-run)

## TensorRT Python API

```bash
# 1. generate lenet5.wts from https://github.com/wang-xinyu/pytorchx/tree/master/lenet

# 2. put lenet5.wts into tensorrtx/lenet

# 3. install Python dependencies (tensorrt/pycuda/numpy)

cd tensorrtx/lenet

# 4.1 serialize model to plan file, i.e. 'lenet5.engine'
python lenet.py -s

# 4.2 deserialize plan file and run inference
python lenet.py -d

# 5. (Optional) see if the output is same as pytorchx/lenet
```
