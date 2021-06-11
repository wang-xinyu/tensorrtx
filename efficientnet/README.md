# EfficientNet

A TensorRT implementation of EfficientNet.
For the Pytorch implementation, you can refer to [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## How to run

1. install `efficientnet_pytorch`
```
pip install efficientnet_pytorch
```

2. gennerate `.wts` file
```
python gen_wts.py
```

3. build

```
mkdir build
cd build
cmake ..
make
```
4. serialize model to engine
```
./efficientnet -s [.wts] [.engine] [b0 b1 b2 b3 ... b7]  // serialize model to engine file
```
such as
```
./efficientnet -s ../efficientnet-b3.wts efficientnet-b3.engine b3
```
5. deserialize and do infer
```
./efficientnet -d [.engine] [b0 b1 b2 b3 ... b7]   // deserialize engine file and run inference
```
such as 
```
./efficientnet -d efficientnet-b3.engine b3
```
6. see if the output is same as pytorch side


For more models, please refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
