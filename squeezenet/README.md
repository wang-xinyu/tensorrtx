# squeezenet v1.1

SqueezeNet 1.1 model from the official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>

SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

For the Pytorch implementation, you can refer to [pytorchx/squeezenet](https://github.com/wang-xinyu/pytorchx/tree/master/squeezenet)

```
// 1. generate squeezenet.wts from [pytorchx/squeezenet](https://github.com/wang-xinyu/pytorchx/tree/master/squeezenet)

// 2. put squeezenet.wts into tensorrtx/squeezenet

// 3. build and run

cd tensorrtx/squeezenet

mkdir build

cd build

cmake ..

make

sudo ./squeezenet -s   // serialize model to plan file i.e. 'squeezenet.engine'
sudo ./squeezenet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/squeezenet
```

