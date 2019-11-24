# alexnet

AlexNet model architecture from the "One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

For the details, you can refer to [pytorchx/alexnet](https://github.com/wang-xinyu/pytorchx/tree/master/alexnet)

This alexnet is just several `conv-relu-pool` blocks followed by several `fc-relu`, nothing special. All layers can be implemented by tensorrt api, including `addConvolution`, `addActivation`, `addPooling`, `addFullyConnected`.

```
// 1. generate alexnet.wts from [pytorchx/alexnet](https://github.com/wang-xinyu/pytorchx/tree/master/alexnet)

// 2. put alexnet.wts into tensorrtx/alexnet

// 3. build and run

cd tensorrtx/alexnet

mkdir build

cd build

cmake ..

make

sudo ./alexnet -s   // serialize model to plan file i.e. 'alexnet.engine'

sudo ./alexnet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/alexnet
```


