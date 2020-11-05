# SENet

An implementation of SENet, proposed in Squeeze-and-Excitation Networks by Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

[https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)

For the Pytorch implementation, you can refer to [wang-xinyu/senet.pytorch](https://github.com/wang-xinyu/senet.pytorch), which is forked from [moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch).


```
// 1. generate se_resnet50.wts from [wang-xinyu/senet.pytorch](https://github.com/wang-xinyu/senet.pytorch)

// 2. put se_resnet50.wts into tensorrtx/senet

// 3. build and run

cd tensorrtx/senet

mkdir build

cd build

cmake ..

make

sudo ./se_resnet -s   // serialize model to plan file i.e. 'se_resnet50.engine'

sudo ./se_resnet -d   // deserialize plan file and run inference

// 4. see if the output is same as [wang-xinyu/senet.pytorch]
```

