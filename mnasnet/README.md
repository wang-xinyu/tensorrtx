# mnasnet

MNASNet with depth multiplier of 0.5 from 
"MnasNet: Platform-Aware Neural Architecture Search for Mobile"   <https://arxiv.org/pdf/1807.11626.pdf>

For the Pytorch implementation, you can refer to [pytorchx/mnasnet](https://github.com/wang-xinyu/pytorchx/tree/master/mnasnet)

Following tricks are used in this mnasnet, nothing special, group conv and batchnorm are used.

- Batchnorm layer, implemented by scale layer.

```
// 1. generate mnasnet.wts from [pytorchx/mnasnet](https://github.com/wang-xinyu/pytorchx/tree/master/mnasnet)

// 2. put mnasnet.wts into tensorrtx/mnasnet

// 3. build and run

cd tensorrtx/mnasnet

mkdir build

cd build

cmake ..

make

sudo ./mnasnet -s   // serialize model to plan file i.e. 'mnasnet.engine'

sudo ./mnasnet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/mnasnet
```


