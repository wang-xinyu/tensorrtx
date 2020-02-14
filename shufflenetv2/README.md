# shufflenet v2

ShuffleNetV2 with 0.5x output channels, as described in
 "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
            <https://arxiv.org/abs/1807.11164>

For the Pytorch implementation, you can refer to [pytorchx/shufflenet](https://github.com/wang-xinyu/pytorchx/tree/master/shufflenet)

Following tricks are used in this shufflenet,

- `torch.chunk` is used in shufflenet v2. We implemented the 'chunk(2, dim=C)' by tensorrt plugin. Which is the simplest plugin in this tensorrtx project. You can learn the basic procedures of build tensorrt plugin.
- shuffle layer is used, the `channel_shuffle()` in pytorchx/shufflenet can be implemented by two shuffle layers in tensorrt.
- Batchnorm layer, implemented by scale layer.

```
// 1. generate shufflenet.wts from [pytorchx/shufflenet](https://github.com/wang-xinyu/pytorchx/tree/master/shufflenet)

// 2. put shufflenet.wts into tensorrtx/shufflenet

// 3. build and run

cd tensorrtx/shufflenet

mkdir build

cd build

cmake ..

make

sudo ./shufflenet -s   // serialize model to plan file i.e. 'shufflenet.engine'
sudo ./shufflenet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/shufflenet
```


