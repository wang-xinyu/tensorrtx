# Densenet121

The Pytorch implementation is [makaveli10/densenet](https://github.com/makaveli10/torchtrtz/tree/main/densenet). Model from torchvision.
The tensorrt implemenation is taken from [makaveli10/cpptensorrtz](https://github.com/makaveli10/cpptensorrtz/).

## How to Run

1. generate densenet121.wts from pytorch

```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/makaveli10/torchtrtz.git

// go to torchtrtz/densenet
// Enter these two commands to create densenet121.wts
python models.py
python gen_trtwts.py
```

2. build densenet and run

```
// put densenet121.wts into tensorrtx/densenet
// go to tensorrtx/densenet
mkdir build
cd build
cmake ..
make
sudo ./densenet -s  // serialize model to file i.e. 'densenet.engine'
sudo ./densenet -d  // deserialize model and run inference
```

3. Verify output from [torch impl](https://github.com/makaveli10/torchtrtz/blob/main/densenet/README.md)

TensorRT output[:5]:
```
    [-0.587389, -0.329202, -1.83404, -1.89935, -0.928404]
```

