# swin_transform

The Pytorch implementation is [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer.git).


## How to Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
git clone https://github.com/microsoft/Swin-Transformer.git
git clone https://github.com/wang-xinyu/tensorrtx.git

python gen_wts.py best.pt
// a file 'Swin-Transform.wts' will be generated.
```

2. build tensorrtx/Swin-Transform and run

```
cd {tensorrtx}/Swin-Transform/
mkdir build
cd build
cp {microsoft}/Swin-Transformer/Swin-Transform.wts {tensorrtx}/swin-transformer/build
cmake ..
make
sudo ./swintransformer -s [.wts] [.engine]   // serialize model to plan file
sudo ./swintransformer -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.

```

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

