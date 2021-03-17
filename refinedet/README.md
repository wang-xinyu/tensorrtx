# RefineDet

For the Pytorch implementation, you can refer to [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch)

## How to run
```
1. generate wts file. from pytorch
python gen_wts_refinedet.py
// a file 'refinedet.wts' will be generated.

2. build tensorrtx/RefineDet and run or Using clion to open a project(recommend)
Configuration file in configure.h
You need configure your own paths and modes(SERIALIZE or INFER)
Detailed information reference configure.h
mkdir build
cd build
cmake ..
make
```

## dependence
```
TensorRT7.0.0.11 
OpenCV >= 3.4
libtorch >=1.1.0
```


## feature
1.tensorrt Multi output  
2.L2norm  
3.Postprocessing with libtorch


## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)  
[tensorrt tutorials](https://github.com/wang-xinyu/tensorrtx/tree/master/tutorials)  
For more detailed guidance, see [yhl blog](https://www.cnblogs.com/yanghailin/p/14525128.html)
