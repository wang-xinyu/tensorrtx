# SuperPoint

The PyTorch implementation is from [magicleap/SuperPointPretrainedNetwork.](https://github.com/magicleap/SuperPointPretrainedNetwork)

The pretrained models are from [magicleap/SuperPointPretrainedNetwork.](https://github.com/magicleap/SuperPointPretrainedNetwork)


## Config

- FP16/FP32 can be selected by the macro `USE_FP16` in supernet.cpp
- GPU id and batch size can be selected by the macro `DEVICE` & `BATCH_SIZE` in supernet.cpp


## How to Run
1.Generate .wts file from the baseline pytorch implementation of pretrained model. The following example described how to generate superpoint_v1.wts from pytorch implementation of superpoint_v1. 
```
git clone https://github.com/xiang-wuu/SuperPointPretrainedNetwork
cd SuperPointPretrainedNetwork
git checkout deploy
// copy tensorrtx/superpoint/gen_wts.py to here(SuperPointPretrainedNetwork)
python gen_wts.py
// a file 'superpoint_v1.wts' will be generated.
// before running gen_wts.py python script make sure you cloned private fork and checkout to deploy branch.
```

2.Put .wts file into tensorrtx/superpoint, build and run
```
cd tensorrtx/superpoint
mkdir build
cd build
cmake ..
make
./supernet -s SuperPointPretrainedNetwork/superpoint_v1.wts    // serialize model to plan file i.e. 'supernet.engine'
```

## Run Demo using SuperPointPretrainedNetwork Python Script
The live demo can be run by inffering TensorRT generated engine file or by the pre-trained pytorch weight file , the `demo_superpoint.py` script is modified to infer automatically by either using TensorRT or PyTorch based on the provided input weight file.
```
cd SuperPointPretrainedNetwork
python demo_superpoint.py assets/nyu_snippet.mp4 --cuda --weights_path tensorrtx/superpoint/build/supernet.engine
// provide absolute path to supernet.engine as input weight file 
python demo_superpoint.py assets/nyu_snippet.mp4 --cuda --weights_path superpoint_v1.pth
// execute above command to infer using pytorch pre-trained weight files instead of tensorrt engine file.
```