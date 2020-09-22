# Ultra-Fast-Lane-Detection(UFLD)

The Pytorch implementation is [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection).

## How to Run
```
1. generate lane.wts and lane.onnx from pytorch with tusimple_18.pth

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
// download its weights 'tusimple_18.pth'
// copy tensorrtx/ufld/gen_wts.py into Ultra-Fast-Lane-Detection/
// ensure the file name is tusimple_18.pth and lane.wts in gen_wts.py
// go to Ultra-Fast-Lane-Detection
python gen_wts.py
// a file 'lane.wts' will be generated.
// then ( not necessary )
python pth2onnx.py
//a file 'lane.onnx' will be generated.

2. build tensorrtx/ufld and run

mkdir build
cd build
cmake ..
make
sudo ./lane_det -s          // serialize model to plan file i.e. 'lane.engine'
sudo ./lane_det -d  PATH_TO_YOUR_IMAGE_FOLDER // deserialize plan file and run inference, the images will be processed.

```

## More Information
1. Changed the preprocess and postprocess in tensorrtx, give a different way to convert NHWC to NCHW in preprocess and just show the result using opencv rather than saving the result in postprocess.
2. If there are some bugs where you inference with multi batch_size, just modify the code in preprocess or postprocess, it's not complicated.
3. Some results are stored in resluts folder.
