# yolov1

The Pytorch implementation is [yolov1](https://github.com/ivanwhaf/yolov1-pytorch). It provides a trained weights of yolov1 in releases.

# Execute

1. generate wts file and bin file.
Because YOLOv1 ends with two fully connected layer, the wts file becomes extremely large. Reading the file using Python will cause an OutOfMemoryError(OOM).Therefore, we save the parameters of each layer in binary form, which is the bin file.
```
1. clone the YOLOv1 project.
git clone https://github.com/ivanwhaf/yolov1-pytorch.git

2. download the yolov1.pth file from the releases and move the pth file to the weights folder.

3. move gen_wts_bin_files.py to the YOLOv1 project.
python gen_wts_bin_files.py cfg/yolov1.yaml weights/yolov1.pth
The command will gene yolov1.wts and yolov1_bin folder which contain all of the bin files and weight_name_and_shape.txt.

```

2. create a models folder and move all of the wts and bin and weight_name_and_shape.txt files to this folder.

3. run the yolov1.py file.(Python API)
```
python yolov1.py -s # serialize the model.

# move a test.jpg file to yolov1 folder.
python yolov1.py -d # deserialize the model and inference the image.

```

4. C++ API
```
mkdir build
cd build
cmake ..
make
./yolov1 -s // serialize the model.
./yolov1 -d // deserialize the model and run the inference.

```

5. check the image generated.

# Config
- FP16/FP32 can be selected by the macro in yolov1.cpp

# INT8 Quantization

1. Prepare calibration images, you can randomly select images from your train set. For coco, you can also download coco 2007 images `coco2007` from
 [kaggle](https://www.kaggle.com/datasets/zaraks/pascal-voc-2007).

2. unzip it in yolov1/build.

3. select about 4000 images from train dataset and make a folder named `coco2007_calib`

4. set the macro `USE_INT8` in yolov1.cpp and make.

5. serialize the model and test.
