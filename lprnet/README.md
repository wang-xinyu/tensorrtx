# LPRNet

The Pytorch implementation is [xuexingyu24/License_Plate_Detection_Pytorch](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch).

## Usage

1. download model from [HERE](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/blob/master/LPRNet/weights/Final_LPRNet_model.pth) and put it into `models` folder

2. use `genwts.py` to generate wts file

```bash
python3 genwts.py
```

3. build C++ code

```bash
pushd tensorrtx/lprnet
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

4. serialize wts model to engine file

```bash
./build/LPRnet -s
```

now you may see `LPRNet.engine` under `models`

5. run inference

sample code use the image under assets by default:

![sample](../assets/car_plate.jpg)

```bash
./build/LPRnet -d
```

output looks like:

```bash
...
Execution time: 205us
-65.58, -28.74, -52.1, -70.79, -53.36, -57.58, -70.97, -60.66, -48.18, -57.38, -54.07, -58.56, -49.04, -52.39, -51.94, -53.4, -49.04, -45.89, -49.42, -7.863, -42.12,
====
Execution time: 202us
-65.58, -28.74, -52.1, -70.79, -53.36, -57.58, -70.97, -60.66, -48.18, -57.38, -54.07, -58.56, -49.04, -52.39, -51.94, -53.4, -49.04, -45.89, -49.42, -7.863, -42.12,
====
result: 沪BKB770
```
