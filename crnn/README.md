# crnn

The Pytorch implementation is [meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch).

## How to Run

```
1. generate crnn.wts from pytorch

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/meijieru/crnn.pytorch.git
// download its weights 'crnn.pth'
// copy tensorrtx/crnn/genwts.py into crnn.pytorch/
// go to crnn.pytorch/
python genwts.py
// a file 'crnn.wts' will be generated.

2. build tensorrtx/crnn and run

// put crnn.wts into tensorrtx/crnn
// go to tensorrtx/crnn
mkdir build
cd build
cmake ..
make
sudo ./crnn -s  // serialize model to plan file i.e. 'crnn.engine'
// copy crnn.pytorch/data/demo.png here
sudo ./crnn -d  // deserialize plan file and run inference

3. check the output as follows:

raw: a-----v--a-i-l-a-bb-l-e---
sim: available

```

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

## Acknowledgment

Thanks for the donation for this crnn tensorrt implementation from @雍.

