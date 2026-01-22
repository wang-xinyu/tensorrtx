# ConvNeXtV2 TensorRT

## Environment

- ubuntu20.04
-  cuda11.8
-  cudnn8.9.7
-  TensorRT8.6.1.6
-  OpenCV4.13

## Support

[ConvNext-V2](https://github.com/facebookresearch/ConvNeXt-V2.git)provides official pre-trained models such as ImageNet-1K fine-tuned models, ImageNet-22K fine-tuned models, and custom dataset classification models trained using these pre-trained weights.

## Build and Run

``````
# Downloda dependencies
pip install torch tensorrt pycuda numpy opencv-python

# Generate .wts
cd path-to-tensorrtx/convnextv2
python path-to-gen_wts.py path-to-pt path-to-wts

# Build convnextv2
cmake -B build
make -C build

# Update config.yaml to match your selected model

# Generate .engine
./build/convnextv2 path-to-wts path-to-engine

# Inference(python)
python path-to-inference.py path-to-engine path-to-your-image path-to-your-labels.txt

# Inference(cpp)
./build/inference_cpp path-to-engine path-to-your-image path-to-your-labels.txt
``````

## More Information

An interesting fact is that the suffix of the engine file can be arbitrarily specified; it does not need to be “engine”, and you can even use your own name as the suffix.
