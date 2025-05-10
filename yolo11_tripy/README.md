# YOLO11 Tripy

This example implements a YOLO11 classifier model using [Tripy](https://nvidia.github.io/TensorRT-Incubator/).

## Running The Example

Run the following commands from the [`yolo11_tripy`](./) directory:

1. Install Dependencies:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Download ImageNet classes file:

    ```bash
    wget https://raw.githubusercontent.com/joannzhang00/ImageNet-dataset-classes-labels/main/imagenet_classes.txt
    ```

3. [*Optional*] Download some images:

    ```bash
    wget https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01558993_robin.JPEG
    wget https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n04389033_tank.JPEG
    ```

    You can skip this step if you already have images you'd like to classify.

3. Build the model:

    ```bash
    python3 compile_classifier.py
    ```

    You can configure various aspects of the model when you compile.
    Run `python3 compile_classifier.py -h` for details.

4. Run inference:

    ```bash
    python3 classify.py n01558993_robin.JPEG n04389033_tank.JPEG
    ```

    The `classify.py` script allows you to pass one or more image file paths on the command line.
    The images are batched and classified in a single forward pass.
