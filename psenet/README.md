# PSENet

The Tensorflow implementation is [tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet).

<p align="center">

## How to Run

* 1. generate .wts

  Download pretrained model from https://github.com/liuheng92/tensorflow_PSENet
  and put `model.ckpt.*` to `model` dir. Add a file `model/checkpoint` with content
    ```
    model_checkpoint_path: "model.ckpt"
    all_model_checkpoint_paths: "model.ckpt"
    ```
    Then

    ```
    python gen_tf_wts.py
    ```
    which gengerate a `psenet.wts`.
* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  cp ../psenet.wts ./
  cp ../test.jpg ./
  ./psenet -s  // serialize model to plan file
  ./psenet -d  // deserialize plan file and run inference"
  ```

## Known issues
1. The output of network is not completely same as the tf's due to the difference between tensorrt's addResize and tf.image.resize, I will figure it out.

## Todo

* 1. use `ExponentialMovingAverage` weight.
* 2. faster preporcess and postprocess.