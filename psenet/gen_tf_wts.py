from sys import prefix
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import struct

model_dir = "model"

ckpt = tf.train.get_checkpoint_state(model_dir)
ckpt_path = ckpt.model_checkpoint_path

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
param_dict = reader.get_variable_to_shape_map()


f = open(r"psenet.wts", "w")
keys = param_dict.keys()
f.write("{}\n".format(len(keys)))

for key in keys:
    weight = reader.get_tensor(key)
    print(key, weight.shape)
    if len(weight.shape) == 4:
        weight = np.transpose(weight, (3, 2, 0, 1))
        print(weight.shape)
    weight = np.reshape(weight, -1)
    f.write("{} {} ".format(key, len(weight)))
    for w in weight:
        f.write(" ")
        f.write(struct.pack(">f", float(w)).hex())
    f.write("\n")