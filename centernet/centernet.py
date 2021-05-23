import numpy as np

import tensorrt as trt
import torch

from sample import common
import argparse
import time

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    if plugin_creator.name == 'DCNv2_TRT':
        dcnCreator = plugin_creator


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 512, 512)
    OUTPUT_NAME = "prob"
    DTYPE = trt.float16


class Centernet_dla34(object):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights
        self.levels = [1, 1, 1, 2, 2, 1]
        self.channels = [16, 32, 64, 128, 256, 512]
        self.down_ratio = 4
        self.last_level = 5
        self.engine = self.build_engine()

    def add_batchnorm_2d(self, input_tensor, parent):
        gamma = self.weights[parent + '.weight'].numpy()
        beta = self.weights[parent + '.bias'].numpy()
        mean = self.weights[parent + '.running_mean'].numpy()
        var = self.weights[parent + '.running_var'].numpy()
        eps = 1e-5

        scale = gamma / np.sqrt(var + eps)
        shift = beta - mean * gamma / np.sqrt(var + eps)
        power = np.ones_like(scale)

        return self.network.add_scale(input=input_tensor.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale, power=power)

    def add_basic_block(self, input_tensor, out_channels, residual=None, stride=1, dilation=1, parent=''):
        conv1_w = self.weights[parent + '.conv1.weight'].numpy()
        conv1 = self.network.add_convolution(input=input_tensor.get_output(
            0), num_output_maps=out_channels, kernel_shape=(3, 3), kernel=conv1_w)
        conv1.stride = (stride, stride)
        conv1.padding = (dilation, dilation)
        conv1.dilation = (dilation, dilation)

        bn1 = self.add_batchnorm_2d(conv1, parent + '.bn1')
        ac1 = self.network.add_activation(
            input=bn1.get_output(0), type=trt.ActivationType.RELU)

        conv2_w = self.weights[parent + '.conv2.weight'].numpy()
        conv2 = self.network.add_convolution(input=ac1.get_output(
            0), num_output_maps=out_channels, kernel_shape=(3, 3), kernel=conv2_w)
        conv2.padding = (dilation, dilation)
        conv2.dilation = (dilation, dilation)

        out = self.add_batchnorm_2d(conv2, parent + '.bn2')

        if residual is None:
            out = self.network.add_elementwise(input_tensor.get_output(
                0), out.get_output(0), trt.ElementWiseOperation.SUM)
        else:
            out = self.network.add_elementwise(residual.get_output(
                0), out.get_output(0), trt.ElementWiseOperation.SUM)
        return self.network.add_activation(input=out.get_output(0), type=trt.ActivationType.RELU)

    def add_level(self, input_tensor, out_channels, stride=1, dilation=1, parent=''):
        conv1_w = self.weights[parent + '.0.weight'].numpy()
        conv1 = self.network.add_convolution(input=input_tensor.get_output(
            0), num_output_maps=out_channels, kernel_shape=(3, 3), kernel=conv1_w)
        conv1.stride = (stride, stride)
        conv1.padding = (dilation, dilation)
        conv1.dilation = (dilation, dilation)

        bn1 = self.add_batchnorm_2d(conv1, parent + '.1')
        ac1 = self.network.add_activation(
            input=bn1.get_output(0), type=trt.ActivationType.RELU)
        return ac1

    def add_root(self, input_tensors: list, out_channels, kernel_size=1, residual=False, parent=''):
        ct = self.network.add_concatenation(
            [x.get_output(0) for x in input_tensors])

        conv_w = self.weights[parent + '.conv.weight'].numpy()
        conv = self.network.add_convolution(input=ct.get_output(
            0), num_output_maps=out_channels, kernel_shape=(1, 1), kernel=conv_w)
        conv.padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)

        bn1 = self.add_batchnorm_2d(conv, parent + '.bn')
        out = self.network.add_activation(
            input=bn1.get_output(0), type=trt.ActivationType.RELU)

        if residual:
            out = self.network.add_elementwise(input_tensors[0].get_output(
                0), out.get_output(0), trt.ElementWiseOperation.SUM)

        return self.network.add_activation(input=out.get_output(0), type=trt.ActivationType.RELU)

    def add_tree(self, input_tensor, level, out_channels, residual=None, children=None, stride=1, level_root=False, parent=''):
        children = [] if children is None else children
        if stride > 1:
            bottom = self.network.add_pooling(input_tensor.get_output(
                0), trt.PoolingType.MAX, (stride, stride))
            bottom.stride = (stride, stride)
        else:
            bottom = input_tensor

        if input_tensor.get_output(0).shape[0] != out_channels:
            project_conv1_w = self.weights[parent +
                                           '.project.0.weight'].numpy()
            project_conv1 = self.network.add_convolution(input=bottom.get_output(
                0), num_output_maps=out_channels, kernel_shape=(1, 1), kernel=project_conv1_w)
            residual = self.add_batchnorm_2d(
                project_conv1, parent + '.project.1')
        else:
            residual = bottom

        if level_root:
            children.append(bottom)

        if level == 1:
            tree1 = self.add_basic_block(
                input_tensor, out_channels, residual, stride, parent=parent+'.tree1')
            tree2 = self.add_basic_block(
                tree1, out_channels, parent=parent+'.tree2')
            return self.add_root([tree2, tree1]+children, out_channels, parent=parent+'.root')
        else:
            tree1 = self.add_tree(input_tensor, level-1, out_channels,
                                  residual, stride=stride, parent=parent+'.tree1')
            children.append(tree1)
            return self.add_tree(tree1, level-1, out_channels, children=children, parent=parent+'.tree2')

    def add_base(self, input_tensor, parent):
        base_conv1_w = self.weights[parent+'.base_layer.0.weight'].numpy()
        base_conv1 = self.network.add_convolution(
            input=input_tensor, num_output_maps=self.channels[0], kernel_shape=(7, 7), kernel=base_conv1_w)
        base_conv1.padding = (3, 3)

        base_bn1 = self.add_batchnorm_2d(base_conv1, parent+'.base_layer.1')
        base_ac1 = self.network.add_activation(
            input=base_bn1.get_output(0), type=trt.ActivationType.RELU)

        level0 = self.add_level(
            base_ac1, self.channels[0],    parent=parent+'.level0')
        level1 = self.add_level(
            level0,   self.channels[1], 2, parent=parent+'.level1')

        level2 = self.add_tree(
            level1, self.levels[2], self.channels[2], stride=2, level_root=False, parent=parent+'.level2')
        level3 = self.add_tree(
            level2, self.levels[3], self.channels[3], stride=2, level_root=True, parent=parent+'.level3')
        level4 = self.add_tree(
            level3, self.levels[4], self.channels[4], stride=2, level_root=True, parent=parent+'.level4')
        level5 = self.add_tree(
            level4, self.levels[5], self.channels[5], stride=2, level_root=True, parent=parent+'.level5')

        return [level0, level1, level2, level3, level4, level5]

    def add_deform_conv(self, input_tensor, out_channels, kernel=3, stride=1, padding=1, dilation=1, deformable_group=1, parent=''):
        conv_offset_mask_w = self.weights[parent +
                                          '.conv.conv_offset_mask.weight'].numpy()
        conv_offset_mask_b = self.weights[parent +
                                          '.conv.conv_offset_mask.bias'].numpy()
        conv_offset_mask = self.network.add_convolution(input=input_tensor.get_output(0),
                                                        num_output_maps=deformable_group*3*kernel*kernel,
                                                        kernel_shape=(
                                                            kernel, kernel),
                                                        kernel=conv_offset_mask_w,
                                                        bias=conv_offset_mask_b)
        conv_offset_mask.stride = (stride, stride)
        conv_offset_mask.padding = (padding, padding)

        out_channels = trt.PluginField("out_channels", np.array(
            [out_channels], dtype=np.int32), trt.PluginFieldType.INT32)
        kernel = trt.PluginField("kernel", np.array(
            [kernel], dtype=np.int32), trt.PluginFieldType.INT32)
        deformable_group = trt.PluginField("deformable_group", np.array(
            [deformable_group], dtype=np.int32), trt.PluginFieldType.INT32)
        dilation = trt.PluginField("dilation", np.array(
            [dilation], dtype=np.int32), trt.PluginFieldType.INT32)
        padding = trt.PluginField("padding", np.array(
            [padding], dtype=np.int32), trt.PluginFieldType.INT32)
        stride = trt.PluginField("stride", np.array(
            [stride], dtype=np.int32), trt.PluginFieldType.INT32)
        weight = trt.PluginField(
            "weight", self.weights[parent + '.conv.weight'].numpy(), trt.PluginFieldType.FLOAT32)
        bias = trt.PluginField(
            "bias", self.weights[parent + '.conv.bias'].numpy(), trt.PluginFieldType.FLOAT32)
        field_collection = trt.PluginFieldCollection(
            [out_channels, kernel, deformable_group, dilation, padding, stride, weight, bias])
        DCN = dcnCreator.create_plugin(
            name='DCNv2_TRT', field_collection=field_collection)

        sigmoid_conv_offset_mask = self.network.add_activation(
            input=conv_offset_mask.get_output(0), type=trt.ActivationType.SIGMOID)

        dcn = self.network.add_plugin_v2(inputs=[input_tensor.get_output(
            0), conv_offset_mask.get_output(0), sigmoid_conv_offset_mask.get_output(0)], plugin=DCN)
        bn = self.add_batchnorm_2d(dcn, parent+'.actf.0')
        return self.network.add_activation(input=bn.get_output(0), type=trt.ActivationType.RELU)

    def add_ida_up(self, input_tensors, out_channels, up_f, startp, parent):
        for i in range(startp + 1, len(input_tensors)):
            proj = self.add_deform_conv(
                input_tensors[i], out_channels, parent=parent+'.proj_%d' % (i-startp))
            f = up_f[i-startp]
            up_w = self.weights[parent + '.up_%d.weight' % (i-startp)].numpy()
            up = self.network.add_deconvolution(
                proj.get_output(0), out_channels, (f*2, f*2), up_w)
            up.stride = (f, f)
            up.padding = (f//2, f//2)
            up.num_groups = out_channels
            node = self.network.add_elementwise(
                input_tensors[i-1].get_output(0), up.get_output(0), trt.ElementWiseOperation.SUM)
            input_tensors[i] = self.add_deform_conv(
                node, out_channels, parent=parent+'.node_%d' % (i-startp))
        return input_tensors

    def add_dla_up(self, input_tensors, first_level, parent):
        channels = self.channels[first_level:]
        scales = [2 ** i for i in range(len(self.channels[first_level:]))]
        scales = np.array(scales, dtype=int)
        out = [input_tensors[-1]]
        for i in range(len(channels) - 1):
            j = -i - 2
            input_tensors = self.add_ida_up(
                input_tensors, channels[j], scales[j:] // scales[j], len(input_tensors) - i - 2, parent+'.ida_%d' % i)
            out.insert(0, input_tensors[-1])
            scales[j + 1:] = scales[j]
            channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
        return out

    def add_head(self, input_tensor, out_channels, head, head_conv=256, final_kernal=1):
        conv1_w = self.weights[head+'.0.weight'].numpy()
        conv1_b = self.weights[head+'.0.bias'].numpy()
        conv1 = self.network.add_convolution(
            input_tensor.get_output(0), head_conv, (3, 3), conv1_w, conv1_b)
        conv1.padding = (1, 1)
        ac1 = self.network.add_activation(
            input=conv1.get_output(0), type=trt.ActivationType.RELU)
        conv2_w = self.weights[head + '.2.weight'].numpy()
        conv2_b = self.weights[head+'.2.bias'].numpy()
        conv2 = self.network.add_convolution(ac1.get_output(
            0), out_channels, (final_kernal, final_kernal), conv2_w, conv2_b)
        return conv2

    def populate_network(self):
        # Configure the network layers based on the self.weights provided.
        input_tensor = self.network.add_input(
            name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

        y = self.add_base(input_tensor, 'module.base')

        first_level = int(np.log2(self.down_ratio))
        last_level = self.last_level
        dla_up = self.add_dla_up(y, first_level, 'module.dla_up')
        ida_up = self.add_ida_up(dla_up[:last_level-first_level], self.channels[first_level], [
                                 2 ** i for i in range(last_level - first_level)], 0, 'module.ida_up')

        hm = self.add_head(ida_up[-1], 80, 'module.hm')
        wh = self.add_head(ida_up[-1], 2, 'module.wh')
        reg = self.add_head(ida_up[-1], 2, 'module.reg')

        hm.get_output(0).name = 'hm'
        wh.get_output(0).name = 'wh'
        reg.get_output(0).name = 'reg'
        self.network.mark_output(tensor=hm.get_output(0))
        self.network.mark_output(tensor=wh.get_output(0))
        self.network.mark_output(tensor=reg.get_output(0))

    def build_engine(self):
        # For more information on TRT basics, refer to the introductory samples.
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
            self.network = network
            builder.max_workspace_size = common.GiB(1)
            builder.max_batch_size = 1
            # Populate the network using self.weights from the PyTorch model.
            self.populate_network()
            # Build and return an engine.
            return builder.build_cuda_engine(self.network)


def load_random_test_case(pagelocked_buffer):
    # Select an image at random to be the test case.
    img = np.random.randn(1, 3, 512, 512).astype(np.float32)
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img.ravel())
    return img


def main(args):
    # Get the PyTorch weights
    weights = torch.load(args.model, map_location={
                         'cuda:0': 'cpu'})['state_dict']
    # Do inference with TensorRT.
    with Centernet_dla34(weights).engine as engine:
        if args.save_engine:
            with open('centernet.engine', "wb") as f:
                f.write(engine.serialize())
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            img = load_random_test_case(pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            t = time.time()
            [hm, wh, reg] = common.do_inference(
                context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
            t = time.time() - t
            print('output:   hm:%f, wh:%f, reg:%f' %
                  (hm.mean(), wh.mean(), reg.mean()))
            print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CenterNet dla34 ctdet')
    parser.add_argument('--model',  '-m', type=str,
                        default='./ctdet_coco_dla_2x.pth', help='path of pytorch .pth')
    parser.add_argument('--save_engine', '-s',
                        action='store_true', help='if save trt engine')
    args = parser.parse_args()
    main(args)
