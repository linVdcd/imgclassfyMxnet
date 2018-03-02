"""
Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
Inception-v4, Inception-ResNet and the Impact of Residual Connections
on Learning
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
import mxnet as mx


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True):
    conv = mx.symbol.Convolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    if with_act:
        act = mx.symbol.Activation(
            data=bn, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return bn


def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = ConvFactory(net, 8, (1, 1))
    tower_conv1_0 = ConvFactory(net, 8, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 8, (3, 3), pad=(1, 1))
    tower_conv2_0 = ConvFactory(net, 8, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2_0, 12, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 16, (3, 3), pad=(1, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)

    net += scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = ConvFactory(net, 48, (1, 1))
    tower_conv1_0 = ConvFactory(net, 32, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 40, (1, 7), pad=(1, 2))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 48, (7, 1), pad=(2, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)
    net += scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = ConvFactory(net, 48, (1, 1))
    tower_conv1_0 = ConvFactory(net, 48, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 56, (1, 3), pad=(0, 1))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 64, (3, 1), pad=(1, 0))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)
    net += scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def repeat(inputs, repetitions, layer, *args, **kwargs):
    outputs = inputs
    for i in range(repetitions):
        outputs = layer(outputs, *args, **kwargs)
    return outputs


def get_symbol(num_classes=1000, **kwargs):
    data = mx.symbol.Variable(name='data')
    conv1a_3_3 = ConvFactory(data=data, num_filter=8,
                             kernel=(3, 3), stride=(2, 2))
    conv2a_3_3 = ConvFactory(conv1a_3_3, 8, (3, 3))
    conv2b_3_3 = ConvFactory(conv2a_3_3, 16, (3, 3), pad=(1, 1))
    maxpool3a_3_3 = mx.symbol.Pooling(
        data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
    conv3b_1_1 = ConvFactory(maxpool3a_3_3, 20, (1, 1))
    conv4a_3_3 = ConvFactory(conv3b_1_1, 48, (3, 3))
    maxpool5a_3_3 = mx.symbol.Pooling(
        data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')

    tower_conv = ConvFactory(maxpool5a_3_3, 24, (1, 1))
    tower_conv1_0 = ConvFactory(maxpool5a_3_3, 12, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 16, (5, 5), pad=(2, 2))

    tower_conv2_0 = ConvFactory(maxpool5a_3_3, 16, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2_0, 24, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 24, (3, 3), pad=(1, 1))

    tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    tower_conv3_1 = ConvFactory(tower_pool3_0, 16, (1, 1))
    tower_5b_out = mx.symbol.Concat(
        *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
    net = repeat(tower_5b_out, 10, block35, scale=0.17, input_num_channels=80)
    tower_conv = ConvFactory(net, 96, (3, 3), stride=(2, 2))
    tower_conv1_0 = ConvFactory(net, 64, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 64, (3, 3), pad=(1, 1))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 96, (3, 3), stride=(2, 2))
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])
    net = repeat(net, 5, block17, scale=0.1, input_num_channels=272)
    tower_conv = ConvFactory(net, 64, (1, 1))
    tower_conv0_1 = ConvFactory(tower_conv, 96, (3, 3), stride=(2, 2))
    tower_conv1 = ConvFactory(net, 64, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1, 72, (3, 3), stride=(2, 2))
    tower_conv2 = ConvFactory(net, 64, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2, 72, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 80, (3, 3),  stride=(2, 2))
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(
        *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])

    net = repeat(net, 3, block8, scale=0.2, input_num_channels=520)
    net = block8(net, with_act=False, input_num_channels=520)

    net = ConvFactory(net, 384, (1, 1))
    net = mx.symbol.Pooling(net, kernel=(
        1, 1), global_pool=True, stride=(2, 2), pool_type='avg')
    net = mx.symbol.Flatten(net)
    net = mx.symbol.Dropout(data=net, p=0.3)
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    #softmax =  mx.sym.Custom(op_type='FocalLoss', name='softmax', data=net, alpha=0.75, gamma=2)
    return softmax
