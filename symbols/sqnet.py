import mxnet as mx


def squeeze(data, num_filter, name, kernel=(1, 1), stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}):
    squeeze_1x1 = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                        name=name)
    act = mx.symbol.Activation(data=squeeze_1x1, act_type=act_type, attr=mirror_attr, name=name + "_relu")
    return act


def Fire_module(data, num_filter_squeeze, num_filter_fire, fire_num, kernel_sequeeze=(1, 1), kernel_1x1=(1, 1),
                kernel_3x3=(3, 3), stride_squeeze=(1, 1), stride_1x1=(1, 1), stride_3x3=(1, 1),
                pad_1x1=(0, 0), pad_3x3=(1, 1), act_type="relu", mirror_attr={}):
    squeeze_1x1 = squeeze(data, num_filter_squeeze, "fire%d_squeeze1x1" % fire_num, kernel_sequeeze, stride_squeeze, )
    expand1x1 = mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_1x1,
                                      stride=stride_1x1, pad=pad_1x1, name="fire%d_expand1x1" % fire_num)
    relu_expand1x1 = mx.symbol.Activation(data=expand1x1, act_type=act_type, attr=mirror_attr,
                                          name="fire%d_expand1x1_relu" % fire_num)

    expand3x3 = mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_3x3,
                                      stride=stride_3x3, pad=pad_3x3, name="fire%d_expand3x3" % fire_num)
    relu_expand3x3 = mx.symbol.Activation(data=expand3x3, act_type=act_type, attr=mirror_attr,
                                          name="fire%d_expand3x3_relu" % fire_num)
    # return relu_expand1x1+relu_expand3x3
    return mx.symbol.Concat(relu_expand1x1, relu_expand3x3, dim=1)


def SqueezeNetV1_0(data, num_classes):
    """
    data (batch_size, 3, 227, 227)
    """
    conv1 = mx.symbol.Convolution(data=data, num_filter=96, kernel=(7, 7), stride=(2, 2), pad=(0, 0), name="conv1")
    relu_conv1 = mx.symbol.Activation(data=conv1, act_type="relu", attr={})
    pool_conv1 = mx.symbol.Pooling(data=relu_conv1, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire2 = Fire_module(pool_conv1, num_filter_squeeze=16, num_filter_fire=64, fire_num=2)
    fire3 = Fire_module(fire2, num_filter_squeeze=16, num_filter_fire=64, fire_num=3)
    fire4 = Fire_module(fire3, num_filter_squeeze=32, num_filter_fire=128, fire_num=4)

    pool4 = mx.symbol.Pooling(data=fire4, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})
    fire5 = Fire_module(pool4, num_filter_squeeze=32, num_filter_fire=128, fire_num=5)
    fire6 = Fire_module(fire5, num_filter_squeeze=48, num_filter_fire=192, fire_num=6)
    fire7 = Fire_module(fire6, num_filter_squeeze=48, num_filter_fire=192, fire_num=7)
    fire8 = Fire_module(fire7, num_filter_squeeze=64, num_filter_fire=256, fire_num=8)
    pool8 = mx.symbol.Pooling(data=fire8, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})
    fire9 = Fire_module(pool8, num_filter_squeeze=64, num_filter_fire=256, fire_num=9)
    drop9 = mx.sym.Dropout(data=fire9, p=0.5)
    conv10 = mx.symbol.Convolution(data=drop9, num_filter=num_classes, kernel=(1, 1), stride=(1, 1))
    relu_conv10 = mx.symbol.Activation(data=conv10, act_type="relu", attr={})
    pool10 = mx.symbol.Pooling(data=relu_conv10, kernel=(13, 13), pool_type='avg', attr={})

    flatten = mx.symbol.Flatten(data=pool10, name='flatten')
    softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')
    return softmax


def SqueezeNetV1_1(data, num_classes):
    """
    data (batch_size, 3, 224, 224)
    """
    conv1 = mx.symbol.Convolution(data=data, num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name="conv1")
    relu_conv1 = mx.symbol.Activation(data=conv1, act_type="relu", attr={})
    pool_conv1 = mx.symbol.Pooling(data=relu_conv1, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire2 = Fire_module(pool_conv1, num_filter_squeeze=16, num_filter_fire=64, fire_num=2)
    fire3 = Fire_module(fire2, num_filter_squeeze=16, num_filter_fire=64, fire_num=3)
    pool3 = mx.symbol.Pooling(data=fire3, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire4 = Fire_module(pool3, num_filter_squeeze=32, num_filter_fire=128, fire_num=4)
    fire5 = Fire_module(fire4, num_filter_squeeze=32, num_filter_fire=128, fire_num=5)
    pool5 = mx.symbol.Pooling(data=fire5, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire6 = Fire_module(pool5, num_filter_squeeze=48, num_filter_fire=192, fire_num=6)
    fire7 = Fire_module(fire6, num_filter_squeeze=48, num_filter_fire=192, fire_num=7)
    fire8 = Fire_module(fire7, num_filter_squeeze=64, num_filter_fire=256, fire_num=8)
    fire9 = Fire_module(fire8, num_filter_squeeze=64, num_filter_fire=256, fire_num=9)

    drop9 = mx.sym.Dropout(data=fire9, p=0.5)
    conv10 = mx.symbol.Convolution(data=drop9, num_filter=1024, kernel=(1, 1), stride=(1, 1), name="conv10")
    relu_conv10 = mx.symbol.Activation(data=conv10, act_type="relu", attr={})

    pool10 = mx.symbol.Pooling(data=relu_conv10, kernel=(13, 13), pool_type='avg',global_pool=True, attr={})

    flatten = mx.symbol.Flatten(data=pool10, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')

    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax


def get_symbol(num_classes=10,**kwargs):
    net = SqueezeNetV1_1(mx.symbol.Variable(name='data'), num_classes)
    return net


if __name__ == "__main__":
    import pprint

    sqnet = get_symbol(2)
    # sqnet = SqueezeNetV1_0(mx.symbol.Variable(name='data'),2)
    pprint.pprint(sqnet.list_arguments())
    pprint.pprint(sqnet.infer_shape(**{'data': (1, 3, 224, 224)}))