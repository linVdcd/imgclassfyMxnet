import mxnet as mx

def get_symbol(num_classes,**kwargs):
    data = mx.symbol.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=32, pad=(0, 0), kernel=(3, 3), stride=(1, 1),
                                  no_bias=False)
    relu_conv1 = mx.symbol.Activation(name='relu_conv1', data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu_conv1, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    fire2_squeeze1x1 = mx.symbol.Convolution(name='fire2_squeeze1x1', data=pool1, num_filter=16, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire2_relu_squeeze1x1 = mx.symbol.Activation(name='fire2_relu_squeeze1x1', data=fire2_squeeze1x1, act_type='relu')
    fire2_expand1x1 = mx.symbol.Convolution(name='fire2_expand1x1', data=fire2_relu_squeeze1x1, num_filter=48,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire2_relu_expand1x1 = mx.symbol.Activation(name='fire2_relu_expand1x1', data=fire2_expand1x1, act_type='relu')
    fire2_expand3x3 = mx.symbol.Convolution(name='fire2_expand3x3', data=fire2_relu_squeeze1x1, num_filter=48,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire2_relu_expand3x3 = mx.symbol.Activation(name='fire2_relu_expand3x3', data=fire2_expand3x3, act_type='relu')
    fire2_concat = mx.symbol.Concat(name='fire2_concat', *[fire2_relu_expand1x1, fire2_relu_expand3x3])
    pool2 = mx.symbol.Pooling(name='pool2', data=fire2_concat, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    fire4_squeeze1x1 = mx.symbol.Convolution(name='fire4_squeeze1x1', data=pool2, num_filter=32, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire4_relu_squeeze1x1 = mx.symbol.Activation(name='fire4_relu_squeeze1x1', data=fire4_squeeze1x1, act_type='relu')
    fire4_expand1x1 = mx.symbol.Convolution(name='fire4_expand1x1', data=fire4_relu_squeeze1x1, num_filter=64,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire4_relu_expand1x1 = mx.symbol.Activation(name='fire4_relu_expand1x1', data=fire4_expand1x1, act_type='relu')
    fire4_expand3x3 = mx.symbol.Convolution(name='fire4_expand3x3', data=fire4_relu_squeeze1x1, num_filter=64,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire4_relu_expand3x3 = mx.symbol.Activation(name='fire4_relu_expand3x3', data=fire4_expand3x3, act_type='relu')
    fire4_concat = mx.symbol.Concat(name='fire4_concat', *[fire4_relu_expand1x1, fire4_relu_expand3x3])
    fire5_squeeze1x1 = mx.symbol.Convolution(name='fire5_squeeze1x1', data=fire4_concat, num_filter=32, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire5_relu_squeeze1x1 = mx.symbol.Activation(name='fire5_relu_squeeze1x1', data=fire5_squeeze1x1, act_type='relu')
    fire5_expand1x1 = mx.symbol.Convolution(name='fire5_expand1x1', data=fire5_relu_squeeze1x1, num_filter=96,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire5_relu_expand1x1 = mx.symbol.Activation(name='fire5_relu_expand1x1', data=fire5_expand1x1, act_type='relu')
    fire5_expand3x3 = mx.symbol.Convolution(name='fire5_expand3x3', data=fire5_relu_squeeze1x1, num_filter=96,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire5_relu_expand3x3 = mx.symbol.Activation(name='fire5_relu_expand3x3', data=fire5_expand3x3, act_type='relu')
    fire5_concat = mx.symbol.Concat(name='fire5_concat', *[fire5_relu_expand1x1, fire5_relu_expand3x3])
    pool5 = mx.symbol.Pooling(name='pool5', data=fire5_concat, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    fire6_squeeze1x1 = mx.symbol.Convolution(name='fire6_squeeze1x1', data=pool5, num_filter=48, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire6_relu_squeeze1x1 = mx.symbol.Activation(name='fire6_relu_squeeze1x1', data=fire6_squeeze1x1, act_type='relu')
    fire6_expand1x1 = mx.symbol.Convolution(name='fire6_expand1x1', data=fire6_relu_squeeze1x1, num_filter=128,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire6_relu_expand1x1 = mx.symbol.Activation(name='fire6_relu_expand1x1', data=fire6_expand1x1, act_type='relu')
    fire6_expand3x3 = mx.symbol.Convolution(name='fire6_expand3x3', data=fire6_relu_squeeze1x1, num_filter=128,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire6_relu_expand3x3 = mx.symbol.Activation(name='fire6_relu_expand3x3', data=fire6_expand3x3, act_type='relu')
    fire6_concat = mx.symbol.Concat(name='fire6_concat', *[fire6_relu_expand1x1, fire6_relu_expand3x3])
    drop6 = mx.symbol.Dropout(name='drop6', data=fire6_concat, p=0.500000)
    conv10race = mx.symbol.Convolution(name='conv10race', data=drop6, num_filter=num_classes, pad=(0, 0), kernel=(1, 1),
                                       stride=(1, 1), no_bias=False)
    relu_conv10 = mx.symbol.Activation(name='relu_conv10', data=conv10race, act_type='relu')
    pool10 = mx.symbol.Pooling(name='pool10', data=relu_conv10, pooling_convention='full', global_pool=True,
                               kernel=(1, 1), pool_type='avg')
    flat = mx.sym.Flatten(data=pool10,name='flat')
    loss = mx.symbol.SoftmaxOutput(name='softmax', data=flat)



    return loss