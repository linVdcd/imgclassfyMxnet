import mxnet as mx
def get_symbol(num_classes,**kwargs):
    rate=1
    data = mx.symbol.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=16*rate, pad=(1, 1), kernel=(3, 3), stride=(1, 1)
                                  )
    relu_conv1 = mx.symbol.Activation(name='relu_conv1', data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu_conv1, pad=(0, 0), kernel=(2, 2),
                              stride=(2, 2), pool_type='max')
    conv2 = mx.symbol.Convolution(name='conv2', data=pool1, num_filter=8*rate, pad=(0, 0), kernel=(1, 1), stride=(1, 1))
    relu_conv2 = mx.symbol.Activation(name='relu_conv2', data=conv2, act_type='relu')
    conv3 = mx.symbol.Convolution(name='conv3', data=relu_conv2, num_filter=16*rate, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
    relu_conv2 = mx.symbol.Activation(name='relu_conv3', data=conv3, act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu_conv2, pad=(0, 0), kernel=(2, 2),
                              stride=(2, 2), pool_type='max')

    conv4 = mx.symbol.Convolution(name='conv4', data=pool3, num_filter=8*rate, pad=(0, 0), kernel=(1, 1), stride=(1, 1))
    relu_conv4 = mx.symbol.Activation(name='relu_conv4', data=conv4, act_type='relu')
    conv5 = mx.symbol.Convolution(name='conv5', data=relu_conv4, num_filter=16*rate, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
    relu_conv5 = mx.symbol.Activation(name='relu_conv5', data=conv5, act_type='relu')
    pool5 = mx.symbol.Pooling(name='pool5', data=relu_conv5, pad=(0, 0), kernel=(2, 2),
                              stride=(2, 2), pool_type='max')
    conv6 = mx.symbol.Convolution(name='conv6',data=pool5,num_filter=16*rate,pad=(0, 0),kernel=(1, 1), stride=(1, 1))
    relu_conv6 = mx.symbol.Activation(name='relu_conv6', data=conv6, act_type='relu')
    conv7 = mx.symbol.Convolution(name='conv7', data=relu_conv6, num_filter=16*rate,pad=(1, 1), kernel=(3, 3), stride=(1, 1))
    relu_conv7 = mx.symbol.Activation(name='relu_conv7', data=conv7, act_type='relu')
    flat = mx.symbol.Flatten(data=relu_conv7)

    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')