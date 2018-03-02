import mxnet as mx

def blod(data,filter_num,name):
    bn = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    act = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu')
    conv = mx.sym.Convolution(data=act, num_filter=filter_num, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=256, name=name + '_conv')

    bn1 = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=filter_num, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=256, name=name + '_conv1')

    bn2 = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=filter_num/4, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=256,name=name + '_conv2')
    cc = mx.symbol.Concat(*[conv1,conv2])
    bn3 = mx.sym.BatchNorm(data=cc, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=filter_num, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=256, name=name + '_conv3')


    return conv3+conv


def get_symbol(num_classes,):
    data = mx.sym.Variable(name='data')
    # Nr = mx.nd.random.normal(loc=0,scale=1,shape = image_shape)

    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_data')
    c1 = mx.sym.Convolution(data=data, num_filter=16, kernel=(7, 7), stride=(3,3), pad=(0, 0),
                               no_bias=True, workspace=256, name='conv1')



    p1 = mx.symbol.Pooling(data=c1, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max',name='p1')

    b2 = blod(data=p1,filter_num=16,name="b2")
    b3 = blod(data=b2,filter_num=32,name="b3")

    p2 = mx.symbol.Pooling(data=b3, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max',name='p2')

    b4 = blod(data=p2,filter_num=32,name="b4")
    b5 = blod(data=b4,filter_num=64,name="b5")

    p3 = mx.symbol.Pooling(data=b5, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max',name='p3')

    b6 = blod(data=p3,filter_num=64,name='b6')

    p4 = mx.symbol.Pooling(data=b6,kernel=(2,2),stride = (2,2),pad=(1,1),pool_type='max',name='p4')

    b7 = blod(data=p4, filter_num=128, name='b7')

    p5 = mx.symbol.Pooling(data=b7, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max',name='p5')

    b8 = blod(data=p5, filter_num=128, name='b8')

    p6 = mx.symbol.Pooling(data=b8, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max',name='p6')

    b9 = blod(data=p6, filter_num=64, name='b9')
    bn9 = mx.sym.BatchNorm(data=b9, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn9')
    act9 = mx.sym.Activation(data=bn9, act_type='relu', name='relu9')
    #p7 = mx.symbol.Pooling(data=act9, global_pool=True, kernel=(7, 7), pool_type='avg', name='p7')
    flat = mx.symbol.Flatten(data=act9)
    fd = mx.symbol.Dropout(data=flat,p=0.5)
    fc1 = mx.symbol.FullyConnected(data=fd, num_hidden=num_classes, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')