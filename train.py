
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import  data, fit
import numpy as np
import random



import mxnet as mx

if __name__ == '__main__':


    #train_fname = '/home/research/data/linmingan/fjxgd/fjxgd_train.rec'
    #val_fname = '/home/research/data/linmingan/fjxgd/fjxgd_val.rec'
    task = 'fjxgd'
    netName = 'irv2S'
    root ='/home/research/data/linmingan/fjxgd/'
    train_fname = root+task+'_train.rec'
    val_fname = root+task+'_val.rec'
    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser,3)

    parser.set_defaults(
        # network
        model_prefix='model/'+netName+'_'+task,
        network        = netName,
        num_layers     = 50,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 4,
        num_examples  = 32000,
        image_shape    = '3,224,224',
        # train
        batch_size     = 64,
        num_epochs     = 32000,
        lr             = .05,
        lr_factor     = 0.1,
        lr_step_epochs = '2',
        gpus='2',
        load_epoch=None
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))
    #sym = nn.get_symbol(args.num_classes)
    #mx.viz.plot_network(sym, shape={"data": (64, 3, 79, 79)}).view()// Draw net
    # train
    fit.fit(args, sym, data.get_rec_iter)
