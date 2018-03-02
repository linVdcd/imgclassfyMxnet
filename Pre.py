import mxnet as mx
import cv2
import os
h = 224 #change size acording to you net input size
w = 224
sym, arg_params, aux_params = mx.model.load_checkpoint('model/jc1_g', 844)#change model name acording to your model
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,w,h))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

labels = [l for l in [0,1,2,3,4,5,6]]
import matplotlib.pyplot as plt

import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(fname, show=False):
    # download and show the image
    img = cv2.imread(fname)
    imgg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (w, h))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img,imgg

def predict(fname):

    img ,imgg= get_image(fname, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]

    return a[0]


photo = '/home/research/data/linmingan/multiTask/crop/rz/' #crop photo path,photos to pre
save = '/home/research/data/linmingan/multiTask/crop/rzG/' #photo save path acording pre result

os.system('mkdir '+save)
for i in range(2):
    os.system('mkdir '+save+'/'+str(i))


for path,dir,files in os.walk(photo):
    if len(files)==0:
        continue
    for f in files:
        print f
        c = predict(path+'/'+f)
        src = path+'/'+f
        dts = save+'/'+str(c)+'/'+f.split('/')[-1]
        os.system('cp '+src+' '+dts)
