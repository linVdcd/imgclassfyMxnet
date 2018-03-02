import mxnet as mx
import cv2
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
imgSize = 224
sym, arg_params, aux_params = mx.model.load_checkpoint('model/irv2S_fjxgd', 64)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(2), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,imgSize,imgSize))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
from sklearn.metrics import confusion_matrix
labels = [l for l in range(mod.output_shapes[0][1][1])]
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
    img = cv2.resize(img, (imgSize, imgSize))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img,imgg

def predict(fname):

    img ,imgg= get_image(fname, show=False)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    p = 0
    l = -1

    return int(a[0])
c=0
ac = 0


#root ='/home/research/data/linmingan/crop_photo/1/'
#saveroot = '/home/research/data/linmingan/facePre/'
#photoroot = '/home/research/data/linmingan/photo/'


root ='/home/research/data/linmingan/fjxgd/'

lst = root+'fjxgd_val.lst'
crop = root+'test/'

y_ture=[]
y_pre=[]


f = open(lst,'r')
lines = f.readlines()
len1 = len(lines)

cc=0
pc=0
for line in lines:
    line = line.strip('\n').strip('\r')
    if line=='':
        continue
    line = line.split('\t')


    classe = int(float(line[1]))

    file = crop+line[-1]
    print file
    l = predict(file)
    cc+=1
    if classe==l:
        pc+=1
    y_ture.append(classe)
    y_pre.append(l)

print float(float(pc)/float(cc))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_ture, y_pre)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()