#!/bin/bash
echo "name dir resize"
echo "example: face path/to/crop 224"
python ~/tools/mxnet/tools/im2rec.py --list 1 --recursive 1 --train-ratio 0.9 $1 $2
python ~/tools/mxnet/tools/im2rec.py --resize $3 --num-thread 12 $1 $2
