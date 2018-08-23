from __future__ import print_function
from show3d_balls import *
import argparse
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetDenseCls
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = './seg/seg_model_24.params',  help='model path')
parser.add_argument('--idx', type=int, default = 11,   help='model index')



opt = parser.parse_args()
print (opt)

d = PartDataset(root = '/mnt/mdisk/xcq/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], train = False)

idx = opt.idx

print("model %d/%d" %( idx, len(d)))

point, seg = d[idx]
print(point.shape, seg.shape)

point_np = point.asnumpy()
ctx = mx.gpu()


cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]
gt = cmap[seg.asnumpy().astype(np.uint8) - 1, :]

classifier = PointNetDenseCls(k = 4)
classifier.load_parameters(opt.model, ctx=ctx)

point = nd.expand_dims(point.transpose((1,0)), axis=0)

pred, _ = classifier(point.as_in_context(ctx))
pred_choice = pred.argmax(2)
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.asnumpy().astype(np.uint8)[0], :]

#print(pred_color.shape)
showpoints(point_np, gt, pred_color)

