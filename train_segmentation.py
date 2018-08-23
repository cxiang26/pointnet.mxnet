from __future__ import print_function
import argparse
import os
import random

from datasets import PartDataset
from pointnet import PointNetDenseCls

import mxnet as mx
from mxnet.gluon.data import DataLoader
from mxnet.gluon import Trainer, loss
import mxnet.autograd as ag


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
mx.random.seed(opt.manualSeed)

dataset = PartDataset(root = '/mnt/mdisk/xcq/shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'])
dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = '/mnt/mdisk/xcq/shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'], train = False)
testdataloader = DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'

ctx = mx.gpu()
classifier = PointNetDenseCls(k = num_classes)
classifier.initialize(ctx=ctx)

if opt.model != '':
    classifier.load_parameters(opt.model)

optimizer = Trainer(classifier.collect_params(), 'adam', {'learning_rate':0.01, })
L_loss = loss.SoftmaxCrossEntropyLoss(from_logits=True)

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data

        points = points.transpose((0,2,1))

        with ag.record():
            pred, _ = classifier(points.as_in_context(ctx))
            pred = pred.reshape((-1, num_classes))
            target = target.reshape((-1,1)) - 1
            #print(pred.shape, target.shape)
            loss = L_loss(pred, target.as_in_context(ctx))
        loss.backward()
        optimizer.step(opt.batchSize)
        pred_choice = pred.argmax(1)
        correct = (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.mean().asscalar(), correct.asscalar()/float(opt.batchSize * 2500)))
        
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose((0,2,1))


            pred, _ = classifier(points.as_in_context(ctx))
            pred = pred.reshape((-1, num_classes))
            target = target.reshape((-1,1)) - 1

            loss = L_loss(pred, target.as_in_context(ctx))
            pred_choice = pred.argmax(1)
            correct = (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.mean().asscalar(), correct.asscalar()/float(opt.batchSize * 2500)))
    
    classifier.save_parameters('%s/seg_model_%d.params' % (opt.outf, epoch))