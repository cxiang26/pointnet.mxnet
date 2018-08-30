from __future__ import print_function
import argparse
import os
import random
import mxnet as mx
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet import gluon, nd
from mxnet import autograd as ag
from datasets import PartDataset
# from pointnet import PointNetCls
from models.pointnet_cls import PointNetCls
import datetime
import logging
import sys

logger = logging.getLogger("Poitnet")
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
file_handler = logging.FileHandler(filename='./logs/cls_%s.log'%nowTime,mode='w')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print(opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
mx.random.seed(opt.manualSeed)

dataset = PartDataset(root = '/mnt/mdisk/xcq/shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = '/mnt/mdisk/xcq/shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
testdataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls(k = num_classes, num_points = opt.num_points, routing=None)
ctx = mx.gpu(1)
classifier.initialize(ctx=ctx)

if opt.model != '':
    classifier.load_parameters(opt.model)

optimizer = Trainer(params=classifier.collect_params(), optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum':0.9})
L_loss = gluon.loss.SoftmaxCrossEntropyLoss()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    correct = 0.
    L = 0.
    count = 0
    L_eval = 0.
    correct_eval = 0.
    count_eval = 0

    ## training
    for i, data in enumerate(dataloader):
        points, target = data
        points = points.transpose((0,2,1))
        with ag.record():
            pred, _ = classifier(points.as_in_context(ctx))
            loss = L_loss(pred, target.as_in_context(ctx))
        loss.backward()
        optimizer.step(batch_size=opt.batchSize)
        pred_choice = pred.argmax(1)
        correct += (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum().asscalar()
        L += loss.mean().asscalar()
        count += 1
    # logger.info('[epoch: %d] train loss: %f accuracy: %f' %(epoch, L / count, correct/ float(len(dataset))))


    ## evaluating
    for j, data in enumerate(testdataloader):
        points, target = data
        points = points.transpose((0,2,1))
        pred, _ = classifier(points.as_in_context(ctx))
        loss = L_loss(pred, target.as_in_context(ctx))
        pred_choice = pred.argmax(1)
        correct_eval += (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum().asscalar()
        L_eval += loss.mean().asscalar()
        count_eval += 1
    logger.info('[epoch: %d] train loss: %f | test loss: %f | train_acc: %f | test_acc: %f'
                %(epoch, L/count, L_eval/count_eval, correct/ float(len(dataset)), correct_eval/float(len(test_dataset))))

    classifier.save_parameters('%s/cls_model_%d.params' % (opt.outf, epoch))
