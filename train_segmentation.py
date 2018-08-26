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
import datetime
import logging
import sys

logger = logging.getLogger("Poitnet")
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
file_handler = logging.FileHandler(filename='./logs/seg_%s.log'%nowTime,mode='w')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


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

optimizer = Trainer(classifier.collect_params(), 'sgd', {'learning_rate':0.01, 'momentum':0.9})
L_loss = loss.SoftmaxCrossEntropyLoss(from_logits=True)

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    correct = 0.
    L = 0.
    count = 0
    L_eval = 0.
    correct_eval = 0.
    count_eval = 0

    ## training
    for data in dataloader:
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
        correct += (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum().asscalar()/target.shape[0]
        L += loss.mean().asscalar()
        count += 1

    for data in testdataloader:
        points, target = data
        points = points.transpose((0,2,1))
        pred, _ = classifier(points.as_in_context(ctx))
        pred = pred.reshape((-1, num_classes))
        target = target.reshape((-1,1)) - 1
        loss = L_loss(pred, target.as_in_context(ctx))
        pred_choice = pred.argmax(1)
        correct_eval += (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum().asscalar()/target.shape[0]
        L_eval += loss.mean().asscalar()
        count_eval += 1
    logger.info('[epoch: %d] train loss: %f | test loss: %f | train_acc: %f | test_acc: %f'
                % (epoch, L / count, L_eval / count_eval, correct / count,
                   correct_eval / count_eval))
    
    classifier.save_parameters('%s/seg_model_%d.params' % (opt.outf, epoch))