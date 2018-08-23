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
from pointnet import PointNetCls



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

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


classifier = PointNetCls(k = num_classes, num_points = opt.num_points)
ctx = mx.gpu()
classifier.initialize(ctx=ctx)

if opt.model != '':
    classifier.load_parameters(opt.model)
    # classifier.load_state_dict(torch.load(opt.model))


# optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
optimizer = Trainer(params=classifier.collect_params(), optimizer='sgd', optimizer_params={'learning_rate':0.01,'momentum':0.9})
L_loss = gluon.loss.SoftmaxCrossEntropyLoss()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader):
        points, target = data
        points = points.transpose((0,2,1))
        with ag.record():
            pred, _ = classifier(points.as_in_context(ctx))
            loss = L_loss(pred, target.as_in_context(ctx))
        loss.backward()
        optimizer.step(batch_size=opt.batchSize)
        pred_choice = pred.argmax(1)
        correct = (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.mean().asscalar(),correct.asscalar() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose((0,2,1))
            pred, _ = classifier(points.as_in_context(ctx))
            loss = L_loss(pred, target.as_in_context(ctx))
            pred_choice = pred.argmax(1)
            correct = (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.mean().asscalar(), correct.asscalar()/float(opt.batchSize)))

    classifier.save_parameters('%s/cls_model_%d.params' % (opt.outf, epoch))
