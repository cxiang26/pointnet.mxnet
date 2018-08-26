from __future__ import print_function
import argparse
from datasets import PartDataset
from pointnet import PointNetCls

from mxnet.gluon.data import DataLoader
import mxnet as mx
from mxnet.gluon import loss

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')


opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0' , train = False, classification = True,  npoints = opt.num_points)

testdataloader = DataLoader(test_dataset, batch_size=32, shuffle = True)

ctx = mx.gpu()
classifier = PointNetCls(k = len(test_dataset.classes), num_points = opt.num_points)
classifier.load_parameters(opt.model, ctx=ctx)
L_loss = loss.SoftmaxCrossEntropyLoss(from_logits=True)


for i, data in enumerate(testdataloader, 0):
    points, target = data
    points = points.transpose((0,2, 1))
    pred, _ = classifier(points.as_in_context(ctx))
    loss = L_loss(pred, target)

    pred_choice = pred.argmax(1)
    correct = (target[:,0] == pred_choice.as_in_context(mx.cpu())).sum()
    print('i:%d  loss: %f accuracy: %f' %(i, loss.mean().asscaler(), correct.asscalar()/float(32)))
