from __future__ import print_function

from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx

def Squash(vector, axis):
    norm = nd.sum(nd.square(vector), axis, keepdims=True)
    v_j = norm/(1+norm)/nd.sqrt(norm, keepdims=True)*vector
    return v_j

class STN3d(nn.Block):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.mp1 = nn.MaxPool1D(num_points)
        self.fc1 = nn.Dense(512)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(9)
        self.relu = nn.Activation('relu')

        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.bn4 = nn.BatchNorm(in_channels=512)
        self.bn5 = nn.BatchNorm(in_channels=256)
        self.iden = self.params.get_constant('iden', value=nd.array([1,0,0,0,1,0,0,0,1],dtype='float32').reshape(1,9))


    def forward(self, x):

        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.flatten()

        x = nd.relu(self.bn4(self.fc1(x)))
        x = nd.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + self.iden.data()
        x = nd.reshape(x,(-1, 3, 3))
        return x


class PointNetfeat(nn.Block):
    def __init__(self, num_points = 2500, global_feat = True, routing=None):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.routing = routing
        self.conv1 = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.mp1 = nn.MaxPool1D(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):

        if self.routing is not None:
            routing_weight = nd.softmax(nd.zeros(shape=(1, 1, self.num_points), ctx=x.context),axis=2)
        trans = self.stn(x)
        x = nd.transpose(x,(0,2,1))
        x = nd.batch_dot(x, trans)
        x = nd.transpose(x,(0,2,1))
        x = nd.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = nd.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.routing is not None:
            s = nd.sum(x * routing_weight, axis=2, keepdims=True)
            # v = Squash(s, axis=1)
            for _ in range(self.routing):
                routing_weight = routing_weight + nd.sum(x * s, axis=1,keepdims=True)
                c = nd.softmax(routing_weight, axis=2)
                s = nd.sum(x * c, axis=2, keepdims=True)
                # v = Squash(s, axis=1)
            x = s
        else:
            x = self.mp1(x)
        if self.global_feat:
            return x, trans
        else:
            x = x.repeat(self.num_points, axis=2)
            return nd.concat(x, pointfeat, dim=1), trans

class PointNetCls(nn.Block):
    def __init__(self, num_points = 2500, k = 2, routing=None):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True, routing=routing)
        self.fc1 = nn.Dense(512)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(k)
        self.bn1 = nn.BatchNorm(in_channels=512)
        self.bn2 = nn.BatchNorm(in_channels=256)
        self.relu = nn.Activation('relu')
    def forward(self, x):
        x, trans = self.feat(x)
        x = nd.relu(self.bn1(self.fc1(x)))
        x = nd.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # return nd.log_softmax(x, axis=-1), trans
        return x, trans


class PointNetDenseCls(nn.Block):
    def __init__(self, num_points = 2500, k = 2, routing=None):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False, routing=routing)
        self.conv1 = nn.Conv1D(512, 1)
        self.conv2 = nn.Conv1D(256, 1)
        self.conv3 = nn.Conv1D(128, 1)
        self.conv4 = nn.Conv1D(self.k, 1)
        self.bn1 = nn.BatchNorm(in_channels=512)
        self.bn2 = nn.BatchNorm(in_channels=256)
        self.bn3 = nn.BatchNorm(in_channels=128)

    def forward(self, x):
        # batchsize = x.shape[0]
        x, trans = self.feat(x)
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose((0,2,1))
        # x = x.log_softmax(axis=-1)
        # x = x.reshape(batchsize, self.num_points, self.k)
        return x, trans


if __name__ == '__main__':
    ctx = mx.gpu(2)
    sim_data = nd.random.uniform(shape=(32,3,2500), ctx=ctx)
    trans = STN3d()
    trans.initialize(ctx=ctx)
    out = trans(sim_data)
    print('stn', out.shape)

    pointfeat = PointNetfeat(global_feat=True)
    pointfeat.initialize(ctx=ctx)
    out, _ = pointfeat(sim_data)
    print('global feat', out.shape)

    pointfeat = PointNetfeat(global_feat=False)
    pointfeat.initialize(ctx=ctx)
    out, _ = pointfeat(sim_data)
    print('point feat', out.shape)

    cls = PointNetCls(k = 5)
    cls.initialize(ctx=ctx)
    out, _ = cls(sim_data)
    print('class', out.shape)

    seg = PointNetDenseCls(k = 3)
    seg.initialize(ctx=ctx)
    out, _ = seg(sim_data)
    print('seg', out.shape)
