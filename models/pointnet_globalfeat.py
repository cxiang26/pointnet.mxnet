
from mxnet.gluon import nn
from models.transform_nets import input_transform_net, feature_transform_net

class PointNetfeat_vanilla(nn.HybridBlock):
    def __init__(self, num_points = 2500, global_feat = True, routing=None):
        super(PointNetfeat_vanilla, self).__init__()
        self.stn = input_transform_net(num_points = num_points)
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
    def hybrid_forward(self, F, x):

        if self.routing is not None:
            routing_weight = F.softmax(F.zeros(shape=(1, 1, self.num_points), ctx=x.context),axis=2)
        trans = self.stn(x)
        x = F.transpose(x,(0,2,1))
        x = F.batch_dot(x, trans)
        x = F.transpose(x,(0,2,1))
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.routing is not None:
            s = F.sum(x * routing_weight, axis=2, keepdims=True)
            # v = Squash(s, axis=1)
            for _ in range(self.routing):
                routing_weight = routing_weight + F.sum(x * s, axis=1,keepdims=True)
                c = F.softmax(routing_weight, axis=2)
                s = F.sum(x * c, axis=2, keepdims=True)
                # v = Squash(s, axis=1)
            x = s
        else:
            x = self.mp1(x)
        if self.global_feat:
            return x, trans
        else:
            x = x.repeat(self.num_points, axis=2)
            return F.concat(x, pointfeat, dim=1), trans

class PointNetfeat(nn.HybridBlock):
    def __init__(self, num_points = 2500, global_feat = True, routing=None):
        super(PointNetfeat, self).__init__()
        self.stn1 = input_transform_net(num_points = num_points)
        self.stn2 = feature_transform_net(num_points = num_points, K=64)
        self.routing = routing
        self.conv1 = nn.Conv1D(64, 1)
        self.conv1_feat_trans = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn1_feat_trans = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.mp1 = nn.MaxPool1D(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def hybrid_forward(self, F, x):

        if self.routing is not None:
            routing_weight = F.softmax(F.zeros(shape=(1, 1, self.num_points), ctx=x.context),axis=2)
        input_trans = self.stn1(x)
        x = F.transpose(x,(0,2,1))
        x = F.batch_dot(x, input_trans)
        x = F.transpose(x,(0,2,1))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1_feat_trans(self.conv1_feat_trans(x)))
        feat_trans = self.stn2(x)
        x = F.transpose(x, (0, 2, 1))
        x = F.batch_dot(x, feat_trans)
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.routing is not None:
            s = F.sum(x * routing_weight, axis=2, keepdims=True)
            # v = Squash(s, axis=1)
            for _ in range(self.routing):
                routing_weight = routing_weight + F.sum(x * s, axis=1,keepdims=True)
                c = F.softmax(routing_weight, axis=2)
                s = F.sum(x * c, axis=2, keepdims=True)
                # v = Squash(s, axis=1)
            x = s
        else:
            x = self.mp1(x)
        if self.global_feat:
            return x, feat_trans
        else:
            x = x.repeat(self.num_points, axis=2)
            return F.concat(x, pointfeat, dim=1), feat_trans