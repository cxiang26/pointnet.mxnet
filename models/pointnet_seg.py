
from mxnet.gluon import nn
from models.pointnet_globalfeat import PointNetfeat_vanilla

class PointNetDenseCls(nn.HybridBlock):
    def __init__(self, num_points=2500, k=2, routing=None):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat_vanilla(num_points, global_feat=False, routing=routing)
        self.conv1 = nn.Conv1D(512, 1)
        self.conv2 = nn.Conv1D(256, 1)
        self.conv3 = nn.Conv1D(128, 1)
        self.conv4 = nn.Conv1D(self.k, 1)
        self.bn1 = nn.BatchNorm(in_channels=512)
        self.bn2 = nn.BatchNorm(in_channels=256)
        self.bn3 = nn.BatchNorm(in_channels=128)

    def hybrid_forward(self, F, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = F.transpose(x, (0,2,1))
        return x, trans