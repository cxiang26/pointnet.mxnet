
from mxnet.gluon import nn
from models.pointnet_globalfeat import PointNetfeat_vanilla

class PointNetCls_vanilla(nn.HybridBlock):
    def __init__(self, num_points=2500, k=2, routing=None):
        super(PointNetCls_vanilla, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat_vanilla(num_points, global_feat=True, routing=routing)
        self.fc1 = nn.Dense(512)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(k)
        self.dp = nn.Dropout(.7)
        self.bn1 = nn.BatchNorm(in_channels=512)
        self.bn2 = nn.BatchNorm(in_channels=256)

    def hybrid_forward(self, F, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dp(x)
        x = self.fc3(x)
        return x, trans