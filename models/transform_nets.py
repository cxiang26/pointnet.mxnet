
from mxnet.gluon import nn
from mxnet import nd

class input_transform_net(nn.Block):
    def __init__(self, num_points=2500):
        super(input_transform_net,self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.mp1 = nn.MaxPool1D(num_points)
        self.fc1 = nn.Dense(512)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(9)
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.bn4 = nn.BatchNorm(in_channels=512)
        self.bn5 = nn.BatchNorm(in_channels=256)
        self.iden = self.params.get_constant('iden', value=nd.eye(3,3,0,dtype='float32'))

    def forward(self, x):
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = nd.flatten(x)
        x = nd.relu(self.bn4(self.fc1(x)))
        x = nd.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = nd.reshape(x, (-1, 3, 3))
        x = x + self.iden.data()
        return x

class feature_transform_net(nn.Block):
    def __init__(self, num_points=2500, K=64):
        super(feature_transform_net, self).__init__()
        self.num_points = num_points
        self.K = K
        self.conv1 = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.mp1 = nn.MaxPool1D(num_points)
        self.fc1 = nn.Dense(512)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(self.K ** 2)
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.bn4 = nn.BatchNorm(in_channels=512)
        self.bn5 = nn.BatchNorm(in_channels=256)
        self.biases = self.params.get_constant('biases', value=nd.eye(self.K, self.K, 0, dtype='float32'))

    def forward(self, x):
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = nd.flatten(x)
        x = nd.relu(self.bn4(self.fc1(x)))
        x = nd.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = nd.reshape(x, (-1, self.K, self.K))
        x = x + self.biases.data()
        return x