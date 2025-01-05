import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, k=1):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        
        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        # x shape: (batch_size, num_points, 6)

        if x.dim() != 3:
            # 动态获取 batch_size
            batch_size = x.shape[0]

            # 动态获取 num_points
            num_points = x.shape[1] // 6  # 第二个维度的总大小除以每个点的维度

            # 调整 shape
            x = x.reshape(batch_size, num_points, 6)

        x = x.permute(0, 2, 1)  # 转换为 (batch_size, 6, num_points)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]  # 全局最大池化，得到 (batch_size, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)  # 输出形状为 (batch_size, k)
        return x

    def _initialize_weights(self):
        # 使用 Kaiming 正态初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)