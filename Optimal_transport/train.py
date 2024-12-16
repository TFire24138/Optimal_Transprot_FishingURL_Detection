from utils import train
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(85, 128)  # 嵌入层
        self.dropout = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    def forward(self, x,use_fc1=True):
        if use_fc1:  # 如果标志变量为 True，使用 self.fc1 层
            x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在 fc2 后使用 dropout
        x = F.relu(self.fc2(x))  # 再次通过 fc2
        x = self.dropout(x)  # 再次使用 dropout
        x = self.fc3(x)  # 最后的全连接层
        return x

# 加载有标签数据
train_data = pd.read_csv("../splited_data/train_data_version_25.csv")
train_data = train_data.iloc[:, 1:]  # 去掉第一列索引
labels = train_data.pop("label")
numpy_features = train_data.values
tensor_features = torch.from_numpy(numpy_features).float()
tensor_labels = torch.tensor(labels.values, dtype=torch.int)
train_dataset = TensorDataset(tensor_features, tensor_labels)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 加载无标签数据
unlabeled_data = pd.read_csv("../splited_data/train_data_nolabel.csv")
unlabeled_data = unlabeled_data.iloc[:, 2:]  # 去掉 URL 和标签列
unlabeled_data = unlabeled_data.head(10000)
numpy_unlabeled_features = unlabeled_data.values
tensor_unlabeled_features = torch.from_numpy(numpy_unlabeled_features).float()
unlabeled_dataset = TensorDataset(tensor_unlabeled_features)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=100, shuffle=False)

# 创建模型、优化器和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0005)

num_epochs = 100
initial_threshold = 0.8
threshold_step = 0.01
# 训练模型
for epoch in range(num_epochs):
    epoch_loss = train(model, train_loader, unlabeled_loader, optimizer, device, epoch, logit_scale=1.0)
    epoch_loss_value = epoch_loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss_value:.4f}")
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        # 确保文件夹存在，如果不存在则创建
        save_dir = './25_ot_semi'
        os.makedirs(save_dir, exist_ok=True)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))
        print("保存模型成功")
