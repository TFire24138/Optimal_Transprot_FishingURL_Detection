import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment

# 确定设备，优先使用 GPU，如果 GPU 不可用则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def optimal_transport_pseudo_labels(model, labeled_loader, unlabeled_loader):
    model.eval()
    labeled_features, labeled_labels = [], []
    unlabeled_features = []
    # 获取有标签数据的嵌入
    with torch.no_grad():
        for data, target in labeled_loader:  #target就是label
            data, target = data.to(device), target.to(device)  # 移动到GPU
            features = model.fc1(data)  # 提取嵌入
            labeled_features.append(features) #有标签特征经过fc1层后作为特征嵌入，添加到数组中
            labeled_labels.append(target)
    labeled_features = torch.cat(labeled_features, dim=0) #将多个批次的数据拼接在一块
    labeled_labels = torch.cat(labeled_labels, dim=0)
    # 获取未标注数据的嵌入
    with torch.no_grad():
        for data in unlabeled_loader:
            data = data[0].to(device)  # 移动到GPU
            features = model.fc1(data) #将无标签数据的特征也经过fc1层，生成特征嵌入
            unlabeled_features.append(features)
    unlabeled_features = torch.cat(unlabeled_features, dim=0)
    # 使用余弦相似度作为代价矩阵
    cost_matrix = torch.cdist(unlabeled_features, labeled_features, p=2) #计算两个特征集合之间的距离，这里p=2表示使用的是L2距离（即欧几里得距离）
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy()) #匈牙利算法，目的是找到无标签特征嵌入和有标签特征嵌入之间的最佳匹配组合（适合在cpu上算）
    #row_ind 表示哪个无标签样本匹配到哪个有标签样本，col_ind 给出了有标签样本的索引。
    pseudo_labels = labeled_labels[col_ind]  #获取与 col_ind 对应的有标签数据的标签，作为无标签样本的伪标签。
    return unlabeled_features[row_ind], pseudo_labels

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# 加载有标签数据
train_data = pd.read_csv("../splited_data/train_data_version_40.csv")
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

# 创建模型实例并将模型移到GPU
model = Net().to(device)
criterion = FocalLoss(alpha=1, gamma=2).to(device)  # 将损失函数移到GPU
optimizer = optim.Adam(model.parameters(), lr=0.0015)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0005)
# 半监督训练
num_epochs = 100
initial_threshold = 0.8
threshold_step = 0.01
for epoch in range(num_epochs):
    model.train()
    # 使用有标签数据训练
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 移动到GPU
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
    # 每个epoch调整学习率
    scheduler.step()
    # 使用伪标签数据
    if epoch >= 30:  # 从第10个epoch开始使用伪标签
        """把完整的无标签特征和有标签特征传入"""
        unlabeled_features, pseudo_labels = optimal_transport_pseudo_labels(
            model, train_loader, unlabeled_loader
        )
        pseudo_dataset = TensorDataset(unlabeled_features, pseudo_labels)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=100, shuffle=True)
        for data, target in pseudo_loader:
            data, target = data.to(device), target.to(device)  # 移动到GPU
            optimizer.zero_grad()
            output = model(data,use_fc1 = False)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 2 == 0:
        print(f'Epoch {epoch}')
    # 输出日志并保存模型
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # 确保文件夹存在，如果不存在则创建
        save_dir = './40_lnn_focalloss'
        os.makedirs(save_dir, exist_ok=True)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))
