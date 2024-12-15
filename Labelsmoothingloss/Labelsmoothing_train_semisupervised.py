import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import os
"""定义Focal Loss"""
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    def forward(self, inputs, targets):
        # inputs: 预测值 (batch_size, num_classes)
        # targets: 原始标签 (batch_size)
        true_dist = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), self.confidence)
        true_dist += self.smoothing / self.num_classes
        return torch.mean(torch.sum(-true_dist * F.log_softmax(inputs, dim=1), dim=1))

"""定义卷积神经网络模型"""
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         kernel_size = 9
#         padding = ((kernel_size - 1) // 2)
#         self.conv1 = nn.Conv1d(in_channels=85, out_channels=32, kernel_size=kernel_size, padding=padding)
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.fc1 = nn.Linear(16, 10)
#         self.dropout = nn.Dropout(p=0.1)
#         self.fc2 = nn.Linear(10, 2)
#     def forward(self, x):
#         x = x.permute(1, 0)
#         x = nn.functional.relu(self.conv1(x))
#         x = self.dropout(x)
#         x = x.permute(1, 0)
#         x = self.pool(x)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

"""定义神经网络"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 假设输入的特征数量是85
        self.fc1 = nn.Linear(85, 128)  # 第一个全连接层，输入特征85，输出128个特征
        self.dropout = nn.Dropout(p=0.19)  # Dropout层
        self.fc2 = nn.Linear(128, 64)  # 第二个全连接层，输入特征128，输出64个特征
        self.fc3 = nn.Linear(64, 2)  # 第三个全连接层，输入特征64，输出2个特征（分类数）
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # 第一个全连接层后接ReLU激活函数
        x = self.dropout(x)  # Dropout层
        x = nn.functional.relu(self.fc2(x))  # 第二个全连接层后接ReLU激活函数
        x = self.dropout(x)  # Dropout层
        x = self.fc3(x)  # 第三个全连接层
        return x

"""置信度过滤策略生成伪标签"""
def generate_pseudo_labels(model, data_loader, threshold):
    model.eval()
    pseudo_data = []
    pseudo_labels = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]  # 确保正确解包输入数据
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_label = probs.max(dim=1)
            mask = conf >= threshold
            # 调试信息：检查是否有数据超过阈值
            if torch.sum(mask).item() > 0:
                pseudo_data.append(inputs[mask])
                pseudo_labels.append(pseudo_label[mask])
            # else:
            #     print("No samples in this batch met the confidence threshold.")
    # 在连接之前检查列表是否非空
    if pseudo_data and pseudo_labels:
        total_pseudo_data = torch.cat(pseudo_data, dim=0)
        total_pseudo_labels = torch.cat(pseudo_labels, dim=0)
        print(f"Generated {len(total_pseudo_data)} pseudo-labeled samples.")
        return DataLoader(TensorDataset(total_pseudo_data, total_pseudo_labels), batch_size=100, shuffle=True)
    else:
        # print("No pseudo-labeled samples generated. Consider lowering the threshold.")
        return None



"""加载有标签数据"""
train_file_path = r'E:\大学作业\大创\模型\第二期工作\splited_data\train_data_version_40.csv'
train_data = pd.read_csv(train_file_path)
train_data = train_data.iloc[:, 1:]
labels = train_data.pop("label")
numpy_features = train_data.values
tensor_features = torch.from_numpy(numpy_features).float()
tensor_labels = torch.tensor(labels.values, dtype=torch.int)
train_dataset = TensorDataset(tensor_features, tensor_labels)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
"""加载无标签数据"""
nolabel_file_path = r'E:\大学作业\大创\模型\第二期工作\splited_data\train_data_nolabel.csv'
unlabeled_data = pd.read_csv(nolabel_file_path)
unlabeled_data = unlabeled_data.iloc[:, 2:]#把URL和label列都去掉
numpy_unlabeled_features = unlabeled_data.values
tensor_unlabeled_features = torch.from_numpy(numpy_unlabeled_features).float()
unlabeled_dataset = TensorDataset(tensor_unlabeled_features)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=100, shuffle=False)

# 创建模型实例
model = Net()
num_classes = 2
criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0015)
# 引入余弦退火学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0005)
# 半监督训练
num_epochs = 100
initial_threshold = 0.8
threshold_step = 0.01
for epoch in range(num_epochs):
    model.train()
    # 使用有标签数据训练
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch+1)%2==0:
        print(f"完成第{epoch}个epoch的训练")
    """第一段训练"""
    if (epoch+1)%5==0 and 30<=epoch+1<=60:
        current_threshold = initial_threshold + (epoch // 10) * threshold_step
        # if current_threshold > 0.85:
        #      current_threshold = 0.85
        pseudo_loader = generate_pseudo_labels(model, unlabeled_loader, threshold=current_threshold)
        if pseudo_loader is not None:
            print(f"Epoch [{epoch + 1}/{num_epochs}], 使用伪标签数据进行训练")
            for data, target in pseudo_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()
    """第二段训练"""
    if (epoch+1)>60:
        current_threshold = initial_threshold + (epoch // 10) * threshold_step * 2
        if current_threshold > 0.95:
            current_threshold = 0.95
        pseudo_loader = generate_pseudo_labels(model, unlabeled_loader, threshold=current_threshold)
        if pseudo_loader is not None:
            print(f"Epoch [{epoch + 1}/{num_epochs}], 使用伪标签数据进行训练")
            for data, target in pseudo_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()
    # 每10个epoch更新一次无标签数据集的伪标签
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), f'./40_lnn_semi_epoch_{epoch}.pth')



