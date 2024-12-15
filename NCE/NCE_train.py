import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

def noisy_ce_loss(pred, target, noise_rate=0.4):
    """
    Noisy Cross-Entropy Loss with noise rate regularization.
    Args:
    - pred (Tensor): model predictions (logits)
    - target (Tensor): true labels
    - noise_rate (float): rate of noisy labels (default 0.1)
    Returns:
    - loss (Tensor): computed loss value
    """
    # Convert target to one-hot encoding
    target_one_hot = torch.eye(pred.size(1))[target].to(pred.device)

    # Adjust loss by considering noisy label
    loss = F.binary_cross_entropy_with_logits(pred, target_one_hot.float())
    return loss * (1 - noise_rate)  # Weight the loss by noise rate


"""定义一维卷积神经网络模型"""
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 一维卷积层，它有853个输入通道、32个输出通道，卷积核大小为7，padding大小为4
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

"""定义线性神经网络"""
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

# 创建模型实例
model = Net()
print(model)
# 使用标签平滑和焦点损失
num_classes = 2

optimizer = optim.Adam(model.parameters(), lr=0.0025)
#读取数据
train_data = pd.read_csv("../splited_data/train_data_version_40.csv")
train_data = train_data.iloc[:, 1:]
labels = train_data.pop("label")
numpy_features = train_data.values
tensor_features = torch.from_numpy(numpy_features).float()
tensor_labels = torch.tensor(labels.values, dtype=torch.int)
train_dataset = TensorDataset(tensor_features, tensor_labels)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 模型训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = noisy_ce_loss(output, target.long())
        loss.backward()
        optimizer.step()
    # 每10个epoch进行一次数据过滤
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), './40_lnn_epoch_{}.pth'.format(epoch))

print("over!")