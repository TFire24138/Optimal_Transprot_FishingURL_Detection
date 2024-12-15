import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import os
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



# 创建模型实例
model = Net()
num_classes = 2
criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.11)
optimizer = optim.Adam(model.parameters(), lr=0.0015)
# 引入余弦退火学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0005)
# 半监督训练
num_epochs = 100
initial_threshold = 0.85
threshold_step = 0.01
model=Net()

#加载测试数据
test_file_path = r'E:\大学作业\大创\模型\第二期工作\splited_data\test_data.csv'
test_data = pd.read_csv(test_file_path)
test_data = test_data.iloc[:, 1:]
test_labels = test_data.pop("label")
test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.int)
test_input = test_data.values
test_input = torch.from_numpy(test_input).float()
test_labels_np = test_labels_tensor.cpu().numpy()

# 加载并验证模型
model_epochs = range(9, 100, 10)
best_models = {}  # 存储最好的三个模型的评价指标
best_epochs = []  # 存储最好的三个模型的epoch值
for epoch in model_epochs:
    # 加载模型
    model.load_state_dict(torch.load(f"40_lnn_semi_epoch_{epoch}.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    with torch.no_grad():
        # 进行预测
        output = model(test_input)
        pred_labels = torch.argmax(output, dim=1)
        pred_labels_np = pred_labels.cpu().numpy()
        # 计算评价指标
        precision = precision_score(test_labels_np, pred_labels_np, zero_division=1, average='macro')
        recall = recall_score(test_labels_np, pred_labels_np, average='macro')
        f1 = f1_score(test_labels_np, pred_labels_np, average='macro')
        # 存储评价指标
        best_models[epoch] = (precision, recall, f1)
        # 打印评价指标
        print(f'Model Epoch {epoch}:')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print('---')

# 找到最好的三个模型
for epoch, scores in sorted(best_models.items(), key=lambda item: item[1][2], reverse=True)[:3]:
    best_epochs.append(epoch)


# 删除不是最好的三个模型的其他模型文件
import os
for epoch in model_epochs:
    if epoch not in best_epochs:
        os.remove(f"40_lnn_semi_epoch_{epoch}.pth")
        print(f"已经删除{epoch}的模型")