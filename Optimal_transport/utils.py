import torch
import torch.nn.functional as F
import ot  # POT库，用于最优传输

def ot_plan(query, support, logit_scale):
    """"""
    """
    计算无标签样本和有标签样本之间的最优传输计划。
    Args:
        query (torch.Tensor): 无标签数据特征，形状为 (num_unlabeled, feature_dim)
        support (torch.Tensor): 有标签数据特征，形状为 (num_labeled, feature_dim)
        logit_scale (float): 温度超参数，调节匹配的“软”程度，默认为 1.0
    Returns:
        torch.Tensor: 计算得到的最优传输计划，形状为 (num_unlabeled, num_labeled)
    """
    # 计算代价矩阵，使用欧几里得距离（L2 距离）
    C = torch.cdist(query, support, p=2)  #是一个40000*80000的结果。即每个无标签数据和8w个有标签数据的距离
    # 正则化项
    reg = 1 / logit_scale  # 通过调节logit_scale来调整温度
    # 计算最优传输计划，使用 Sinkhorn 算法
    dim_p, dim_q = C.shape
    p = torch.ones(dim_p, device=C.device, dtype=torch.double) / dim_p  # query 的均匀分布
    q = torch.ones(dim_q, device=C.device, dtype=torch.double) / dim_q  # support 的均匀分布
    C = torch.clamp(C, min=0, max=60)  # 设置一个合理的阈值，例如 threshold = 10.0
    C = C.to(torch.double)
    # 使用 Sinkhorn 算法计算最优传输计划
    P = ot.bregman.sinkhorn(p, q, C, reg=reg, numItermax=20)
    # 对传输计划进行标准化
    plan = P / P.sum(dim=1, keepdim=True)  # 每一行归一化
    # 将计算结果转换为支持的类型
    plan_1 = plan.type_as(support)
    return plan_1

def normalize_and_log_transform(logits_per_query):
    # Step 1: 计算每一行的最小值和最大值
    min_row = logits_per_query.min(dim=1, keepdim=True)[0]  # (100, 1)
    max_row = logits_per_query.max(dim=1, keepdim=True)[0]  # (100, 1)
    # Step 2: 归一化到 [0, 1] 范围
    logits_per_query_norm = (logits_per_query - min_row) / (max_row - min_row + 1e-6)  # 防止除以零
    # Step 3: 对数变换 (log(1 + x))，进一步压缩数值范围
    logits_per_query_log = torch.log1p(logits_per_query_norm)  # log1p(x) = log(1 + x)
    return logits_per_query_log

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
    #target_one_hot = torch.eye(pred.size(1))[target].to(pred.device)
    target_one_hot = torch.eye(pred.size(1), device=pred.device)[target]
    # Adjust loss by considering noisy label
    loss = F.binary_cross_entropy_with_logits(pred, target_one_hot.float())
    return loss * (1 - noise_rate)  # Weight the loss by noise rate


def train_model(model, train_loader, unlabeled_loader, optimizer, device,epoch, logit_scale):
    model.train()  # 设置模型为训练模式
    total_samples = 0
    """迭代全部有标签数据，计算总的损失"""
    # 初始化存储所有特征的列表
    labeled_features = []
    # 初始化存储所有标签的列表
    all_labeled_labels = []
    labeled_loss = 0
    for labeled_data, labeled_labels in train_loader:
        labeled_data, labeled_labels = labeled_data.to(device), labeled_labels.to(device)
        # 提取有标签数据的特征
        features = model.fc1(labeled_data)
        output = model(labeled_data).to(device)
        # 计算有标签样本的损失
        #loss =  noisy_ce_loss(output, labeled_labels.long()).to(device)
        loss = F.cross_entropy(output, labeled_labels.long())
        """反向传播更新"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """累加全部的loss，拼接feature"""
        labeled_loss += loss
        labeled_features.append(features)
        all_labeled_labels.append(labeled_labels)
        total_samples += len(labeled_data)
    # 将所有特征和标签拼接成一个大的张量
    all_features = torch.cat(labeled_features, dim=0)

    unlabeled_loss = 0
    if epoch>30:
        """迭代无标签数据"""
        for unlabeled_data in unlabeled_loader:
            unlabeled_data = unlabeled_data[0].to(device)
            # 提取无标签数据的特征
            query_features = model.fc1(unlabeled_data)
            # 计算伪标签（注意：这里需要确保 labeled_features 不带梯度）
            with torch.no_grad():
                labeled_features_detached = torch.cat(labeled_features, dim=0).detach()
                """detach可以确保其不进行梯度计算"""
                labeled_labels_detached = torch.cat(all_labeled_labels, dim=0).detach().long()  # Detach labeled_labels
                plan = ot_plan(query_features, labeled_features_detached, logit_scale)
                x = F.one_hot(labeled_labels_detached, num_classes=2).float()
                pseudo_labels = plan @ x
            # if torch.isnan(pseudo_labels).any():
            #     print("Found NaN values in pseudo_labels, removing them.")
            #     # 查找 NaN 的索引
            #     nan_indices = torch.isnan(pseudo_labels).any(dim=1)
            #     # 删除包含 NaN 的数据
            #     pseudo_labels = pseudo_labels[~nan_indices]
            #     query_features = query_features[~nan_indices]
            # if pseudo_labels.size(0) == 0:
            #     print("No valid pseudo labels after removing NaN values, skipping this iteration.")
            #     continue  # 跳过本次训练迭代
            # 计算无标签数据的损失
            logits_per_query = logit_scale * query_features @ labeled_features_detached.T
            logits_per_query_normalized = normalize_and_log_transform(logits_per_query)
            one_hot_labels = F.one_hot(labeled_labels_detached, num_classes=2).float()
            logits_per_class = logits_per_query_normalized @ one_hot_labels
            logits_per_class_log = torch.log1p(logits_per_class)
            logits_per_class_log = torch.nn.functional.softmax(logits_per_class_log, dim=1)
            pseudo_loss = soft_cross_entropy(pseudo_labels,logits_per_class_log) / 2
            """反向传播并更新参数"""
            optimizer.zero_grad()
            pseudo_loss.backward()
            optimizer.step()
            # 计算总损失
            unlabeled_loss = unlabeled_loss+pseudo_loss
            total_samples = total_samples+len(unlabeled_data)
    # 返回平均损失
    return (labeled_loss+unlabeled_loss) *100 / total_samples


def soft_cross_entropy(outputs, targets, weight=1.):
    """计算软交叉熵损失"""
    a= F.log_softmax(outputs, dim=1)
    loss = -targets * a  # 计算交叉熵损失
    return (loss * weight).sum(dim=1).mean()  # 对所有样本的损失取均值