import torch
import ot  # 确保安装了POT库


def ot_plan(query, support, logit_scale=1.0):
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
    C = torch.cdist(query, support, p=2)  #是一个10000*80000的结果。即每个无标签数据和8w个有标签数据的距离
    # 正则化项
    reg = 1 / logit_scale  # 通过调节logit_scale来调整温度
    # 计算最优传输计划，使用 Sinkhorn 算法
    dim_p, dim_q = C.shape
    p = torch.ones(dim_p, device=C.device, dtype=torch.double) / dim_p  # query 的均匀分布
    q = torch.ones(dim_q, device=C.device, dtype=torch.double) / dim_q  # support 的均匀分布
    # 使用 Sinkhorn 算法计算最优传输计划
    P = ot.bregman.sinkhorn(p, q, C, reg=reg, numItermax=10)
    # 对传输计划进行标准化
    plan = P / P.sum(dim=1, keepdim=True)  # 每一行归一化
    # 将计算结果转换为支持的类型
    plan = plan.type_as(support)

    return plan
