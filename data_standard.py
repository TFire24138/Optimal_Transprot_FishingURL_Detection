import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""读取url特征，预处理"""
url_csv_path = "URLfeatures.csv"
url_data = pd.read_csv(url_csv_path)
url_data.index = url_data.index + 1
# 确保url_data的索引列是整数类型
url_data.index = url_data.index.astype(int)

# 假设除了'label'和'URL'列以外的列需要标准化
features_to_standardize = url_data.columns.difference(['label', 'URL'])
# 初始化标准化器
scaler = StandardScaler()
# 对指定列进行标准化
url_data[features_to_standardize] = scaler.fit_transform(url_data[features_to_standardize])

url_data.to_csv('URL_standard.csv',index = True)
print("标准化URL数据保存成功")