import pandas as pd

# 假设你的数据存储在DataFrame中，且已经加载到变量df中
df = pd.read_csv("splited_data/train_data_label.csv")
df_true = pd.read_csv("splited_data/train_data_label.csv")
# 检查数据确保'label'列存在
if 'label' in df.columns:
    # 确保数据集中至少有2万条数据
    num_samples_to_flip = 20000
    if len(df) < num_samples_to_flip:
        print("Error: The dataset does not contain enough samples.")
    else:
        # 随机选择2万条数据的索引
        indices_to_flip = df.sample(n=num_samples_to_flip).index
        # 将这些数据的'label'列的值取反
        # 假设标签是二分类的，可以直接取反
        df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
        # 保存修改后的数据
        df.to_csv("splited_data/train_data_version_25%.csv", index=False)
        print("特征取反的数据保存成功！")
else:
    print("Error: 'label' column not found in the data.")









