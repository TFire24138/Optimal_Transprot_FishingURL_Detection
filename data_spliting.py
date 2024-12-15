import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


if __name__ == '__main__':
    origin_data_filename = "URL_standard.csv"
    data = pd.read_csv(origin_data_filename)
    data = data.set_index('Unnamed: 0')
    labels = data['label']
    #缺失值填0
    data = data.fillna(0)
    #划分训练集和验证+测试集
    data_next, data_train_label, label_next, label_train = train_test_split(data, labels, test_size=80000, stratify=labels)
    #划分验证集和测试集
    labels = data_next['label']
    data_test, data_train_nolabel, label_test, label_val = train_test_split(data_next, labels, test_size=40000, stratify=labels)
    # 将特征和标签合并
    data_train_label['label'] = label_train
    data_train_nolabel['label'] = label_val
    data_test['label'] = label_test
    print('划分数据完成')
    # 保存数据集
    data_train_label.to_csv('splited_data/train_data_label.csv', index=False)
    data_train_nolabel.to_csv('splited_data/train_data_nolabel.csv', index=False)
    data_test.to_csv('splited_data/test_data.csv', index=False)
    print('划分后数据保存成功')