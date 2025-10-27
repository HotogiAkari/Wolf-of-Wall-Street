# 文件路径: utils/encoding_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df_train: pd.DataFrame, df_val: pd.DataFrame, cat_features: list) -> tuple:
    """
    (公共工具函数) 对类别特征进行安全的标签编码，确保编码从 0 开始。
    
    :param df_train: 训练集 DataFrame。
    :param df_val: 验证集 DataFrame。
    :param cat_features: 需要编码的类别特征列名列表。
    :return: (编码后的训练集, 编码后的验证集, 编码器字典)
    """
    encoders = {}
    
    df_train_encoded = df_train.copy()
    df_val_encoded = df_val.copy()

    for col in cat_features:
        if col not in df_train_encoded.columns:
            continue # 如果列不存在则跳过

        le = LabelEncoder()
        
        # 在训练集上 fit 和 transform
        df_train_encoded[col] = le.fit_transform(df_train_encoded[col].astype(str))
        
        # 为验证集中的新类别（在训练集中未出现）创建一个 'unknown' 标记
        # 获取所有已知类别
        known_classes = set(le.classes_)
        
        # 将验证集中不在已知类别中的值替换为 '<unknown>' 字符串
        # 首先确保 '<unknown>' 是一个已知类别
        if '<unknown>' not in known_classes:
            le.classes_ = np.append(le.classes_, '<unknown>')
            
        df_val_encoded[col] = df_val_encoded[col].astype(str).apply(lambda x: x if x in known_classes else '<unknown>')
        
        # 现在可以安全地 transform 验证集
        df_val_encoded[col] = le.transform(df_val_encoded[col])
        
        encoders[col] = le
        
    return df_train_encoded, df_val_encoded, encoders