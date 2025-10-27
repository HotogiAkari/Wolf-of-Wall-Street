# 文件路径: model/builders/base_builder.py

import pandas as pd
from typing import Any, Dict
from abc import ABC, abstractmethod

class BaseBuilder(ABC):
    """
    所有模型构建器的抽象基类。
    它定义了所有 Builder 必须遵守的接口和返回类型约定。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Builder。
        :param config: 一个包含所有相关配置的字典。
        """
        self.config = config

    @abstractmethod
    def train_and_evaluate_fold(self, cached_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        在一个数据折 (fold) 上训练和评估模型。

        :param cached_data: 从 L3 缓存加载的、包含预处理好的 Tensors 或 DataFrames 的字典。
        :param kwargs: 其他可选参数。
        :return: 一个包含标准化 key 的字典，例如:
                 {
                     'artifacts': ...,      # 训练产物，如 model state_dict
                     'ic_series': ...,     # 包含 'rank_ic' 的 pd.DataFrame
                     'oof_preds': ...,     # 包含 'date', 'y_true', 'y_pred' 的 pd.DataFrame
                     'fold_stats': ...     # 包含 'best_loss' 等信息的字典
                 }
        """
        pass

    @abstractmethod
    def train_final_model(self, full_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        在全部数据上训练最终的生产模型。

        :param full_df: 包含了所有特征和标签的全量 DataFrame。
        :param kwargs: 其他可选参数。
        :return: 一个包含标准化 key 的字典，例如:
                 {
                     'model': ...,         # 最终的模型对象或模型字典
                     'scaler': ...,        # 对应的 scaler
                     'metadata': ...,      # 包含 feature_cols, model_structure 等的元数据
                     'encoders': ...      # (如果适用) 类别编码器
                 }
        """
        pass