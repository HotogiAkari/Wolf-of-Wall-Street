# 文件路径: data_process/feature_postprocessors.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from abc import ABC, abstractmethod

class FeaturePostprocessor(ABC):
    """所有特征后处理器的抽象基类。"""
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """接收一个包含所有基础特征的 DataFrame，并对其进行处理。"""
        pass

class StationarityTransformer(FeaturePostprocessor):
    """对指定的列进行平稳化处理。"""
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] Running: Stationarity Transformer...")
        
        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        cols_to_log_return = df.columns.intersection(['open', 'high', 'low', 'close', 'cpi', 'm2', 'benchmark_close', 'industry_close'])
        
        for col in cols_to_log_return:
            if col in df.columns:
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        return df

class AlphaLabelCalculator(FeaturePostprocessor):
    """
    通过多因子模型回归，计算 Alpha 作为预测标签。
    这是一个核心的后处理器，因为它依赖于所有基础特征都已计算完毕。
    """
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        :param df: 包含了所有基础特征的 DataFrame。
        :param kwargs: 必须包含 'factors_df'，即因子收益率数据。
        :return: 增加了 Alpha 标签列的 DataFrame。
        """
        factors_df = kwargs.get('factors_df')
        if factors_df is None:
            # 如果没有提供因子数据，则不执行任何操作，由后续的回退逻辑处理
            if self.config.get('global_settings', {}).get('verbose', True):
                print("    - 警告: AlphaLabelCalculator 未收到因子数据，跳过计算。")
            return df
            
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] 正在运行: Alpha 标签计算器 (向量化)...")

        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        horizon = run_config.get("labeling_horizon", 30)
        label_col = run_config.get('label_column', 'label_alpha')
        
        # 1. 准备数据：合并股票数据和因子数据
        df_merged = df.join(factors_df, how='left')
        
        # 计算股票的【日】超额收益率 (用于历史回归)
        df_merged['daily_excess_return'] = df_merged['close'].pct_change() - df_merged['rf']
        
        # 计算【未来 N 日】的超额收益率 (作为我们最终要剥离的目标)
        df_merged['future_excess_return'] = df_merged['close'].pct_change(periods=horizon).shift(-horizon) - df_merged['rf']

        factors = ['mkt_rf', 'smb', 'hml']
        # 填充初始的 NaN，以确保 RollingOLS 的数据窗口是连续的、无缺失的
        df_merged[factors + ['daily_excess_return']] = df_merged[factors + ['daily_excess_return']].fillna(0)

        # 2. 使用 RollingOLS 高效计算滚动的因子暴露 (Betas)
        rolling_window = 252 # 使用过去一年的数据进行滚动回归
        
        # 准备回归的 X (自变量) 和 Y (因变量)
        X_reg = sm.add_constant(df_merged[factors])
        Y_reg = df_merged['daily_excess_return']
        
        # min_nobs 确保只有在窗口内有足够多的有效数据时才开始计算
        rols = RollingOLS(Y_reg, X_reg, window=rolling_window, min_nobs=int(rolling_window * 0.8))
        rres = rols.fit()
        betas_df = rres.params.copy()

        # 3. 计算 Alpha
        # 核心公式: Alpha(t) = FutureExcessReturn(t) - E[FutureFactorReturn(t)]
        # 预期未来因子收益 = Beta(t) * E[FutureFactorReturn(t)]
        # 近似: 使用 Beta(t) * (历史因子平均收益 * horizon) 作为预期的未来因子收益
        # 计算每个时间点的预期超额收益 (由因子解释的部分)
        expected_return_from_factors = (betas_df[factors] * df_merged[factors]).sum(axis=1) * horizon
        
        # Alpha 是真实未来超额收益中，无法被因子模型解释的“残差”部分
        df[label_col] = df_merged['future_excess_return'] - expected_return_from_factors

        # 4. 对计算出的 Alpha 进行去极值处理，以增强标签的稳定性
        # 这可以移除由于市场极端事件或模型不稳定造成的极端 Alpha 值
        lower_bound = df[label_col].quantile(0.01)
        upper_bound = df[label_col].quantile(0.99)
        df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
        
        return df

class RawReturnLabelCalculator(FeaturePostprocessor):
    """(回退) 计算原始的未来收益率作为标签。"""
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] (Fallback) Running: Raw Return Label Calculator...")
        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        horizon = run_config.get("labeling_horizon", 30)
        
        # 直接从 global_settings 读取，不再提供可能导致混淆的默认值
        # 如果 config 中没有 label_column，就让它报错，因为这是一个关键配置
        try:
            label_col = self.config['global_settings']['label_column']
        except KeyError:
            print("错误: 在 global_settings 中未找到关键配置 'label_column'。无法计算标签。")
            return df
        
        df[label_col] = df['close'].pct_change(periods=horizon).shift(-horizon)
        
        # 对标签进行去极值处理
        # 检查列是否存在，防止 pct_change 后全是 NaN
        if label_col in df.columns and not df[label_col].isnull().all():
            lower_bound, upper_bound = df[label_col].quantile(0.01), df[label_col].quantile(0.99)
            df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
            
        return df

class CorrelationSelector(FeaturePostprocessor):
    """根据相关性剔除冗余特征。"""
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] Running: Correlation Selector...")
        
        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        core_features = {'open', 'high', 'low', 'close', 'volume'}
        numeric_df = df.select_dtypes(include=np.number)
        features_to_check = [col for col in numeric_df.columns if col not in core_features and not col.startswith('future_')]
        if not features_to_check: return df
        
        corr_matrix = numeric_df[features_to_check].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        threshold = run_config.get("correlation_threshold", 0.90)
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        if to_drop:
            df.drop(columns=to_drop, inplace=True, errors='ignore')
            if self.config.get('global_settings', {}).get('verbose', True):
                print(f"    - Removed {len(to_drop)} highly correlated features.")
        return df

# --- 注册并运行所有后处理器 ---
ALL_POSTPROCESSORS = [
    StationarityTransformer,
    CorrelationSelector,
]

def run_all_feature_postprocessors(df: pd.DataFrame, config: dict, **kwargs) -> pd.DataFrame:
    """
    运行所有后处理器。
    这现在是所有后处理逻辑（包括标签生成）的唯一入口点。
    """
    df_copy = df.copy()
    
    # --- 2.1 运行通用后处理器 ---
    for processor_class in ALL_POSTPROCESSORS:
        processor = processor_class(config)
        df_copy = processor.process(df_copy, **kwargs)

    # --- 2.2 根据传入的`kwargs`决定运行哪个标签计算器 ---
    factors_df = kwargs.get('factors_df')
    if factors_df is not None:
        if config.get('global_settings', {}).get('verbose', True):
            print("INFO: Factor data provided. Using AlphaLabelCalculator.")
        label_calculator = AlphaLabelCalculator(config)
        df_copy = label_calculator.process(df_copy, factors_df=factors_df)
    else:
        if config.get('global_settings', {}).get('verbose', True):
            print("INFO: Factor data not provided. Falling back to RawReturnLabelCalculator.")
        label_calculator = RawReturnLabelCalculator(config)
        df_copy = label_calculator.process(df_copy)
        
    return df_copy