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
    """通过多因子模型回归，计算 Alpha 作为预测标签。"""
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        factors_df = kwargs.get('factors_df')
        if factors_df is None:
            if self.config.get('global_settings', {}).get('verbose', True):
                print("    - WARNNING: Factor data not provided. Skipping Alpha calculation.")
            return df
            
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] Running: Alpha Label Calculator (Vectorized)...")

        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        horizon = run_config.get("labeling_horizon", 30)
        label_col = run_config.get('label_column', 'label_alpha')
        
        df_merged = df.join(factors_df, how='left')
        df_merged['daily_excess_return'] = df_merged['close'].pct_change() - df_merged['rf']
        df_merged['future_excess_return'] = df_merged['close'].pct_change(periods=horizon).shift(-horizon) - df_merged['rf']

        factors = ['mkt_rf', 'smb', 'hml']
        df_merged[factors + ['daily_excess_return']] = df_merged[factors + ['daily_excess_return']].fillna(0)

        rolling_window = 252
        X_reg = sm.add_constant(df_merged[factors])
        Y_reg = df_merged['daily_excess_return']
        
        rols = RollingOLS(Y_reg, X_reg, window=rolling_window, min_nobs=int(rolling_window * 0.8))
        rres = rols.fit()
        betas_df = rres.params.copy()

        expected_return = (betas_df[factors] * df_merged[factors]).sum(axis=1) * horizon
        df[label_col] = df_merged['future_excess_return'] - expected_return

        lower_bound, upper_bound = df[label_col].quantile(0.01), df[label_col].quantile(0.99)
        df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
        return df

class RawReturnLabelCalculator(FeaturePostprocessor):
    """(回退) 计算原始的未来收益率作为标签。"""
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] (Fallback) Running: Raw Return Label Calculator...")
        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        horizon = run_config.get("labeling_horizon", 30)
        label_col = run_config.get('label_column', 'label_alpha') 
        df[label_col] = df['close'].pct_change(periods=horizon).shift(-horizon)
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
    # AlphaLabelCalculator (这个逻辑特殊，在 get_data 中单独处理)
    CorrelationSelector,
]

def run_all_feature_postprocessors(df: pd.DataFrame, config: dict, **kwargs) -> pd.DataFrame:
    df_copy = df.copy()
    for processor_class in ALL_POSTPROCESSORS:
        processor = processor_class(config)
        df_copy = processor.process(df_copy, **kwargs)
    return df_copy