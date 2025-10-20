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
    """
    计算原始的未来收益率作为标签。
    """
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        keyword = kwargs.get('keyword', '未知股票')
        ticker = kwargs.get('ticker', '未知代码')
        
        print(f"\n  --- [数据后处理] INFO: 为 {keyword} ({ticker}) 计算原始收益率标签... ---")
        
        try:
            # 1. 检查输入数据
            if df.empty:
                print(f"    - WARNNING: 为 {keyword} ({ticker}) 输入的 DataFrame 为空，无法计算标签。")
                return df
            if 'close' not in df.columns:
                print(f"    - ERROR: 为 {keyword} ({ticker}) 输入的 DataFrame 中缺少 'close' 列。")
                return df

            # 2. 获取配置
            # self.config 已经是合并后的 run_config，我们可以直接从顶层获取
            horizon = self.config.get("labeling_horizon")
            label_col = self.config.get("label_column")

            # 增加对关键配置是否存在的检查
            if not all([horizon, label_col]):
                missing_keys = []
                if not horizon: missing_keys.append("'labeling_horizon'")
                if not label_col: missing_keys.append("'label_column'")
                print(f"    - ERROR: 配置不完整，缺少关键键: {', '.join(missing_keys)}。无法计算标签。")
                return df

            print(f"    - INFO: 将使用标签列名 '{label_col}'，预测未来 {horizon} 天的收益率。")

            # 3. 执行核心计算
            print(f"    - INFO: 正在对 'close' 列执行 pct_change(periods={horizon}).shift(-{horizon})...")
            future_returns = df['close'].pct_change(periods=horizon).shift(-horizon)
            
            # 4. 检查计算结果
            nan_count = future_returns.isnull().sum()
            total_count = len(future_returns)
            print(f"    - INFO: 计算完成。结果包含 {nan_count} 个 NaN 值 (共 {total_count} 行)。")

            if nan_count == total_count:
                print(f"    - WARNNING: 计算出的未来收益率序列全部为 NaN。这可能是因为数据量 ({total_count}) 小于预测周期 ({horizon})。")
            
            # 5. 赋值
            df[label_col] = future_returns
            print(f"    - INFO: 已成功将计算结果添加到 DataFrame 的 '{label_col}' 列。")

            # 6. 去极值处理
            if label_col in df.columns and not df[label_col].isnull().all():
                print("    - INFO: 正在对标签列进行去极值处理 (clip at 1% and 99%)...")
                lower_bound, upper_bound = df[label_col].quantile(0.01), df[label_col].quantile(0.99)
                df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
            
            print(f"  --- 成功为 {keyword} ({ticker}) 生成原始收益率标签 ---")
            return df

        except Exception as e:
            print(f"    - 致命错误: 在 RawReturnLabelCalculator 内部为 {keyword} ({ticker}) 计算时发生意外异常: {e}")
            import traceback
            traceback.print_exc() # 打印完整的错误堆栈
            # 返回原始df，避免破坏流程
            return df

class CorrelationSelector(FeaturePostprocessor):
    """
    (修复后) 根据相关性剔除冗余特征。
    此版本会明确地将标签列从相关性分析中排除。
    """
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print("  - [Post-processing] Running: Correlation Selector...")
        
        # --- (修改开始) ---
        
        # 1. 合并配置，以便轻松访问所有参数
        run_config = {**self.config.get('global_settings', {}), **self.config.get('strategy_config', {})}
        
        # 2. 定义不应参与相关性筛选的核心列和标签列
        # 核心的 OHLCV 数据不应被移除
        features_to_exclude = {'open', 'high', 'low', 'close', 'volume'}
        
        # 从配置中获取标签列的名称，并将其加入排除列表
        label_col = run_config.get('label_column')
        if label_col:
            features_to_exclude.add(label_col)
            
        # 3. 筛选出所有数值类型的特征列
        numeric_df = df.select_dtypes(include=np.number)
        
        # 4. 确定最终要进行相关性检查的特征列表
        # 排除核心列、标签列以及所有代表未来信息的列
        features_to_check = [
            col for col in numeric_df.columns 
            if col not in features_to_exclude and not col.startswith('future_')
        ]
        
        # --- (修改结束) ---

        if not features_to_check:
            if self.config.get('global_settings', {}).get('verbose', True):
                print("    - INFO: No features available for correlation check. Skipping.")
            return df
        
        # --- (后续逻辑完全不变) ---
        
        corr_matrix = numeric_df[features_to_check].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        threshold = run_config.get("correlation_threshold", 0.90)
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            df.drop(columns=to_drop, inplace=True, errors='ignore')
            if self.config.get('global_settings', {}).get('verbose', True):
                print(f"    - Removed {len(to_drop)} highly correlated features: {to_drop}")
        else:
            if self.config.get('global_settings', {}).get('verbose', True):
                print("    - INFO: No highly correlated features found to remove.")
                
        return df

# --- 注册并运行所有后处理器 ---
ALL_POSTPROCESSORS = [
    StationarityTransformer,
    CorrelationSelector,
]

def run_all_feature_postprocessors(df: pd.DataFrame, config: dict, **kwargs) -> pd.DataFrame:
    """
    (已最终修复) 运行所有后处理器。
    确保了包含 ticker 和 keyword 的 kwargs 在所有调用链中被正确传递。
    """
    df_copy = df.copy()
    
    # --- 1. 运行通用后处理器 ---
    # 此循环现在可以正确地将 kwargs 传递给每个 process 方法
    for processor_class in ALL_POSTPROCESSORS:
        processor = processor_class(config)
        df_copy = processor.process(df_copy, **kwargs)

    # --- 2. (已重构) 根据配置明确选择标签计算器 ---
    strategy_cfg = config.get('strategy_config', {})
    labeling_method = strategy_cfg.get('labeling_method', 'raw_return') 
    
    factors_df = kwargs.get('factors_df')
    keyword = kwargs.get('keyword', '未知股票')
    ticker = kwargs.get('ticker', '未知代码')

    # 逻辑分支：只有当用户明确要求使用 'alpha' 且因子数据确实存在时，才使用 AlphaLabelCalculator
    if labeling_method.lower() == 'alpha' and factors_df is not None:
        print(f"  - [数据后处理] INFO: 对于 {keyword} ({ticker})，配置要求使用 'alpha' 标签且因子数据可用。正在运行 Alpha 标签计算器。")
        label_calculator = AlphaLabelCalculator(config)
        
        # (核心修复点) 将 kwargs 传递给 AlphaLabelCalculator 的 process 方法
        df_copy = label_calculator.process(df_copy, factors_df=factors_df, **kwargs)
    
    # 在所有其他情况下 (要求 alpha 但因子数据缺失，或直接要求 raw_return)，都使用 RawReturnLabelCalculator
    else:
        if labeling_method.lower() == 'alpha' and factors_df is None:
            print(f"  - [数据后处理] WARNNING: 对于 {keyword} ({ticker})，配置要求使用 'alpha' 标签，但因子数据不可用。将回退至【原始收益率】标签。")
        
        label_calculator = RawReturnLabelCalculator(config)
        
        # (核心修复点) 将 kwargs 传递给 RawReturnLabelCalculator 的 process 方法
        df_copy = label_calculator.process(df_copy, **kwargs)
        
    return df_copy