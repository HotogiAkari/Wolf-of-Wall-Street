# 文件路径: data_process/data_contracts.py
'''
校验数据
'''

import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import Series
from scipy.stats import ks_2samp
from typing import List, Dict, Any

# ===================================================================
# 1. 自定义校验函数 (Custom Checks)
# ===================================================================

def check_no_large_gaps(series: Series[pd.Timestamp]) -> bool:
    """校验时间序列索引中是否存在超过常规阈值的大间隙。"""
    if series.empty:
        return True
    time_diffs = series.sort_values().diff().dropna()
    if time_diffs.empty:
        return True
    # 阈值定义为正常时间间隔中位数的10倍
    threshold = time_diffs.median() * 10
    large_gaps = time_diffs[time_diffs > threshold]
    if not large_gaps.empty:
        print(f"WARNNING: Found {len(large_gaps)} large gaps in time series index. Max gap: {large_gaps.max()}")
    return True # 返回True以作记录，而不是硬性失败

def check_outlier_percentage(series: Series[float], threshold: float = 0.05) -> bool:
    """校验系列中极端异常值的比例是否低于阈值。"""
    if series.empty:
        return True
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr # 使用更严格的3倍IQR
    upper_bound = q3 + 3 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_ratio = len(outliers) / len(series) if not series.empty else 0
    return outlier_ratio < threshold

# ===================================================================
# 2. 数据漂移检测器 (Drift Detector)
# ===================================================================

class DriftDetector:
    """
    一个封装了多种数据漂移检测算法的类。
    """
    def __init__(self, ref_df: pd.DataFrame):
        if ref_df.empty:
            raise ValueError("Reference DataFrame for DriftDetector cannot be empty.")
        self.ref_df = ref_df

    def _calculate_psi(self, ref_series: pd.Series, new_series: pd.Series, bins: int = 10) -> float:
        """计算单个特征的 Population Stability Index (PSI)。"""
        ref_series = ref_series.replace([np.inf, -np.inf], np.nan).dropna()
        new_series = new_series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if ref_series.empty or new_series.empty:
            return 0.0

        # --- 核心改进：使用 np.unique 处理重复的箱体边界 ---
        breakpoints = np.quantile(ref_series, np.linspace(0, 1, bins + 1))
        breakpoints = np.unique(breakpoints) 

        if len(breakpoints) < 2:
            print(f"  - WARNNING (PSI): Not enough unique breakpoints for '{ref_series.name}'. Skipping PSI calculation.")
            return 0.0
            
        ref_counts = pd.cut(ref_series, bins=breakpoints, right=True, include_lowest=True).value_counts(normalize=True)
        new_counts = pd.cut(new_series, bins=breakpoints, right=True, include_lowest=True).value_counts(normalize=True)
        
        psi_df = pd.DataFrame({'ref': ref_counts, 'new': new_counts}).fillna(0)
        psi_df.replace(0, 0.0001, inplace=True)

        psi_df['psi'] = (psi_df['new'] - psi_df['ref']) * np.log(psi_df['new'] / psi_df['ref'])
        return psi_df['psi'].sum()

    def _calculate_ks(self, ref_series: pd.Series, new_series: pd.Series) -> tuple[float, float]:
        """执行 Kolmogorov-Smirnov (KS) 检验。"""
        ref_series = ref_series.replace([np.inf, -np.inf], np.nan).dropna()
        new_series = new_series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if ref_series.empty or new_series.empty:
            return 0.0, 1.0

        statistic, p_value = ks_2samp(ref_series, new_series)
        return statistic, p_value

    def check(self, new_df: pd.DataFrame, feature_list: List[str], method: str = 'psi', threshold_map: Dict[str, float] = None) -> List[str]:
        """检查指定特征列表的分布漂移。"""
        drifted_features = []
        default_threshold = 0.2 if method == 'psi' else 0.05
        
        for feature in feature_list:
            if feature not in self.ref_df.columns or feature not in new_df.columns:
                continue

            threshold = (threshold_map or {}).get(feature, default_threshold)
            
            if method == 'psi':
                score = self._calculate_psi(self.ref_df[feature], new_df[feature])
                if score > threshold:
                    print(f"  - WARNNING (PSI): Feature '{feature}' has drifted. PSI = {score:.4f} > {threshold}")
                    drifted_features.append(feature)
            elif method == 'ks':
                _, p_value = self._calculate_ks(self.ref_df[feature], new_df[feature])
                if p_value < threshold:
                    print(f"  - WARNNING (KS): Feature '{feature}' has drifted. p-value = {p_value:.4f} < {threshold}")
                    drifted_features.append(feature)
        
        return drifted_features

# ===================================================================
# 3. 统一的验证器接口 (Data Validator)
# ===================================================================

class DataValidator:
    """
    一个统一的接口，用于执行所有数据质量检查。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schema = pa.DataFrameSchema(
            columns={
                "open": pa.Column(float, required=True, checks=pa.Check(check_outlier_percentage)),
                "high": pa.Column(float, required=True, checks=pa.Check(check_outlier_percentage)),
                "low": pa.Column(float, required=True, checks=pa.Check(check_outlier_percentage)),
                "close": pa.Column(float, required=True, checks=pa.Check(check_outlier_percentage)),
                "volume": pa.Column(float, checks=[pa.Check.ge(0), pa.Check(check_outlier_percentage)], required=True),
                "label_return": pa.Column(float, nullable=True, required=False),
                "date": pa.Column(pd.Timestamp, required=True) # 'date' is a column after reset_index
            },
            index=pa.Index(int), # After reset_index, the index is just a range of integers
            strict=False,
            coerce=True,
        )
        self.time_index_schema = pa.SeriesSchema(
            pd.Timestamp, 
            unique=True,
            checks=[
                pa.Check(lambda s: s.is_monotonic_increasing, error="Time index is not monotonic increasing"),
                pa.Check(check_no_large_gaps, error="Large gaps found in time series index.")
            ],
            name="date"
        )

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """对DataFrame执行严格的Schema校验。"""
        print("INFO: Running schema validation...")
        if not isinstance(df.index, pd.DatetimeIndex):
            print("ERROR: DataFrame index must be a DatetimeIndex for validation.")
            return False
        
        try:
            # 1. 校验时间索引
            self.time_index_schema.validate(df.index.to_series())
            # 2. 校验列数据
            self.schema.validate(df.reset_index())
            print("SUCCESS: Data schema and time index validation passed.")
            return True
        except pa.errors.SchemaError as e:
            print(f"ERROR: Data schema validation failed!")
            print("Validation Failure Details:")
            print(e.failure_cases)
            return False

    def check_drift(self, ref_df: pd.DataFrame, new_df: pd.DataFrame) -> List[str]:
        """对新旧两个DataFrame执行漂移检测。"""
        print("INFO: Running feature drift check...")
        core_features = self.config.get('drift_check_features', [])
        drift_method = self.config.get('drift_check_method', 'psi')
        drift_thresholds = self.config.get('drift_check_thresholds', {})
        
        if not core_features:
            print("INFO: No features configured for drift check. Skipping.")
            return []
            
        detector = DriftDetector(ref_df)
        drifted = detector.check(new_df, core_features, method=drift_method, threshold_map=drift_thresholds)
        
        if not drifted:
            print("SUCCESS: No significant feature drift detected.")
        else:
            print(f"WARNNING: Found {len(drifted)} drifted features.")
            
        return drifted