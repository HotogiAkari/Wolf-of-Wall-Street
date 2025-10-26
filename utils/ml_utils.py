import numpy as np
import pandas as pd
import inspect
from scipy.stats import spearmanr

def walk_forward_split(df: pd.DataFrame, config: dict) -> list:
    """
    实现滚动窗口时间序列交叉验证的切分逻辑。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")
    df = df.sort_index()
    train_window = config.get("train_window", 500)
    val_window = config.get("val_window", 60)
    dates = df.index.unique()
    if len(dates) < train_window + val_window:
        print(f"WARNNING: 数据长度 ({len(dates)}) 对于前向传播来说太短了.")
        return []
    
    folds = []
    start_index = train_window
    while start_index + val_window <= len(dates):
        train_end_date = dates[start_index - 1]
        val_end_date = dates[start_index + val_window - 1]
        train_df = df.loc[:train_end_date]
        val_df = df.loc[train_end_date:val_end_date].iloc[1:]
        if not val_df.empty:
            folds.append((train_df, val_df))
        start_index += val_window
    return folds

def spearman_corr_scorer(y_true, y_pred):
    """
    使用 SciPy 计算 Spearman 相关系数，以忽略 Pandas 索引。
    """
    y_true_vals = np.asarray(y_true)
    y_pred_vals = np.asarray(y_pred)
    if np.var(y_true_vals) < 1e-8 or np.var(y_pred_vals) < 1e-8:
        return 0.0
    try:
        correlation, _ = spearmanr(y_true_vals, y_pred_vals)
        return correlation if np.isfinite(correlation) else 0.0
    except Exception:
        return 0.0