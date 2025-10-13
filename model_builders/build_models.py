# 文件路径: model/build_models.py
'''
模型整合与训练
'''
import gc
import sys
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from typing import Any, Dict, List, Optional

# --- 健壮的导入逻辑 ---
try:
    from data_process import get_data
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder
except ImportError:
    print("WARNNING: 标准导入失败. 正在尝试将项目根目录添加到 sys.path.")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
    from data_process import get_data
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder

def _walk_forward_split(df: pd.DataFrame, config: dict) -> list:
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Input DataFrame must have a DatetimeIndex.")
    df = df.sort_index()
    train_window, val_window = config.get("train_window", 500), config.get("val_window", 60)
    dates = df.index.unique()
    if len(dates) < train_window + val_window:
        print(f"WARNNING: 数据长度 ({len(dates)}) 对于前向传播来说太短了."); return []
    
    folds = []
    start_index = train_window
    while start_index + val_window <= len(dates):
        train_end_date = dates[start_index - 1]
        val_end_date = dates[start_index + val_window - 1]
        train_df, val_df = df.loc[:train_end_date], df.loc[train_end_date:val_end_date].iloc[1:]
        if not val_df.empty:
            folds.append((train_df, val_df))
        start_index += val_window
    return folds

def run_training_for_ticker(
    preprocessed_folds: List[Dict[str, Any]],
    ticker: str, 
    model_type: str, 
    config: dict, 
    force_retrain: bool = False, 
    keyword: str = None
) -> Optional[pd.DataFrame]:
    """
    (已重构) 接收预处理好的 folds 列表，执行训练，并生成 OOF 预测文件。
    """
    display_name = keyword if keyword else ticker
    print(f"\n" + "="*80); print(f"--- Starting {model_type.upper()} training for {display_name} ({ticker}) ---")

    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / ticker
    model_dir.mkdir(parents=True, exist_ok=True)
    
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt'}; model_suffix = file_suffixes[model_type]
    
    existing_models = sorted(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
    if existing_models and not force_retrain:
        print(f"INFO: Latest model for {display_name} ({ticker}) already exists. Skipping.")
        ic_history_path = model_dir / f"{model_type}_ic_history.csv"
        if ic_history_path.exists(): return pd.read_csv(ic_history_path, index_col='date', parse_dates=True)
        return None

    builder_map = {'lgbm': LGBMBuilder, 'lstm': LSTMBuilder}
    builder = builder_map[model_type](config)
    
    all_fold_ics = []
    # --- 新增：初始化一个列表来收集 OOF 预测 ---
    oof_predictions = []

    if not preprocessed_folds:
        print(f"WARNNING: No pre-processed folds provided for {display_name}. Skipping validation.")
    else:
        print(f"INFO: Starting Walk-Forward validation for {display_name} across {len(preprocessed_folds)} folds...")
        fold_iterator = tqdm(preprocessed_folds, desc=f"Training {model_type.upper()} on {display_name}", leave=True)
        
        for fold_data in fold_iterator:
            # --- 核心修正：接收所有三个返回值 ---
            artifacts, ic_series_fold, oof_fold_df = builder.train_and_evaluate_fold(
                train_df=None, val_df=None, cached_data=fold_data
            )
            
            if ic_series_fold is not None and not ic_series_fold.empty:
                ic_series_fold['ticker'] = ticker
                all_fold_ics.append(ic_series_fold)
            
            # 将每个 fold 的 OOF 预测添加到列表中
            if oof_fold_df is not None and not oof_fold_df.empty:
                oof_predictions.append(oof_fold_df)
            
            gc.collect()

    # --- 新增：在所有 folds 结束后，保存 OOF 文件 ---
    if oof_predictions:
        full_oof_df = pd.concat(oof_predictions).sort_values('date').drop_duplicates(subset=['date'])
        oof_path = model_dir / f"{model_type}_oof_preds.csv"
        try:
            full_oof_df.to_csv(oof_path, index=False)
            print(f"SUCCESS: Out-of-Fold predictions saved to {oof_path}")
        except Exception as e:
            print(f"ERROR: Failed to save Out-of-Fold predictions: {e}")
    
    full_df = config.get('full_df_for_final_model')
    if full_df is None:
        print("ERROR: full_df_for_final_model not found in config for final training. Skipping.")
    else:
        print(f"INFO: Training final model for {display_name} ({ticker}) on all data...")
        final_artifacts = builder.train_final_model(full_df)

    if all_fold_ics:
        full_ic_history = pd.concat(all_fold_ics).sort_values('date').drop_duplicates('date')
        full_ic_history['model_type'] = model_type
        ic_history_path = model_dir / f"{model_type}_ic_history.csv"
        full_ic_history.to_csv(ic_history_path, index=False) # 保存 IC 时也不需要索引
        return full_ic_history
        
    return pd.DataFrame()