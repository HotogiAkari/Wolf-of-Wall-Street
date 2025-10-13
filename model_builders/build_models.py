# 文件路径: model/build_models.py
'''
模型整合与训练
'''
import gc
import sys
import torch
import joblib
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from typing import Optional, List, Dict, Any

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
    display_name = keyword if keyword else ticker
    print(f"--- 开始 {display_name} ({ticker}) 的模型训练: {model_type.upper()} ---")

    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / ticker
    model_dir.mkdir(parents=True, exist_ok=True)
    
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt'}; model_suffix = file_suffixes[model_type]
    
    existing_models = sorted(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
    if existing_models and not force_retrain:
        print(f"INFO: {display_name} ({ticker}) 的最新模型已退出. 跳过.")
        ic_history_path = model_dir / f"{model_type}_ic_history.csv"
        if ic_history_path.exists(): return pd.read_csv(ic_history_path, index_col='date', parse_dates=True)
        return None

    builder_map = {'lgbm': LGBMBuilder, 'lstm': LSTMBuilder}
    builder = builder_map[model_type](config)
    
    all_fold_ics = []
    
    if not preprocessed_folds:
        print(f"WARNNING: 没有为 {display_name} 提供的预处理 folds. 跳过验证.")
    else:
        print(f"INFO: Starting Walk-Forward validation for {display_name} across {len(preprocessed_folds)} folds...")
        fold_iterator = tqdm(preprocessed_folds, desc=f"Training {model_type.upper()} on {display_name}", leave=True)
        
        for fold_data in fold_iterator:
            _, ic_series_fold = builder.train_and_evaluate_fold(
                train_df=None, val_df=None, cached_data=fold_data
            )
            
            if ic_series_fold is not None and not ic_series_fold.empty:
                ic_series_fold['ticker'] = ticker; all_fold_ics.append(ic_series_fold)
            
            gc.collect()

    full_df = config.get('full_df_for_final_model')
    if full_df is None:
        print("ERROR: 在最终训练的配置中未找到 full_df_for_final_model. 跳过.")
    else:
        print(f"INFO: 训练 {display_name} ({ticker}) 的最终模型...")
        final_artifacts = builder.train_final_model(full_df)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_file = model_dir / f"{model_type}_model_{timestamp}{model_suffix}"
        scaler_file = model_dir / f"{model_type}_scaler_{timestamp}.pkl"

        if model_type == 'lstm': torch.save(final_artifacts['model'].state_dict(), model_file)
        else: joblib.dump(final_artifacts['models'], model_file)
        joblib.dump(final_artifacts['scaler'], scaler_file)
        print(f"SUCCESS: {display_name} 的新模型已保存: {model_file.name}")

    # 6. 清理旧版本
    num_to_keep = config.get('global_settings', {}).get('num_model_versions_to_keep', 3)
    all_model_versions = sorted(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
    if len(all_model_versions) > num_to_keep:
        for old_model in all_model_versions[:-num_to_keep]:
            old_model.unlink()
            old_scaler_name = old_model.name.replace("model", "scaler").replace(model_suffix, ".pkl")
            old_scaler_file = old_model.parent / old_scaler_name
            if old_scaler_file.exists():
                old_scaler_file.unlink()
            print(f"INFO: 已清除旧的模型版本: {old_model.name}")

    if all_fold_ics:
        full_ic_history = pd.concat(all_fold_ics).sort_values('date').drop_duplicates('date')
        full_ic_history['model_type'] = model_type
        ic_history_path = model_dir / f"{model_type}_ic_history.csv"
        full_ic_history.to_csv(ic_history_path)
        return full_ic_history
        
    return pd.DataFrame()