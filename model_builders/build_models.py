# 文件路径: model/build_models.py
'''
模型整合与训练
'''
import pandas as pd
import joblib
from pathlib import Path
import sys
import torch
import gc
from tqdm.autonotebook import tqdm

# --- 健壮的导入逻辑 ---
try:
    from data_process import get_data
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder
except ImportError:
    print("WARNNING: Standard import failed. Attempting to add project root to sys.path.")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
    from data_process import get_data
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder

def _walk_forward_split(df: pd.DataFrame, config: dict):
    """
    实现基于固定滚动窗口的 Walk-Forward 分割。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame for _walk_forward_split must have a DatetimeIndex.")
        
    df = df.sort_index()
    train_window = config.get("train_window", 500)
    val_window = config.get("val_window", 60)
    dates = df.index.unique()
    
    if len(dates) < train_window + val_window:
        print(f"WARNNING: Data length ({len(dates)} days) is too short for walk-forward with train_window={train_window} and val_window={val_window}.")
        return

    start_index = train_window
    while start_index + val_window <= len(dates):
        train_end_date = dates[start_index - 1]
        val_end_date = dates[start_index + val_window - 1]
        
        train_df = df.loc[:train_end_date]
        val_df = df.loc[train_end_date:val_end_date].iloc[1:]

        if not val_df.empty:
            yield train_df, val_df
        
        start_index += val_window

def run_training_for_ticker(ticker: str, model_type: str, config: dict, force_retrain: bool = False, keyword: str = None) -> pd.DataFrame | None:
    """
    为指定的股票和模型类型，执行完整的 Walk-Forward 训练、评估和最终模型保存流程。
    """
    # 如果没有提供 keyword，就用 ticker 代替
    display_name = keyword if keyword else ticker
    
    # --- 核心修正：使用 display_name 进行打印 ---
    print(f"\n" + "="*80)
    print(f"--- Starting {model_type.upper()} training for {display_name} ({ticker}) ---")
    
    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / ticker
    model_dir.mkdir(parents=True, exist_ok=True)
    
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt'}
    model_suffix = file_suffixes[model_type]
    
    existing_models = sorted(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
    if existing_models and not force_retrain:
        print(f"INFO: Latest model for {display_name} ({ticker}) already exists. Skipping.")
        ic_history_path = model_dir / f"{model_type}_ic_history.csv"
        if ic_history_path.exists():
            return pd.read_csv(ic_history_path, index_col='date', parse_dates=True)
        return None

    # 1. 动态选择并实例化构建器
    builder_map = {
        'lgbm': LGBMBuilder,
        'lstm': LSTMBuilder
    }
    builder = builder_map[model_type](config)

    # 2. 健壮的数据加载
    try:
        # --- 核心修正：在打印信息中使用 display_name ---
        df = get_data.get_full_feature_df(ticker, config, keyword=display_name)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"No valid data returned for ticker {display_name} ({ticker})")
    except Exception as e:
        print(f"ERROR: Failed to generate features for {display_name} ({ticker}): {e}. Skipping training.")
        return None

    # 3. 执行通用的 Walk-Forward 循环
    all_fold_ics = []
    folds = list(_walk_forward_split(df, config))
    if not folds:
        print(f"WARNNING: No valid folds generated for {display_name} ({ticker}) with current config. Skipping validation.")
    else:
        print(f"INFO: Starting Walk-Forward validation for {display_name} ({ticker}) across {len(folds)} folds...")
        
        # --- 核心修正：在进度条描述中使用 display_name ---
        fold_iterator = tqdm(folds, desc=f"Training {model_type.upper()} on {display_name}", leave=True)
        
        for fold_num, (train_df, val_df) in enumerate(fold_iterator):
            # LGBM 的 builder.train_and_evaluate_fold 内部没有进度条，所以这里不打印详细信息
            # LSTM 的 builder.train_and_evaluate_fold 内部有 epoch 进度条
            
            _, ic_series_fold = builder.train_and_evaluate_fold(train_df, val_df)
            
            if ic_series_fold is not None and not ic_series_fold.empty:
                ic_series_fold['ticker'] = ticker
                all_fold_ics.append(ic_series_fold)
            
            # 强制垃圾回收，对深度学习模型尤其重要
            gc.collect()
        # --- 修正结束 ---

    # 4. 在全部数据上训练最终模型
    print(f"INFO: Training final model for {display_name} ({ticker}) on all data...")
    final_artifacts = builder.train_final_model(df)
    
    # 5. 版本化模型保存
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_file = model_dir / f"{model_type}_model_{timestamp}{model_suffix}"
    scaler_file = model_dir / f"{model_type}_scaler_{timestamp}.pkl"

    if model_type == 'lstm':
        torch.save(final_artifacts['model'].state_dict(), model_file)
    else: # lgbm
        joblib.dump(final_artifacts['models'], model_file)
    
    joblib.dump(final_artifacts['scaler'], scaler_file)
    print(f"SUCCESS: New model version for {ticker} saved: {model_file.name}")

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
            print(f"INFO: Cleaned up old model version: {old_model.name}")

    # 7. 保存并返回完整的IC历史记录
    if all_fold_ics:
        full_ic_history = pd.concat(all_fold_ics).sort_values('date').drop_duplicates('date')
        full_ic_history['model_type'] = model_type
        # 保存IC历史，以便跳过训练时也能加载
        ic_history_path = model_dir / f"{model_type}_ic_history.csv"
        full_ic_history.to_csv(ic_history_path)
        return full_ic_history
        
    return pd.DataFrame()