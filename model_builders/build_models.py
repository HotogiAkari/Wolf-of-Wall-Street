# 文件路径: model/build_models.py
'''
模型整合与训练
'''
import gc
import sys
import json
import torch
import joblib
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from typing import Any, Dict, List, Optional

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

def walk_forward_split(df: pd.DataFrame, config: dict) -> list:
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
    为单个模型执行完整的滚动训练，并支持断点续训功能。
    """
    display_name = keyword if keyword else ticker
    print(f"\n" + "="*80)
    print(f"--- 开始为 {display_name} ({ticker}) 进行 {model_type.upper()} 模型训练 ---")

    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / ticker
    model_dir.mkdir(parents=True, exist_ok=True)
    
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt'}
    model_suffix = file_suffixes.get(model_type, '.pkl')
    
    ic_history_path = model_dir / f"{model_type}_ic_history.csv"
    oof_path = model_dir / f"{model_type}_oof_preds.csv"
    progress_file = model_dir / f"_in_progress_{model_type}.json" # 遥测/进度文件
    
    # 1: 强制重训
    if force_retrain:
        print("INFO: 强制重训已开启，将删除所有旧的构件...")
        for f in [ic_history_path, oof_path, progress_file]:
            if f.exists(): f.unlink()
        model_files = list(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
        for f in model_files: f.unlink()
        start_fold_idx = 0

    # 2: 断点续训
    elif progress_file.exists():
        print(f"INFO: 检测到上次未完成的训练任务。将尝试从断点恢复...")
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            start_fold_idx = progress_data.get('completed_folds', 0)
            print(f"SUCCESS: 成功从第 {start_fold_idx + 1} 个 fold 开始续训。")
        except Exception as e:
            print(f"WARNNING: 读取进度文件失败: {e}。将从头开始训练。")
            start_fold_idx = 0
            
    # 3: 完全跳过
    elif ic_history_path.exists() and list(model_dir.glob(f"{model_type}_model_*{model_suffix}")):
        print(f"INFO: 检测到已存在的完整训练结果。跳过训练。")
        # 直接加载并返回完整的 IC 历史记录
        return pd.read_csv(ic_history_path)
    
    # 4: 全新开始
    else:
        print("INFO: 未检测到现有进度或完整模型，将从头开始全新训练。")
        start_fold_idx = 0

    builder_map = {'lgbm': LGBMBuilder, 'lstm': LSTMBuilder}
    builder = builder_map[model_type](config)
    
    if not preprocessed_folds:
        print(f"WARNNING: 未提供预处理 folds。跳过验证。")
    else:
        # 只有在需要进行训练时才打印
        if start_fold_idx < len(preprocessed_folds):
            print(f"INFO: 开始对 {display_name} 进行跨 {len(preprocessed_folds)} folds 的前向验证...")
        
            # 在全新训练开始前，创建遥测文件
            if start_fold_idx == 0:
                 with open(progress_file, 'w') as f: json.dump({'completed_folds': 0}, f)

            fold_iterator = tqdm(
                enumerate(preprocessed_folds), 
                desc=f"正在 {display_name} 上训练 {model_type.upper()}",
                total=len(preprocessed_folds),
                initial=start_fold_idx, # 让进度条从断点开始
                leave=True
            )
            
            for i, fold_data in fold_iterator:
                if i < start_fold_idx: continue # 快速跳过已完成的 folds
                
                _, ic_series_fold, oof_fold_df, fold_stats = builder.train_and_evaluate_fold(
                    train_df=None, val_df=None, cached_data=fold_data
                )
                
                # 增量写入结果
                if ic_series_fold is not None and not ic_series_fold.empty:
                    ic_series_fold.to_csv(ic_history_path, mode='a', header=not ic_history_path.exists(), index=False)
                if oof_fold_df is not None and not oof_fold_df.empty:
                    oof_fold_df.to_csv(oof_path, mode='a', header=not oof_path.exists(), index=False)
                
                # 更新遥测文件
                with open(progress_file, 'w') as f: json.dump({'completed_folds': i + 1}, f)
                
                if fold_stats:
                    postfix_str = ", ".join([f"{k}: {v}" for k, v in fold_stats.items()])
                    fold_iterator.set_postfix_str(postfix_str)
                
                gc.collect()

    if progress_file.exists():
        progress_file.unlink() # 成功结束后，删除遥测文件
        print("INFO: 滚动训练成功完成，已移除进度文件。")

    # 训练最终模型
    full_df = config.get('full_df_for_final_model')
    if full_df is not None:
        print(f"INFO: 正在训练最终模型...")
        final_artifacts = builder.train_final_model(full_df)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_file = model_dir / f"{model_type}_model_{timestamp}{model_suffix}"
        scaler_file = model_dir / f"{model_type}_scaler_{timestamp}.pkl"
        joblib.dump(final_artifacts['scaler'], scaler_file)
        if model_type == 'lstm': torch.save(final_artifacts['model'].state_dict(), model_file)
        else: joblib.dump(final_artifacts['models'], model_file)
        print(f"SUCCESS: 新版本模型已保存: {model_file.name}")
        num_to_keep = config.get('global_settings', {}).get('num_model_versions_to_keep', 3)
        all_model_versions = sorted(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
        
        if len(all_model_versions) > num_to_keep:
            versions_to_delete = all_model_versions[:-num_to_keep]
            print(f"INFO: 发现 {len(versions_to_delete)} 个旧模型版本需要清理 (保留最新的 {num_to_keep} 个)。")
            
            for old_model_path in versions_to_delete:
                # 构件对应的 scaler 文件名
                old_scaler_name = old_model_path.name.replace("model", "scaler").replace(model_suffix, ".pkl")
                old_scaler_path = old_model_path.parent / old_scaler_name
                
                try:
                    # 删除旧模型
                    old_model_path.unlink()
                    print(f"  - SUCCESS: 已清理旧模型: {old_model_path.name}")
                    # 删除对应的旧 scaler
                    if old_scaler_path.exists():
                        old_scaler_path.unlink()
                        print(f"  - SUCCESS: 已清理旧 Scaler: {old_scaler_path.name}")
                except Exception as e:
                    print(f"  - ERROR: 清理失败: {old_model_path.name}, 原因: {e}")

    if ic_history_path.exists():
        return pd.read_csv(ic_history_path)
        
    return None