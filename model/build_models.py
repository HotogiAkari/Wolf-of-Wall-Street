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
    from model.builders.lgbm_builder import LGBMBuilder
    from model.builders.lstm_builder import LSTMBuilder
    from model.builders.tabtransformer_builder import TabTransformerBuilder
except ImportError:
    print("WARNNING: 标准导入失败. 正在尝试将项目根目录添加到 sys.path.")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
    from data_process import get_data
    from model.builders.lgbm_builder import LGBMBuilder
    from model.builders.lstm_builder import LSTMBuilder
    from model.builders.tabtransformer_builder import TabTransformerBuilder

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
    (已最终修复) 为单个模型执行完整的滚动训练。
    - 支持强制重训 (force_retrain=True)。
    - 支持中断后继续训练 (通过 _in_progress.json)。
    - 支持在已有结果上进行追加训练 (通过 ic_history.csv)。
    """
    display_name = keyword if keyword else ticker
    print(f"--- 开始为 {display_name} ({ticker}) 进行 {model_type.upper()} 模型训练 ---")

    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / ticker
    model_dir.mkdir(parents=True, exist_ok=True)
    
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt', 'tabtransformer': '.pt'}
    model_suffix = file_suffixes.get(model_type, '.pkl')
    
    ic_history_path = model_dir / f"{model_type}_ic_history.csv"
    oof_path = model_dir / f"{model_type}_oof_preds.csv"
    progress_file = model_dir / f"_in_progress_{model_type}.json"
    
    start_fold_idx = 0
    final_model_pending = False

    # --- 1. 决定训练模式 ---
    # 模式一: 强制重训
    if force_retrain:
        print("INFO: 强制重训已开启，将删除所有旧的构件...")
        for f in [ic_history_path, oof_path, progress_file]:
            f.unlink(missing_ok=True)
        # 删除所有旧版本相关文件
        for f in model_dir.glob(f"{model_type}_*"):
            f.unlink(missing_ok=True)
        start_fold_idx = 0

    # 模式二: 优先处理中断的任务
    elif progress_file.exists():
        print(f"INFO: 检测到上次未完成的训练任务。将从断点恢复...")
        try:
            with open(progress_file, 'r') as f: progress_data = json.load(f)
            if progress_data.get('status') == 'final_model_pending':
                final_model_pending = True
                start_fold_idx = len(preprocessed_folds) or 0
                print("SUCCESS: 滚动训练已完成，直接进入最终模型训练阶段。")
            else:
                start_fold_idx = progress_data.get('completed_folds', 0)
                print(f"SUCCESS: 成功从第 {start_fold_idx + 1} 个 fold 开始续训。")
        except Exception as e:
            print(f"WARNNING: 读取进度文件失败: {e}。将从头开始。"); start_fold_idx = 0
            
    # 模式三: 实现追加模式
    elif ic_history_path.exists():
        try:
            completed_folds_df = pd.read_csv(ic_history_path)
            num_completed = len(completed_folds_df)
            total_new_folds = len(preprocessed_folds) or 0
            
            if num_completed >= total_new_folds:
                print(f"INFO: 已有的训练结果 ({num_completed} folds) 已是最新。跳过滚动训练。")
                final_model_pending = True
                start_fold_idx = num_completed
            else:
                start_fold_idx = num_completed
                print(f"SUCCESS: 检测到 {start_fold_idx} 个已完成的 folds。将从第 {start_fold_idx + 1} 个 fold 开始【追加】训练。")
                
                # 删除旧的最终模型及其构件，因为它们即将过时
                print("INFO: 即将进行追加训练，旧的最终模型及其构件将被删除以便后续重新生成。")
                for f in model_dir.glob(f"{model_type}_model_*"): f.unlink(missing_ok=True)
                for f in model_dir.glob(f"{model_type}_scaler_*"): f.unlink(missing_ok=True)
                for f in model_dir.glob(f"{model_type}_meta_*"): f.unlink(missing_ok=True)
                for f in model_dir.glob(f"{model_type}_encoders_*"): f.unlink(missing_ok=True)

        except Exception as e:
            print(f"WARNNING: 读取历史 IC 文件以确定断点时失败: {e}。将从头开始。"); start_fold_idx = 0
            
    # 模式四: 全新训练
    else:
        print("INFO: 未检测到任何历史记录，将从头开始全新训练。")
        start_fold_idx = 0

    # --- 2. 动态导入 Builder ---
    from model.builders.lgbm_builder import LGBMBuilder
    from model.builders.lstm_builder import LSTMBuilder
    from model.builders.tabtransformer_builder import TabTransformerBuilder
    builder_map = {'lgbm': LGBMBuilder, 'lstm': LSTMBuilder, 'tabtransformer': TabTransformerBuilder}
    builder = builder_map[model_type](config)
    
    # --- 3. 滚动训练 ---
    if not final_model_pending:
        if preprocessed_folds and start_fold_idx < len(preprocessed_folds):
            print(f"INFO: 开始对 {display_name} ({ticker}) 进行跨 {len(preprocessed_folds)} folds 的前向验证...")
            if start_fold_idx == 0 and not progress_file.exists():
                 with open(progress_file, 'w') as f: json.dump({'completed_folds': 0, 'status': 'in_progress'}, f)

            fold_iterator = tqdm(
                enumerate(preprocessed_folds), 
                desc=f"正在 {display_name} 上训练 {model_type.upper()}",
                total=len(preprocessed_folds), initial=start_fold_idx, leave=True
            )
            
            for i, fold_data in fold_iterator:
                if i < start_fold_idx: continue
                
                artifacts, ic_series_fold, oof_fold_df, fold_stats = builder.train_and_evaluate_fold(
                    train_df=None, val_df=None, cached_data=fold_data
                )
                
                if ic_series_fold is not None and not ic_series_fold.empty:
                    ic_series_fold.to_csv(ic_history_path, mode='a', header=not ic_history_path.exists(), index=False)
                if oof_fold_df is not None and not oof_fold_df.empty:
                    oof_fold_df.to_csv(oof_path, mode='a', header=not oof_path.exists(), index=False)
                
                with open(progress_file, 'w') as f: json.dump({'completed_folds': i + 1, 'status': 'in_progress'}, f)
                
                if fold_stats:
                    fold_iterator.set_postfix_str(", ".join([f"{k}: {v}" for k, v in fold_stats.items()]))
                
                gc.collect()

        with open(progress_file, 'w') as f:
            json.dump({'status': 'final_model_pending'}, f)
        print("INFO: 滚动训练成功完成，准备训练最终模型。")

    # --- 4. 训练最终模型 ---
    full_df = config.get('full_df_for_final_model')
    if full_df is not None:
        print(f"INFO: 正在训练最终模型...")
        final_artifacts = builder.train_final_model(full_df)
        
        end_date_str = config.get('strategy_config', {}).get('end_date', 'unknown_date')
        version_date = pd.to_datetime(end_date_str).strftime('%Y%m%d')

        model_file = model_dir / f"{model_type}_model_{version_date}{model_suffix}"
        
        if model_file.exists():
            print(f"INFO: {version_date} 版本的模型已存在。将被覆盖。")

        scaler_file = model_dir / f"{model_type}_scaler_{version_date}.pkl"
        joblib.dump(final_artifacts['scaler'], scaler_file)
        
        if model_type in ['lstm', 'tabtransformer']:
            torch.save(final_artifacts['model'].state_dict(), model_file)
            meta_file = model_dir / f"{model_type}_meta_{version_date}.json"
            with open(meta_file, 'w', encoding='utf-8') as f: json.dump(final_artifacts['metadata'], f, indent=4)
            if 'encoders' in final_artifacts:
                encoders_file = model_dir / f"{model_type}_encoders_{version_date}.pkl"
                joblib.dump(final_artifacts['encoders'], encoders_file)
        else:
            joblib.dump(final_artifacts['models'], model_file)
        
        print(f"SUCCESS: 新版本 ({version_date}) 模型已保存: {model_file.name}")
        
        # 清理旧版本
        num_to_keep = config.get('global_settings', {}).get('num_model_versions_to_keep', 3)
        all_model_files = sorted(model_dir.glob(f"{model_type}_model_*.p*t"))
        
        if len(all_model_files) > num_to_keep:
            files_to_delete = all_model_files[:-num_to_keep]
            for old_model_file in files_to_delete:
                try:
                    old_version_date = old_model_file.stem.split('_')[-1]
                    # 删除所有关联文件
                    (model_dir / f"{model_type}_scaler_{old_version_date}.pkl").unlink(missing_ok=True)
                    (model_dir / f"{model_type}_meta_{old_version_date}.json").unlink(missing_ok=True)
                    (model_dir / f"{model_type}_encoders_{old_version_date}.pkl").unlink(missing_ok=True)
                    old_model_file.unlink(missing_ok=True)
                except Exception: pass
            print(f"INFO: 已清理 {len(files_to_delete)} 个旧模型版本。")

    # --- 5. 最终清理 ---
    if progress_file.exists():
        progress_file.unlink() 
        print("INFO: 整个训练流程（包括最终模型）成功完成，已移除进度文件。")

    if ic_history_path.exists():
        try:
            return pd.read_csv(ic_history_path)
        except Exception: return None
        
    return None