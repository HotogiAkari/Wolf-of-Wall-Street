# 文件路径: model/hpo_utils.py

import sys
import optuna
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from lightgbm.callback import early_stopping

try:
    from model_builders.lgbm_builder import LGBMBuilder
except ImportError:
    print("WARNNING: Standard import failed in hpo_utils. Attempting to add project root to sys.path.")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path: sys.path.append(project_root)
    from model_builders.lgbm_builder import LGBMBuilder

_hpo_cache = {}

def objective(trial, preprocessed_folds: list, config: dict, model_type: str = 'lgbm'):
    """
    (已重构) Optuna 目标函数，直接在预处理好的 folds 上进行评估。
    """
    
    if model_type == 'lgbm':
        lgbm_base_params = config.get('default_model_params', {}).get('lgbm_params', {}).copy()
        hpo_params = config.get('hpo_config', {}).get('lgbm_hpo_params', {}).copy()
        base_params = {**lgbm_base_params, **hpo_params}
        base_params.update({
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        })
        # 创建一个临时的 config 字典，用于实例化 Builder
        hpo_trial_config = config.copy()
        hpo_trial_config['lgbm_params'] = base_params
        BuilderClass = LGBMBuilder
    else:
        raise NotImplementedError(f"HPO for model_type '{model_type}' is not implemented.")

    if not preprocessed_folds:
        return -10.0

    ic_scores = []
    # Builder 现在主要用于获取 quantiles, label_col 等信息，以及调用其训练方法
    builder = BuilderClass(hpo_trial_config)
    
    # HPO 不再需要自己的缓存，因为它接收的已经是最终数据
    for fold_data in preprocessed_folds:
        # 直接调用 builder 的训练方法，传入预处理好的数据
        # 注意：HPO 期间，我们不需要 train_df, val_df，所以可以传 None
        _, ic_series = builder.train_and_evaluate_fold(
            train_df=None, val_df=None, cached_data=fold_data
        )
        if ic_series is not None and not ic_series.empty:
            # .iloc[0] 获取 Series 中的唯一值
            ic_scores.append(ic_series['rank_ic'].iloc[0])

    if not ic_scores or len(ic_scores) < 2:
        return -10.0
        
    mean_ic = np.mean(ic_scores)
    std_ic = np.std(ic_scores)
    
    if std_ic < 1e-8:
        return mean_ic * 10 if mean_ic > 0 else -10
        
    icir = mean_ic / std_ic
    return icir

def run_hpo_for_ticker(preprocessed_folds: list, ticker: str, config: dict, model_type: str = 'lgbm', n_trials: int = 100) -> tuple:
    """
    (已重构) 为指定的股票运行 HPO，直接使用传入的预处理 folds 数据。
    """
    keyword = next((s.get('keyword', ticker) for s in config.get('stocks_to_process', []) if s['ticker'] == ticker), ticker)
    
    print(f"\n--- 开始为 {keyword} ({ticker}) 进行 HPO (共 {n_trials} 轮) ---")
    
    study = optuna.create_study(direction="maximize", storage=None, sampler=optuna.samplers.TPESampler(seed=config.get('global_settings', {}).get('seed', 42)))
    
    try:
        # 将 preprocessed_folds 传递给 lambda 函数
        study.optimize(
            lambda trial: objective(trial, preprocessed_folds, config, model_type), 
            n_trials=n_trials, 
            n_jobs=1, 
            show_progress_bar=True
        )
    except Exception as e: 
        print(f"错误: 在为 {keyword} 进行 HPO 时发生异常: {e}")
        return {}, None
        
    PARAM_MAP_CN = {
        'num_leaves': '叶子节点数', 'learning_rate': '学习率',
        'min_child_samples': '叶节点最小样本数', 'feature_fraction': '特征采样比例',
        'bagging_fraction': '数据采样比例', 'reg_alpha': 'L1正则化', 'reg_lambda': 'L2正则化'
    }
    
    print(f"\n--- {keyword} ({ticker}) 的 HPO 结果 ---")
    print(f"最佳分数 (ICIR): {study.best_value:.4f}")
    print("最佳参数组合:")
    for key, value in study.best_params.items():
        display_key = PARAM_MAP_CN.get(key, key)
        print(f"  {display_key}: {value}")
    
    return study.best_params, study.best_value