# 文件路径: model/hpo_utils.py

import optuna
import numpy as np
import sys
from pathlib import Path

# --- 健壮的导入逻辑 ---
try:
    from model_builders.build_models import _walk_forward_split
    from data_process import get_data
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder
except ImportError:
    print("WARNNING: Standard import failed. Attempting to add project root to sys.path.")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
    from model_builders.build_models import _walk_forward_split
    from data_process import get_data
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder


def objective(trial, ticker: str, config: dict, model_type: str = 'lgbm'):
    """
    Optuna 的目标函数，用于评估一套超参数的好坏。
    """
    hpo_config = config.copy()
    
    # 1. 定义搜索空间
    if model_type == 'lgbm':
        lgbm_params = hpo_config.get('lgbm_params', {}).copy()
        lgbm_params.update({
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        })
        hpo_config['lgbm_params'] = lgbm_params
        BuilderClass = LGBMBuilder
    else:
        # 在此可以为 LSTM 添加超参数搜索空间
        raise NotImplementedError(f"HPO for model_type '{model_type}' is not implemented.")

    # 2. 快速评估
    try:
        df = get_data.get_full_feature_df(ticker, config)
        if df is None or df.empty:
            print("WARNNING: No data for HPO objective.")
            return -10.0
    except Exception as e:
        print(f"ERROR: Data generation failed in HPO objective: {e}")
        return -10.0

    num_eval_folds = config.get('hpo_num_eval_folds', 3)
    folds = list(_walk_forward_split(df, config))[-num_eval_folds:]
    
    if not folds:
        print("WARNNING: Not enough data to generate folds for HPO.")
        return -10.0

    ic_scores = []
    builder = BuilderClass(hpo_config)
    
    for train_df, val_df in folds:
        _, ic_series = builder.train_and_evaluate_fold(train_df, val_df)
        if ic_series is not None and not ic_series.empty:
            ic_scores.append(ic_series['rank_ic'].mean())

    if not ic_scores or np.isnan(ic_scores).any():
        return -10.0
        
    mean_ic = np.mean(ic_scores)
    std_ic = np.std(ic_scores)
    
    # 使用 ICIR (信息比率) 作为优化目标
    icir = mean_ic / (std_ic + 1e-8)
    
    return icir

def run_hpo_for_ticker(ticker: str, config: dict, model_type: str = 'lgbm', n_trials: int = 100):
    """
    为指定的股票和模型类型运行超参数优化。
    """
    print(f"\n" + "="*80)
    print(f"--- Starting HPO for {ticker} ({model_type}) with {n_trials} trials ---")
    
    study = optuna.create_study(
        direction="maximize",
        storage=None,
        sampler=optuna.samplers.TPESampler(seed=config.get('global_settings', {}).get('seed', 42))
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, ticker, config, model_type), 
            n_trials=n_trials,
            n_jobs=-1,
            show_progress_bar=True
        )
    except Exception as e:
        print(f"ERROR: An exception occurred during HPO for {ticker}: {e}")
        return {}

    print(f"\n--- HPO Results for {ticker} ({model_type}) ---")
    print(f"Best Score (ICIR): {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    return study.best_params