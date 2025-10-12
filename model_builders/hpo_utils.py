# 文件路径: model/hpo_utils.py

import sys
import optuna
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from lightgbm.callback import early_stopping
from sklearn.preprocessing import StandardScaler

try:
    from model_builders.build_models import _walk_forward_split
    from model_builders.lgbm_builder import LGBMBuilder
except ImportError:
    print("WARNNING: Standard import failed in hpo_utils. Attempting to add project root to sys.path.")
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path: sys.path.append(project_root)
    from model_builders.build_models import _walk_forward_split
    from model_builders.lgbm_builder import LGBMBuilder

_hpo_cache = {}

def objective(trial, df: pd.DataFrame, config: dict, model_type: str = 'lgbm'):
    """Optuna 的目标函数，现在使用内存缓存来减少 CPU 负载。"""
    
    if model_type == 'lgbm':
        # 1. 从 config 获取 lgbm 的默认参数
        lgbm_defaults = config.get('default_model_params', {}).get('lgbm_params', {}).copy()
        
        # 2. 从 hpo_config 获取 HPO 专属参数，覆盖默认参数
        hpo_params = config.get('hpo_config', {}).get('lgbm_hpo_params', {}).copy()
        
        # 3. 合并基础参数
        base_params = {**lgbm_defaults, **hpo_params}
        
        # 4. 用 trial 建议的参数来更新
        base_params.update({
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        })

        # 5. 将最终确定的参数集放入临时的 config 字典
        hpo_trial_config = config.copy()
        hpo_trial_config['lgbm_params'] = base_params
        BuilderClass = LGBMBuilder
    else:
        raise NotImplementedError(f"HPO for model_type '{model_type}' is not implemented.")

    if df is None or df.empty: return -10.0
    
    num_eval_folds = config.get('hpo_config', {}).get('hpo_num_eval_folds', 2)
    folds = list(_walk_forward_split(df, config.get('strategy_config', {})))[-num_eval_folds:]
    if not folds: return -10.0

    ic_scores = []
    # Builder 的实例化现在非常轻量，可以在循环外进行
    builder = BuilderClass(hpo_trial_config)
    
    for i, (train_df, val_df) in enumerate(folds):
        # 为每个 fold 创建一个唯一的 cache key
        fold_key = f"fold_{i}"

        if fold_key in _hpo_cache:
            # 如果缓存命中，直接读取预处理好的数据
            cached_data = _hpo_cache[fold_key]
            X_train_scaled = cached_data['X_train_scaled']
            y_train = cached_data['y_train']
            X_val_scaled = cached_data['X_val_scaled']
            y_val = cached_data['y_val']
        else:
            # 如果缓存未命中，执行一次预处理
            # 注意：这里的代码是从 LGBMBuilder._extract_xy 和 .train_and_evaluate_fold 中提取的
            # 提取 X, y
            train_df_reset, val_df_reset = train_df.reset_index(), val_df.reset_index()
            label_col = builder.label_col
            features_for_model = [col for col in train_df.columns if col != label_col]
            
            X_train_model = train_df[features_for_model]
            y_train = train_df[label_col]
            X_val_model = val_df[features_for_model]
            y_val = val_df[label_col]
            
            # 标准化
            fold_scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(fold_scaler.fit_transform(X_train_model), index=X_train_model.index, columns=features_for_model)
            X_val_scaled = pd.DataFrame(fold_scaler.transform(X_val_model), index=X_val_model.index, columns=features_for_model)

            # 将结果存入缓存
            _hpo_cache[fold_key] = {
                'X_train_scaled': X_train_scaled,
                'y_train': y_train,
                'X_val_scaled': X_val_scaled,
                'y_val': y_val
            }
            with warnings.catch_warnings():
            # 忽略 Pandas 的 ConstantInputWarning
                warnings.filterwarnings(
                    "ignore", 
                    category=UserWarning, 
                    message="An input array is constant; the correlation coefficient is not defined."
                )
        # 模型训练部分保持不变，但现在它使用的是缓存的数据
        # 我们不再调用 builder.train_and_evaluate_fold，而是在这里直接训练
        quantile_models = {}
        for q in builder.quantiles:
            params = builder.lgbm_params.copy()
            params['alpha'] = q
            model = lgb.LGBMRegressor(**params)
                
            # 从 HPO 专属参数中获取早停轮次
            early_stopping_rounds = builder.lgbm_params.get('early_stopping_rounds', 50)
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='quantile',
                callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )
            if q == 0.5: # 我们只关心 HPO 中的中位数预测
                quantile_models['median_model'] = model

        median_model = quantile_models.get('median_model')
        if median_model and not y_val.empty and len(y_val) > 1:
            preds = median_model.predict(X_val_scaled)
            eval_df = pd.DataFrame({"pred": preds, "y": y_val.values})
            try:
                fold_ic = eval_df['pred'].rank().corr(eval_df['y'].rank(), method='spearman')
                if pd.notna(fold_ic):
                    ic_scores.append(fold_ic)
            except Exception: pass

    if not ic_scores: return -10.0
    mean_ic, std_ic = np.mean(ic_scores), np.std(ic_scores)
    icir = mean_ic / (std_ic + 1e-8)
    return icir

def run_hpo_for_ticker(df: pd.DataFrame, ticker: str, config: dict, model_type: str = 'lgbm', n_trials: int = 100):
    # --- 核心修正 3：在每次新的 HPO 运行前，清空缓存 ---
    global _hpo_cache
    _hpo_cache.clear()
    print("INFO: HPO memory cache cleared.")
    # ---

    keyword = next((s.get('keyword', ticker) for s in config.get('stocks_to_process', []) if s['ticker'] == ticker), ticker)
    print(f"\n" + "="*80); print(f"--- Starting HPO for {keyword} ({ticker}) with {n_trials} trials ---")
    study = optuna.create_study(direction="maximize", storage=None, sampler=optuna.samplers.TPESampler(seed=config.get('global_settings', {}).get('seed', 42)))
    try:
        study.optimize(lambda trial: objective(trial, df, config, model_type), n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    except Exception as e: print(f"ERROR: An exception occurred during HPO for {keyword}: {e}"); return {}
    print(f"\n--- HPO Results for {keyword} ({ticker}) ---"); print(f"Best Score (ICIR): {study.best_value:.4f}"); print("Best Parameters:"); [print(f"  {key}: {value}") for key, value in study.best_params.items()]; print("="*80)
    return study.best_params