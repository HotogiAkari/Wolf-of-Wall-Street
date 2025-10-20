# 文件路径: model/hpo_utils.py

import sys
import optuna
import warnings
import numpy as np
from pathlib import Path
from tqdm.autonotebook import tqdm

try:
    from model.builders.lgbm_builder import LGBMBuilder
    from model.builders.lstm_builder import LSTMBuilder
except ImportError:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path: sys.path.append(project_root)
    from model.builders.lgbm_builder import LGBMBuilder
    from model.builders.lstm_builder import LSTMBuilder

# 定义“无限制”搜索时，各类参数的回退默认范围
DEFAULT_SEARCH_RANGES = {
    'num_leaves':         ["int", 10, 100],
    'learning_rate':      ["float", 1e-4, 0.1, True],
    'min_child_samples':  ["int", 5, 100],
    'feature_fraction':   ["float", 0.5, 1.0, False],
    'bagging_fraction':   ["float", 0.5, 1.0, False],
    'reg_alpha':          ["float", 1e-3, 10.0, True],
    'reg_lambda':         ["float", 1e-3, 10.0, True],
}

def objective(trial, preprocessed_folds: list, config: dict, model_type: str = 'lgbm'):
    """
    Optuna 目标函数，完全由配置文件驱动，并直接在预处理好的 folds 上进行评估。
    """
    hpo_trial_config = config.copy()
    
    model_hpo_config = config.get('hpo_config', {}).get(f'{model_type}_hpo_config', {})
    search_space = model_hpo_config.get('search_space', {})
    
    params_to_tune = {}
    for param, args in search_space.items():
        p_type, low, high = args[0], args[1], args[2]
        log = args[3] if len(args) > 3 else False

        # 实现“0代表无限制”的逻辑
        if (p_type == "int" or p_type == "float") and low == 0 and high == 0:
            if param not in DEFAULT_SEARCH_RANGES:
                print(f"WARNNING: '{param}' 的无限制范围未在 hpo_utils.py 中定义，跳过此参数。")
                continue
            default_args = DEFAULT_SEARCH_RANGES[param]
            p_type_def, low, high = default_args[0], default_args[1], default_args[2]
            log = default_args[3] if len(default_args) > 3 else False
        
        if p_type == 'float':
            params_to_tune[param] = trial.suggest_float(param, low, high, log=log)
        elif p_type == 'int':
            params_to_tune[param] = trial.suggest_int(param, low, high)
        elif p_type == 'categorical':
            params_to_tune[param] = trial.suggest_categorical(param, high) # 对于 categorical, high 是列表

    base_params = config.get('default_model_params', {}).get(f'{model_type}_params', {}).copy()
    hpo_fixed_params = model_hpo_config.get('params', {}).copy()
    final_params = {**base_params, **hpo_fixed_params, **params_to_tune}
    
    hpo_trial_config[f'{model_type}_params'] = final_params
    
    BuilderClass = LGBMBuilder if model_type == 'lgbm' else LSTMBuilder
    
    if not preprocessed_folds:
        return -10.0

    ic_scores = []
    builder = BuilderClass(hpo_trial_config)
    
    for fold_data in preprocessed_folds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # 忽略此 with 块内的所有警告
            
            # 调用 builder 的训练和评估方法，并正确解包 3 个返回值
            _, ic_series, _ = builder.train_and_evaluate_fold(
                train_df=None, val_df=None, cached_data=fold_data
            )
            
            if ic_series is not None and not ic_series.empty:
                ic_scores.append(ic_series['rank_ic'].iloc[0])
    
    if not ic_scores or len(ic_scores) < 2: # 确保至少有两个有效的 IC 值来计算 ICIR
        return -10.0
        
    mean_ic = np.mean(ic_scores)
    std_ic = np.std(ic_scores)
    
    if std_ic < 1e-8:
        return mean_ic * 10 if mean_ic > 0 else -10
        
    icir = mean_ic / std_ic
    return icir

def run_hpo_for_ticker(preprocessed_folds: list, ticker: str, config: dict, model_type: str = 'lgbm') -> tuple:
    """
    (已重构) 运行 HPO，使用 TQDM 回调函数来提供简洁、动态的进度反馈。
    """
    hpo_config = config.get('hpo_config', {})
    model_hpo_config = hpo_config.get(f'{model_type}_hpo_config', {})
    n_trials = model_hpo_config.get('n_trials', hpo_config.get('n_trials', 50))
    
    keyword = next((s.get('keyword', ticker) for s in config.get('stocks_to_process', []) if s['ticker'] == ticker), ticker)
    
    print(f"\n--- 开始为 {keyword} ({ticker}) 进行 {model_type.upper()} HPO (共 {n_trials} 轮) ---")
    
    # --- 核心修正 1：关闭 Optuna 的默认日志 ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", storage=None, sampler=optuna.samplers.TPESampler(seed=config.get('global_settings', {}).get('seed', 42)))
    
    # --- 核心修正 2：创建 TQDM 进度条和回调函数 ---
    best_value_tracker = {'value': -float('inf')}
    pbar = tqdm(total=n_trials, desc=f"HPO on {keyword}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # 每次 trial 结束都更新进度条
        pbar.update(1)
        
        # 只有在找到新的最优值时，才更新后缀信息
        if study.best_value is not None and study.best_value > best_value_tracker['value']:
            best_value_tracker['value'] = study.best_value
            pbar.set_postfix_str(f"New Best ICIR: {study.best_value:.4f} (Trial #{study.best_trial.number})", refresh=True)

    try:
        study.optimize(
            lambda trial: objective(trial, preprocessed_folds, config, model_type), 
            n_trials=n_trials, 
            n_jobs=1, 
            # 关闭 Optuna 自带的进度条，使用我们自己的
            show_progress_bar=False,
            # 传入回调函数
            callbacks=[callback]
        )
    except Exception as e: 
        print(f"错误: 在为 {keyword} 进行 HPO 时发生异常: {e}")
        pbar.close() 
        return {}, None
    
    pbar.close()
        
    PARAM_MAP_CN = {
        'num_leaves': '叶子节点数', 'learning_rate': '学习率',
        'min_child_samples': '叶节点最小样本数', 'feature_fraction': '特征采样比例',
        'bagging_fraction': '数据采样比例', 'reg_alpha': 'L1正则化', 'reg_lambda': 'L2正则化',
        'units_1': '隐藏层1单元数', 'units_2': '隐藏层2单元数', 'dropout': 'Dropout率'
    }
    
    print(f"\n--- {keyword} ({ticker}) 的 HPO 结果 ---")
    print(f"最佳分数 (ICIR): {study.best_value:.4f}")
    print("最佳参数组合:")
    for key, value in study.best_params.items():
        display_key = PARAM_MAP_CN.get(key, key)
        print(f"  {display_key}: {value}")
    
    return study.best_params, study.best_value