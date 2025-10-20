# 文件路径: main_train.py
# 职责: 封装并执行完整的、非交互式的模型研究与训练流水线。

import os
import sys
import yaml
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import StandardScaler

# --- 0. 环境与模块加载 ---

def run_load_config_and_modules(config_path='configs/config.yaml'):
    """
    加载配置文件，并将所有必需的模块动态导入到一个字典中。
    """
    print("--- 正在初始化环境：加载配置与模块... ---")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            temp_config = yaml.safe_load(f)
        
        num_cores = temp_config.get('global_settings', {}).get('num_cpu_cores_for_data', '4')
        
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        os.environ['MKL_NUM_THREADS'] = str(num_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)
        
        print(f"INFO: 底层并行计算库线程数已设置为: {num_cores}")

    except FileNotFoundError:
        print(f"WARNNING: 未找到配置文件 '{config_path}'，无法设置并行线程数。将使用默认值。")
    except Exception as e:
        print(f"WARNNING: 读取配置文件以设置并行线程数时出错: {e}")
    
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        from data_process.get_data import initialize_apis, shutdown_apis, get_full_feature_df
        from data_process.save_data import run_data_pipeline, get_processed_data_path
        from model.build_models import run_training_for_ticker, walk_forward_split
        from model.hpo_utils import run_hpo_for_ticker
        from model.builders.model_fuser import ModelFuser
        from model.builders.lgbm_builder import LGBMBuilder
        from model.builders.lstm_builder import LSTMBuilder, LSTMModel
        from model.builders.tabtransformer_builder import TabTransformerBuilder
        from risk_management.risk_manager import RiskManager
        from backtest.backtester import VectorizedBacktester
        from backtest.event_driven_backtester import run_backtrader_backtest
        print("INFO: 项目模块导入成功。")
    except ImportError as e:
        print(f"ERROR: 模块导入失败: {e}")
        return None, None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"SUCCESS: 配置已从 '{config_path}' 加载。")
    except FileNotFoundError:
        print(f"ERROR: 配置文件未找到: {config_path}")
        return None, None

    modules = {
        'initialize_apis': initialize_apis, 'shutdown_apis': shutdown_apis,
        'get_full_feature_df': get_full_feature_df, 'run_data_pipeline': run_data_pipeline, 
        'get_processed_data_path': get_processed_data_path, 'run_training_for_ticker': run_training_for_ticker, 
        'walk_forward_split': walk_forward_split, 'run_hpo_for_ticker': run_hpo_for_ticker,
        'ModelFuser': ModelFuser, 
        'LGBMBuilder': LGBMBuilder,
        'LSTMBuilder': LSTMBuilder, 'LSTMModel': LSTMModel, 
        'TabTransformerBuilder': TabTransformerBuilder,
        'RiskManager': RiskManager, 'VectorizedBacktester': VectorizedBacktester,
        'run_backtrader_backtest': run_backtrader_backtest,
        'pd': pd, 'torch': torch, 'joblib': joblib, 'tqdm': tqdm, 'StandardScaler': StandardScaler,
        'Path': Path, 'yaml': yaml, 'json': json
    }
    return config, modules

# --- 1. 阶段一：数据流水线 ---

def run_all_data_pipeline(config: dict, modules: dict):
    """
    执行数据准备与特征工程阶段
    """
    print("=== 阶段一：数据准备与特征工程 ===")
    try:
        modules['initialize_apis'](config)
        # run_data_pipeline 会在内部动态计算并注入 'start_date' 到 config 字典中
        modules['run_data_pipeline'](config)
    except Exception as e:
        print(f"ERROR: 数据处理阶段发生严重错误: {e}")
        raise # 抛出异常以终止整个流程
    finally:
        modules['shutdown_apis']()
    print("--- 阶段一成功完成。 ---")

# --- 2. 阶段二：模型流水线 ---

def run_preprocess_l3_cache(config: dict, modules: dict, force_reprocess=False) -> dict:
    """
    执行 L3 数据预处理与缓存
    """
    print("=== 阶段 2.1：数据预加载与全局预处理 (L3 缓存) ===")
    
    # 提取所需模块和配置
    Path = modules['Path']
    joblib = modules['joblib']
    tqdm = modules['tqdm']
    pd = modules['pd']
    torch = modules['torch']
    StandardScaler = modules['StandardScaler']
    get_processed_data_path = modules['get_processed_data_path']
    walk_forward_split = modules['walk_forward_split']
    LSTMBuilder = modules['LSTMBuilder']

    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    default_model_params = config.get('default_model_params', {})
    stocks_to_process = config.get('stocks_to_process', [])

    global_data_cache = {}
    
    L3_CACHE_DIR = Path(global_settings.get('output_dir', 'data/processed'))
    L3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    L3_CACHE_PATH = L3_CACHE_DIR / "_preprocessed_cache.joblib"

    if L3_CACHE_PATH.exists() and not force_reprocess:
        print(f"INFO: 发现已存在的 L3 预处理缓存。正在从 {L3_CACHE_PATH} 加载...")
        try:
            global_data_cache = joblib.load(L3_CACHE_PATH)
            print("SUCCESS: L3 缓存已成功加载到内存。")
        except Exception as e:
            print(f"WARNNING: 加载 L3 缓存失败: {e}。将重新进行预处理。")
            global_data_cache = {}

    if not global_data_cache:
        print("INFO: L3 缓存不存在、为空或被强制重建。开始执行预处理流程...\n")
        if config and stocks_to_process:
            
            # (核心修改) 始终使用 float32 处理数据，以兼容 AMP
            torch_dtype = torch.float32 
            print(f"INFO: 所有 PyTorch 模型数据将被预处理为 {torch_dtype} 类型。")
            
            for stock_info in tqdm(stocks_to_process, desc="正在预处理股票"):
                ticker, keyword = stock_info.get('ticker'), stock_info.get('keyword', stock_info.get('ticker'))
                if not ticker: continue

                data_path = get_processed_data_path(stock_info, config)
                if not data_path.exists():
                    print(f"\n错误: 未找到 {keyword} ({ticker}) 的 L2 特征数据。跳过预处理。")
                    continue
                
                df = pd.read_pickle(data_path)
                df.index.name = 'date'
                folds = walk_forward_split(df, strategy_config)
                if not folds:
                    print(f"\n警告: 未能为 {keyword} ({ticker}) 生成任何 Folds。跳过预-处理。")
                    continue

                preprocessed_folds_lgbm, preprocessed_folds_lstm = [], []
                label_col = global_settings.get('label_column', 'label_alpha')
                features_for_model = [c for c in df.columns if c != label_col and not c.startswith('future_')]

                for train_df, val_df in folds:
                    # --- (核心修复) 使用手动、稳健的标准化 ---
                    train_mean = train_df[features_for_model].mean()
                    train_std = train_df[features_for_model].std() + 1e-8 # 关键保护

                    X_train_scaled = (train_df[features_for_model] - train_mean) / train_std
                    X_val_scaled = (val_df[features_for_model] - train_mean) / train_std

                    y_train = train_df[label_col]
                    y_val = val_df[label_col]
                    
                    # LGBM 和 TabTransformer 的数据准备
                    preprocessed_folds_lgbm.append({
                        'X_train_scaled': X_train_scaled, 'y_train': y_train, 
                        'X_val_scaled': X_val_scaled, 'y_val': y_val,
                        'feature_cols': features_for_model # 传递特征列表以备后用
                    })

                    # LSTM 数据准备
                    use_lstm = stock_info.get('use_lstm', global_settings.get('use_lstm_globally', True))
                    if 'lstm' in global_settings.get('models_to_train', []) and use_lstm:
                        lstm_seq_len = default_model_params.get('lstm_params', {}).get('sequence_length', 60)
                        
                        if len(train_df) < lstm_seq_len: continue

                        # 注意：LSTM 序列化需要使用已经标准化后的数据
                        train_df_scaled_for_lstm = X_train_scaled.copy()
                        train_df_scaled_for_lstm[label_col] = y_train
                        
                        val_df_scaled_for_lstm = X_val_scaled.copy()
                        val_df_scaled_for_lstm[label_col] = y_val

                        train_history_for_val = train_df_scaled_for_lstm.iloc[-lstm_seq_len:]
                        combined_df_for_lstm_val = pd.concat([train_history_for_val, val_df_scaled_for_lstm])
                        
                        # 实例化一次 builder 以便调用 _create_sequences
                        lstm_builder_for_seq = LSTMBuilder(config)
                        X_train_seq, y_train_seq, _ = lstm_builder_for_seq._create_sequences(train_df_scaled_for_lstm.reset_index(), features_for_model)
                        X_val_seq, y_val_seq, dates_val_seq = lstm_builder_for_seq._create_sequences(combined_df_for_lstm_val.reset_index(), features_for_model)

                        preprocessed_folds_lstm.append({
                            'X_train_tensor': torch.from_numpy(X_train_seq).to(dtype=torch_dtype),
                            'y_train_tensor': torch.from_numpy(y_train_seq).unsqueeze(1).to(dtype=torch_dtype),
                            'X_val_tensor': torch.from_numpy(X_val_seq).to(dtype=torch_dtype),
                            'y_val_tensor': torch.from_numpy(y_val_seq).unsqueeze(1).to(dtype=torch_dtype),
                            'y_val_seq': y_val_seq, 'dates_val_seq': dates_val_seq,
                            'feature_cols': features_for_model
                        })
                
                global_data_cache[ticker] = {'full_df': df, 'lgbm_folds': preprocessed_folds_lgbm, 'lstm_folds': preprocessed_folds_lstm}
            
            if not global_data_cache:
                print("WARNNING: 预处理后未能生成任何有效的缓存数据。")
            else:
                try:
                    joblib.dump(global_data_cache, L3_CACHE_PATH)
                    print(f"\nSUCCESS: L3 缓存已成功保存至 {L3_CACHE_PATH}")
                except Exception as e:
                    print(f"ERROR: 保存 L3 缓存失败: {e}")

    print("--- 阶段 2.1 成功完成。 ---")
    return global_data_cache

def run_hpo_train(config: dict, modules: dict, global_data_cache: dict):
    """
    执行超参数优化 (对应 Train.ipynb 阶段 2.2)。
    """
    print("=== 阶段 2.2：超参数优化 ===")

    # 提取所需模块和配置
    pd = modules['pd']
    Path = modules['Path']
    yaml = modules['yaml']
    run_hpo_for_ticker = modules['run_hpo_for_ticker']

    hpo_config = config.get('hpo_config', {})
    stocks_to_process = config.get('stocks_to_process', [])
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    default_model_params = config.get('default_model_params', {})
    
    # 从配置中读取要进行 HPO 的模型列表
    models_for_hpo = hpo_config.get('models_for_hpo', [])
    hpo_tickers = hpo_config.get('tickers_for_hpo', [])
    
    if not models_for_hpo:
        print("INFO: 在配置文件 hpo_config.models_for_hpo 中未指定要优化的模型，跳过此步骤。")
        print("--- 阶段 2.2 已跳过。 ---")
        return

    if not hpo_tickers:
        print("INFO: 在配置文件 hpo_config.tickers_for_hpo 中未指定用于 HPO 的股票，跳过此步骤。")
        print("--- 阶段 2.2 已跳过。 ---")
        return

    if not global_data_cache:
        print("ERROR: 全局数据缓存 (global_data_cache) 为空。请确保阶段 2.1 已成功运行。")
        print("--- 阶段 2.2 执行失败。 ---")
        return

    print(f"--- INFO: 开始为模型 {models_for_hpo} 和股票 {hpo_tickers} 进行超参数优化 ---\n")
    
    for model_type_for_hpo in models_for_hpo:
        print(f"\n" + "#"*80)
        print(f"# 开始为模型 [{model_type_for_hpo.upper()}] 进行 HPO")
        print("#"*80)
        
        hpo_results_list = []
        model_hpo_config = hpo_config.get(f'{model_type_for_hpo}_hpo_config', {})
        num_eval_folds = model_hpo_config.get('hpo_num_eval_folds', hpo_config.get('hpo_num_eval_folds', 2))

        for ticker in hpo_tickers:
            stock_info = next((s for s in stocks_to_process if s['ticker'] == ticker), None)
            if not stock_info:
                print(f"WARNNING: 未在 'stocks_to_process' 中找到 HPO 股票 {ticker} 的配置。跳过。")
                continue
            
            keyword = stock_info.get('keyword', ticker)

            use_lstm = stock_info.get('use_lstm', global_settings.get('use_lstm_globally', True))
            if model_type_for_hpo == 'lstm' and not use_lstm:
                print(f"\nINFO: {keyword} ({ticker}) 已配置为不使用 LSTM，跳过 LSTM 的 HPO。")
                continue

            if ticker not in global_data_cache:
                print(f"ERROR: 预处理数据缓存中未找到 {keyword} ({ticker}) 的数据。跳过。")
                continue

            all_preprocessed_folds = global_data_cache[ticker].get(f'{model_type_for_hpo}_folds', [])
            if not all_preprocessed_folds:
                print(f"WARNNING: 缓存中未找到 {keyword} ({ticker}) 的 '{model_type_for_hpo}' 预处理数据。跳过 HPO。")
                continue
            
            hpo_folds_data = all_preprocessed_folds[-num_eval_folds:]
            
            print(f"\nINFO: 已为 {keyword} ({ticker}) 加载最后 {len(hpo_folds_data)} 个 folds 用于 {model_type_for_hpo.upper()} HPO。")

            hpo_run_config = {
                'global_settings': global_settings, 'strategy_config': strategy_config,
                'default_model_params': default_model_params, 'stocks_to_process': [stock_info],
                'hpo_config': hpo_config
            }
            
            best_params, best_value = run_hpo_for_ticker(
                preprocessed_folds=hpo_folds_data,
                ticker=ticker,
                config=hpo_run_config,
                model_type=model_type_for_hpo
            )
            
            if best_params and best_value is not None:
                hpo_results_list.append({'ticker': ticker, 'keyword': keyword, 'best_score': best_value, **best_params})
        
        if hpo_results_list:
            # 从配置中读取 HPO 日志目录
            hpo_log_dir_name = global_settings.get('hpo_log_dir', 'hpo_logs')
            hpo_log_dir = Path(hpo_log_dir_name)
            hpo_log_dir.mkdir(exist_ok=True)
            
            hpo_best_results_path = hpo_log_dir / f"hpo_best_results_{model_type_for_hpo}.csv"
            
            current_hpo_df = pd.DataFrame(hpo_results_list).set_index('ticker')

            if hpo_best_results_path.exists():
                print(f"\nINFO: 正在加载 [{model_type_for_hpo.upper()}] 的历史最佳 HPO 结果...")
                historical_best_df = pd.read_csv(hpo_best_results_path).set_index('ticker')
                
                for ticker, current_row in current_hpo_df.iterrows():
                    if ticker not in historical_best_df.index or current_row['best_score'] > historical_best_df.loc[ticker, 'best_score']:
                        keyword = current_row.get('keyword', ticker)
                        historical_score = historical_best_df.loc[ticker, 'best_score'] if ticker in historical_best_df.index else 'N/A'
                        score_str = f'{historical_score:.4f}' if isinstance(historical_score, (int, float)) else historical_score
                        print(f"  - 新纪录! [{model_type_for_hpo.upper()}] {keyword} ({ticker}) 的最佳分数从 {score_str} 提升至 {current_row['best_score']:.4f}.")
                        historical_best_df.loc[ticker] = current_row
                final_best_df = historical_best_df
            else:
                print(f"\nINFO: 未找到 [{model_type_for_hpo.upper()}] 的历史 HPO 结果，将本次结果作为初始最佳记录。")
                final_best_df = current_hpo_df

            final_best_df.to_csv(hpo_best_results_path)
            print(f"SUCCESS: 最新的 [{model_type_for_hpo.upper()}] HPO 冠军榜已保存至 {hpo_best_results_path}")
            
            param_cols_original = [c for c in hpo_results_list[0].keys() if c not in ['ticker', 'keyword', 'best_score']]
            final_hpo_params = final_best_df[param_cols_original].mean().to_dict()
            average_best_score = final_best_df['best_score'].mean()
            
            for p in ['num_leaves', 'min_child_samples', 'units_1', 'units_2']:
                if p in final_hpo_params: final_hpo_params[p] = int(round(final_hpo_params[p]))
            
            param_key = f"{model_type_for_hpo}_params"
            config['default_model_params'][param_key].update(final_hpo_params)
            
            print(f"\n--- {model_type_for_hpo.upper()} HPO 综合结果 ---")
            print(f"本轮 HPO 冠军榜平均最高分 (ICIR): {average_best_score:.4f}")
            print(f"将用于后续训练的【{model_type_for_hpo.upper()} 平均参数】已动态更新到 config 中:")
            print(yaml.dump(config['default_model_params'][param_key], allow_unicode=True))

    print("--- 阶段 2.2 成功完成。 ---")

def run_all_models_train(config: dict, modules: dict, global_data_cache: dict, 
                     force_retrain_base=False, 
                     force_retrain_fuser=False, 
                     run_fusion=True) -> list:
    """
    执行基础模型和融合模型的训练。
    """
    print("=== 工作流阶段 2.3：训练所有模型 ===")

    # 提取所需模块和配置
    tqdm, run_training_for_ticker, ModelFuser = modules['tqdm'], modules['run_training_for_ticker'], modules['ModelFuser']
    global_settings, strategy_config, default_model_params, stocks_to_process = config.get('global_settings', {}), config.get('strategy_config', {}), config.get('default_model_params', {}), config.get('stocks_to_process', [])

    # --- 2.3.1 基础模型训练 ---
    print("\n--- 2.3.1 基础模型训练 ---")
    all_ic_history = []
    if not (config and stocks_to_process and global_data_cache):
        print("ERROR: 配置或数据缓存为空，无法训练基础模型。")
        return all_ic_history

    # 从配置中读取要训练的模型列表
    models_to_train = global_settings.get('models_to_train', ['lgbm', 'lstm'])
    stock_iterator = tqdm(stocks_to_process, desc="训练基础模型")

    for stock_info in stock_iterator:
        ticker = stock_info.get('ticker')
        if not ticker or ticker not in global_data_cache:
            continue
        
        keyword = stock_info.get('keyword', ticker)
        stock_iterator.set_description(f"正在处理 {keyword} ({ticker})")
        
        cached_stock_data = global_data_cache[ticker]
        full_df = cached_stock_data['full_df']
        
        for model_type in models_to_train:
            # 检查该模型是否被全局或个股配置启用
            use_model_flag_name = f"use_{model_type}_globally"
            use_model_per_stock_flag_name = f"use_{model_type}"
            
            # 默认启用 lgbm (因为它没有开关)
            is_enabled = True
            if model_type != 'lgbm':
                 is_enabled = stock_info.get(use_model_per_stock_flag_name, global_settings.get(use_model_flag_name, True))

            if not is_enabled:
                print(f"\nINFO: {keyword} ({ticker}) 已配置为不使用 {model_type.upper()}, 跳过训练。")
                continue

            folds_key = f"{model_type}_folds"
            # TabTransformer 复用 LGBM 的 folds 数据
            if model_type == 'tabtransformer' and folds_key not in cached_stock_data:
                folds_key = 'lgbm_folds'

            preprocessed_folds = cached_stock_data.get(folds_key)
            if not preprocessed_folds:
                print(f"\nWARNNING: 未找到 {keyword} ({ticker}) 的 '{model_type}' 预处理 folds。跳过训练。")
                continue

            run_config = {
                'global_settings': global_settings, 'strategy_config': strategy_config,
                'default_model_params': default_model_params, 'stocks_to_process': [stock_info],
                'full_df_for_final_model': full_df
            }

            ic_history = run_training_for_ticker(
                preprocessed_folds=preprocessed_folds,
                ticker=ticker, model_type=model_type, config=run_config, 
                force_retrain=force_retrain_base, keyword=keyword
            )
            
            if ic_history is not None and not ic_history.empty:
                ic_history['ticker'] = ticker
                ic_history['model_type'] = model_type
                all_ic_history.append(ic_history)

    # --- 2.3.5 融合模型训练 ---
    if run_fusion:
        print("\n--- 2.3.5 融合模型训练 ---")
        if config and stocks_to_process:
            if not force_retrain_base and not all_ic_history:
                 print("INFO: 基础模型被跳过且无历史 IC 数据，因此跳过融合模型训练。")
            else:
                fuser_iterator = tqdm(stocks_to_process, desc="训练融合模型")
                for stock_info in fuser_iterator:
                    ticker = stock_info.get('ticker')
                    keyword = stock_info.get('keyword', ticker)
                    fuser_iterator.set_description(f"训练融合模型 for {keyword} ({ticker})")
                    if not ticker: continue

                    fuser = ModelFuser(ticker, config)
                    
                    if not force_retrain_fuser and fuser.meta_path.exists():
                        print(f"INFO: {keyword} ({ticker}) 的融合模型元数据已存在。跳过训练。")
                        continue

                    fuser.train()
    else:
        print("\n--- 2.3.5 融合模型训练 (已跳过) ---")

    print("--- 阶段 2.3 成功完成。 ---")
    return all_ic_history

def run_performance_evaluation(config: dict, modules: dict, all_ic_history: list) -> tuple:
    """
    执行结果聚合与评估，计算 ICIR 和两种回测（向量化与事件驱动）的指标，但不进行可视化。
    返回包含评估结果的 DataFrames。
    """
    print("=== 工作流阶段 2.4a：计算性能评估指标 ===")

    # --- 1. 提取模块和配置 ---
    pd = modules.get('pd')
    Path = modules.get('Path')
    np = modules.get('np')
    json = modules.get('json')
    ModelFuser = modules.get('ModelFuser')
    tqdm = modules.get('tqdm')
    VectorizedBacktester = modules.get('VectorizedBacktester')
    run_backtrader_backtest = modules.get('run_backtrader_backtest') # <-- 核心修复点

    global_settings = config.get('global_settings', {})
    stocks_to_process = config.get('stocks_to_process', [])

    if not all_ic_history:
        print("\nWARNNING: 训练期间未生成任何基础模型的 IC 历史。无法进行评估。")
        return None, None, None

    # --- 1. 汇总 ICIR 性能 ---
    full_ic_df = pd.concat(all_ic_history).drop_duplicates(subset=['ticker', 'model_type', 'date'], keep='last')
    
    fused_oof_preds_list = []
    print("INFO: 正在加载融合模型 (ModelFuser) 并对 OOF 数据进行批量预测...")
    for stock_info in tqdm(stocks_to_process, desc="融合 OOF 预测 (评估)"):
        ticker, keyword = stock_info.get('ticker'), stock_info.get('keyword', stock_info.get('ticker'))
        if not ticker: continue
        try:
            fuser = ModelFuser(ticker, config)
            if not fuser.load(): continue
            
            oof_dfs = []
            model_dir = Path(global_settings.get('model_dir', 'models')) / ticker
            models_trained = global_settings.get('models_to_train', [])
            
            y_true_df_loaded = False
            for model_type in models_trained:
                oof_path = model_dir / f"{model_type}_oof_preds.csv"
                if oof_path.exists():
                    df = pd.read_csv(oof_path, parse_dates=['date'])
                    if not y_true_df_loaded:
                        oof_dfs.append(df.set_index('date')[['y_true']])
                        y_true_df_loaded = True
                    oof_dfs.append(df.set_index('date')[['y_pred']].rename(columns={'y_pred': f'pred_{model_type}'}))
            
            if len(oof_dfs) < 2: continue
            all_oof_df = pd.concat(oof_dfs, axis=1).dropna()
            if all_oof_df.empty: continue
            
            fused_preds_series = fuser.predict_batch(all_oof_df)
            fused_oof_df = pd.DataFrame({'y_pred': fused_preds_series, 'y_true': all_oof_df['y_true'], 'ticker': ticker})
            fused_oof_preds_list.append(fused_oof_df)
        except Exception as e:
            print(f"ERROR: 为 {keyword} ({ticker}) 融合 OOF 数据时出错: {e}")

    fusion_ic_list = []
    if fused_oof_preds_list:
        full_fused_oof_df = pd.concat(fused_oof_preds_list)
        base_ic_dates = pd.concat(all_ic_history)[['ticker', 'date']].drop_duplicates()
        base_ic_dates['date'] = pd.to_datetime(base_ic_dates['date'])
        base_ic_dates.set_index('date', inplace=True)
        for ticker, group_df in full_fused_oof_df.groupby('ticker'):
            ticker_fold_dates = base_ic_dates[base_ic_dates['ticker'] == ticker].index.sort_values()
            if ticker_fold_dates.empty: continue
            bins = pd.to_datetime([group_df.index.min() - pd.DateOffset(days=1)] + list(ticker_fold_dates))
            group_df['fold_period'] = pd.cut(group_df.index, bins=bins, right=True, labels=ticker_fold_dates)
            for period_end_date, fold_data in group_df.groupby('fold_period'):
                if len(fold_data) > 1:
                    rank_ic = fold_data['y_pred'].rank().corr(fold_data['y_true'].rank(), method='spearman')
                    if pd.notna(rank_ic):
                        fusion_ic_list.append({'date': period_end_date, 'rank_ic': rank_ic, 'ticker': ticker, 'model_type': 'FUSION'})
    
    final_eval_df = full_ic_df
    if fusion_ic_list:
        final_eval_df = pd.concat([full_ic_df, pd.DataFrame(fusion_ic_list)], ignore_index=True)
    final_eval_df['ticker_name'] = final_eval_df['ticker'].map({s['ticker']: s.get('keyword', s['ticker']) for s in stocks_to_process})
    
    def safe_std(x): return x.std(ddof=0) if len(x) > 1 else 0.0
    evaluation_summary = final_eval_df.groupby(['ticker_name', 'model_type'])['rank_ic'].agg(mean='mean', std=safe_std).reset_index()
    evaluation_summary['icir'] = np.where(evaluation_summary['std'] > 1e-8, evaluation_summary['mean'] / evaluation_summary['std'], evaluation_summary['mean'] * 100)
    
    print("\n" + "="*80)
    print("### 1. 模型预测能力评估 (ICIR) ###")
    print("="*80)
    print(evaluation_summary.to_string())

    # --- 2. 向量化回测 ---
    all_vectorized_results = []
    if fused_oof_preds_list and VectorizedBacktester:
        print("\n" + "="*80)
        print("### 2. 策略理论表现评估 (向量化回测) ###")
        print("="*80)
        full_fused_oof_df = pd.concat(fused_oof_preds_list)
        for ticker, stock_oof_df in full_fused_oof_df.groupby('ticker'):
            try:
                backtester = VectorizedBacktester(stock_oof_df, config)
                backtester.run()
                all_vectorized_results.append(backtester.get_results())
            except Exception as e:
                print(f"ERROR: 为 {ticker} 执行向量化回测时失败: {e}")
    
    vectorized_summary = pd.DataFrame(all_vectorized_results).set_index('Ticker') if all_vectorized_results else pd.DataFrame()
    if not vectorized_summary.empty:
        print(vectorized_summary.to_string())

    # --- 7. 事件驱动回测 ---
    if run_backtrader_backtest and fused_oof_preds_list:
        print("--- 3. 策略实盘模拟评估 (事件驱动回测) ---")
        get_processed_data_path = modules.get('get_processed_data_path')
        full_fused_oof_df = pd.concat(fused_oof_preds_list)

        for ticker, stock_oof_df in full_fused_oof_df.groupby('ticker'):
            keyword = final_eval_df[final_eval_df['ticker'] == ticker]['ticker_name'].iloc[0]
            print(f"\n--- 正在为 {keyword} ({ticker}) 执行事件驱动回测... ---")
            
            try:
                stock_info = next((s for s in stocks_to_process if s['ticker'] == ticker), None)
                if not stock_info: continue
                
                l2_path = get_processed_data_path(stock_info, config)
                if not l2_path.exists(): continue
                meta_path = l2_path.parent / 'meta.json'
                
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # 构建 L1 缓存的路径
                # 注意：此处的 'raw_{ticker}_{start}_{end}.pkl' 格式需要与 get_data.py 中的 _get_ohlcv_data_bs 函数的缓存文件名格式严格一致
                ticker_bs_format = stock_info.get('ticker', '').replace('.', '_')
                raw_data_path = Path(config['global_settings']['data_cache_dir']) / 'raw_ohlcv' / f"raw_{ticker_bs_format}_{meta['data_start_date']}_{meta['data_end_date']}.pkl"
                
                if not raw_data_path.exists():
                    print(f"WARNNING: 找不到 {keyword} ({ticker}) 的 L1 原始日线数据缓存，路径: {raw_data_path}。跳过事件驱动回测。")
                    continue
                
                daily_ohlcv = pd.read_pickle(raw_data_path)
                
                # 运行回测
                run_backtrader_backtest(daily_ohlcv, stock_oof_df, config, plot=False)

            except Exception as e:
                print(f"  - ERROR: 为 {keyword} ({ticker}) 执行事件驱动回测时失败: {e}")

    print("\n--- 评估阶段成功完成。 ---")
    
    # 将所有结果返回给主工作流
    return evaluation_summary, vectorized_summary, final_eval_df

def run_results_visualization(config: dict, modules: dict, evaluation_summary: pd.DataFrame, backtest_summary: pd.DataFrame, final_eval_df: pd.DataFrame):
    """
    (新增) 对已计算的评估结果进行可视化。
    """
    print("\n" + "="*80)
    print("=== 工作流阶段 2.4b：生成评估可视化图表 ===")
    print("="*80)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('seaborn-v0_8-whitegrid')
        if 'SimHei' in plt.rcParams.get('font.sans-serif', []): plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("WARNNING: Matplotlib 或 Seaborn 未安装，无法进行可视化。")
        return

    vis_settings = config.get('visualization_settings', {})
    custom_palette = vis_settings.get('palette', {"lgbm": "#49b6ff", "lstm": "#ffa915", "FUSION": "#2ecc71"})
    icir_plot_ylim = vis_settings.get('icir_plot_ylim', [-2.0, 2.0])

    if evaluation_summary is not None and not evaluation_summary.empty:
        print("\n--- 生成 ICIR 对比图 ---")
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.barplot(data=evaluation_summary, x='ticker_name', y='icir', hue='model_type', palette=custom_palette, ax=ax)
        # (美化代码)
        plt.show()

    if final_eval_df is not None and not final_eval_df.empty:
        print("\n--- 生成累积 IC 曲线图 ---")
        plot_df = final_eval_df.copy()
        plot_df['date'] = pd.to_datetime(plot_df['date'])
        plot_df.sort_values('date', inplace=True)
        if 'ticker_name' in plot_df.columns and 'model_type' in plot_df.columns:
            plot_df['cumulative_ic'] = plot_df.groupby(['ticker_name', 'model_type'])['rank_ic'].cumsum()
            plt.figure(figsize=(16, 9))
            sns.lineplot(data=plot_df, x='date', y='cumulative_ic', hue='ticker_name', style='model_type')
            plt.show()

    if backtest_summary is not None and not backtest_summary.empty:
        print("\n--- 生成策略业绩对比图 (夏普比率) ---")
        plot_df = backtest_summary.copy()
        plot_df['Sharpe Ratio'] = pd.to_numeric(plot_df['Sharpe Ratio'], errors='coerce')
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.barplot(x=plot_df.index, y='Sharpe Ratio', data=plot_df, ax=ax)
        plt.show()

    print("--- 阶段 2.5 成功完成。 ---")

def run_single_stock_prediction(config: dict, modules: dict, target_ticker: str = None):
    """
    为单只股票执行完整的“加载-预测-决策”流程。
    对应原 Prophet.ipynb 的功能。
    """
    print("### 主工作流：启动单点预测 ###")

    # --- 提取模块和配置 ---
    pd, torch, joblib, Path, json = modules['pd'], modules['torch'], modules['joblib'], modules['Path'], modules['json']
    LSTMModel, ModelFuser, RiskManager = modules['LSTMModel'], modules['ModelFuser'], modules['RiskManager']
    initialize_apis, shutdown_apis, get_full_feature_df = modules['initialize_apis'], modules['shutdown_apis'], modules['get_full_feature_df']
    
    # --- 1. 确定目标股票 ---
    if target_ticker is None:
        target_ticker = config.get('application_settings', {}).get('prophet_target_ticker')
    if not target_ticker:
        print("ERROR: 未在配置或参数中指定要预测的目标股票。工作流终止。")
        return

    stock_info = next((s for s in config.get('stocks_to_process', []) if s['ticker'] == target_ticker), None)
    if not stock_info:
        print(f"ERROR: 在配置文件中未找到股票 {target_ticker} 的信息！工作流终止。")
        return
    
    keyword = stock_info.get('keyword', target_ticker)
    print(f"--- 目标股票已设定: {keyword} ({target_ticker}) ---")

    # --- 2. 初始化并加载所有必需的构件 ---
    print("\n--- 步骤1：加载所有已训练构件 ---")
    models, scalers = {}, {}
    all_components_loaded = True
    
    try:
        model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / target_ticker
        models_to_load = config.get('global_settings', {}).get('models_to_train', ['lgbm', 'lstm'])
        
        for model_type in models_to_load:
            print(f"  - 正在加载 {model_type.upper()} 的构件...")
            model_files = sorted(model_dir.glob(f"{model_type}_model_*.p*t"))
            if not model_files:
                print(f"    - ERROR: 未找到 {model_type.upper()} 的模型文件。")
                all_components_loaded = False; continue
            
            latest_model_file = model_files[-1]
            version_timestamp = latest_model_file.stem.split('_')[-1]
            latest_scaler_file = model_dir / f"{model_type}_scaler_{version_timestamp}.pkl"

            if model_type == 'lgbm':
                models[model_type] = joblib.load(latest_model_file)
            elif model_type == 'lstm':
                latest_meta_file = model_dir / f"{model_type}_meta_{version_timestamp}.json"
                if not latest_meta_file.exists():
                    raise FileNotFoundError(f"未找到 LSTM 的元数据文件: {latest_meta_file}")
                with open(latest_meta_file, 'r', encoding='utf-8') as f:
                    lstm_metadata = json.load(f)
                input_size = lstm_metadata.get('input_size')
                if not input_size: raise ValueError("元数据文件中缺少 'input_size'。")
                
                lstm_cfg = {**config.get('default_model_params',{}), **stock_info}.get('lstm_params',{})
                model_instance = LSTMModel(input_size=input_size, hidden_size_1=lstm_cfg.get('units_1',64), hidden_size_2=lstm_cfg.get('units_2',32), dropout=lstm_cfg.get('dropout',0.2))
                model_instance.load_state_dict(torch.load(latest_model_file))
                model_instance.eval()
                models[model_type] = model_instance
            
            scalers[model_type] = joblib.load(latest_scaler_file)
            print(f"    - SUCCESS: 成功加载版本 '{version_timestamp}' 的模型和 Scaler。")
        
        fuser_instance = ModelFuser(target_ticker, config)
        if not fuser_instance.load():
            print(f"  - WARNNING: 未能加载 ModelFuser，将回退到简单平均。")
            # 即使 fuser 没加载成功，也可以继续，后续会回退

        db_path = config.get('global_settings', {}).get('order_history_db', 'order_history.db')
        risk_manager = RiskManager(db_path=db_path)

        if not all_components_loaded and not models:
             raise RuntimeError("没有任何基础模型被成功加载。")

    except Exception as e:
        print(f"FATAL: 加载构件失败: {e}")
        return

    # --- 3. 获取最新特征数据 ---
    print("\n--- 步骤2：获取最新特征数据 ---")
    latest_features, historical_sequence_for_lstm, full_feature_df_for_risk = None, None, None
    try:
        initialize_apis(config)
        lstm_params = config.get('default_model_params', {}).get('lstm_params', {})
        seq_len = lstm_params.get('sequence_length', 60)
        required_lookback_days = seq_len + 120

        pred_config = config.copy()
        end_date_dt = pd.to_datetime(pred_config['strategy_config']['end_date'])
        required_start_date = (end_date_dt - pd.DateOffset(days=required_lookback_days)).strftime('%Y-%m-%d')
        pred_config['strategy_config']['earliest_start_date'] = min(pred_config['strategy_config'].get('earliest_start_date', '1990-01-01'), required_start_date)

        full_feature_df_for_risk = get_full_feature_df(target_ticker, pred_config, keyword=keyword, prediction_mode=True)
        
        if full_feature_df_for_risk is not None and len(full_feature_df_for_risk) >= seq_len:
            latest_features = full_feature_df_for_risk.iloc[-1:]
            historical_sequence_for_lstm = full_feature_df_for_risk.iloc[-seq_len:]
            print(f"SUCCESS: 成功获取 {keyword} ({target_ticker}) 的最新特征数据 (日期: {latest_features.index[0].date()})。")
        else:
            print(f"ERROR: 获取到的数据长度不足 {seq_len}，无法为 LSTM 生成有效的输入序列。")
    finally:
        shutdown_apis()

    if latest_features is None:
        print("ERROR: 获取最新特征数据失败。工作流终止。")
        return

    # --- 4. 生成独立模型预测 ---
    print("\n--- 步骤3：生成独立模型预测 ---")
    predictions = {}
    label_col = config.get('global_settings', {}).get('label_column', 'label_return')
    feature_cols = [c for c in latest_features.columns if c != label_col and not c.startswith('future_')]
    X_latest = latest_features[feature_cols]

    if 'lgbm' in models:
        X_scaled_lgbm = scalers['lgbm'].transform(X_latest)
        pred_lgbm = models['lgbm']['q_0.5'].predict(X_scaled_lgbm)[0]
        predictions['lgbm'] = pred_lgbm
        print(f"  - LGBM 预测值 (中位数): {pred_lgbm:.6f}")
    if 'lstm' in models and historical_sequence_for_lstm is not None:
        X_sequence_lstm = historical_sequence_for_lstm[feature_cols]
        X_scaled_lstm = scalers['lstm'].transform(X_sequence_lstm)
        X_tensor_lstm = torch.from_numpy(X_scaled_lstm).unsqueeze(0).float()
        with torch.no_grad():
            pred_lstm = models['lstm'](X_tensor_lstm).item()
        predictions['lstm'] = pred_lstm
        print(f"  - LSTM 预测值 (基于真实序列): {pred_lstm:.6f}")

    if not predictions:
        print("ERROR: 未能生成任何模型的预测。工作流终止。")
        return

    # --- 5. 模型融合 ---
    print("\n--- 步骤4：融合模型预测 ---")
    fused_prediction = None
    if fuser_instance and fuser_instance.is_trained:
        preds_dict = {f'pred_{model_type}': pred for model_type, pred in predictions.items()}
        fused_prediction = fuser_instance.predict(preds_dict)
        print(f"  - SUCCESS: ModelFuser 融合成功。")
        print(f"    - 融合后的最终预测信号 (已平滑): {fused_prediction:.6f}")
        if hasattr(fuser_instance.meta_model, 'coef_'):
            coefs = {name: val for name, val in zip(getattr(fuser_instance.meta_model, 'feature_names_in_', ['lgbm', 'lstm']), fuser_instance.meta_model.coef_)}
            print(f"    - 融合模型权重 -> {coefs}")
    else:
        print("  - WARNNING: ModelFuser 不可用或未训练。回退到对基础模型预测进行简单平均。")
        fused_prediction = np.mean(list(predictions.values()))
        print(f"    - 简单平均后的预测信号: {fused_prediction:.6f}")

    # --- 6. 风险审批与决策输出 ---
    print("\n--- 步骤5：风险审批与决策输出 ---")
    signal_threshold = config.get('strategy_config', {}).get('signal_threshold', 0.005)
    direction_str = 'BUY' if fused_prediction > signal_threshold else ('SELL' if fused_prediction < -signal_threshold else 'HOLD')
    trade_price = latest_features['close'].iloc[0]

    decision_approved = False
    decision_notes = "信号强度未达到开仓阈值。"
    order_id = None

    if direction_str in ['BUY', 'SELL']:
        print(f"INFO: 检测到开仓信号 '{direction_str}' (强度: {fused_prediction:+.4%}), 提交至 RiskManager 审批...")
        decision_approved, order_id = risk_manager.approve_trade(
            ticker=target_ticker, direction=direction_str, price=trade_price,
            latest_market_data=full_feature_df_for_risk, config=config
        )
        if decision_approved:
            decision_notes = f"信号通过所有风险检查。已创建待处理订单。"
            print(f"  - SUCCESS: {decision_notes} (Order ID: {order_id})")
        else:
            decision_notes = "信号被 RiskManager 拒绝（详情请见上方日志）。"
            print(f"  - FAILED: {decision_notes}")
    else:
        print("INFO: 当前信号为 'HOLD'，无需进行开仓审批。")

    # --- 7. 生成决策报告 ---
    print("\n--- 最终决策报告 ---")
    final_direction = "看涨 (BUY)" if direction_str == 'BUY' else ("看跌 (SELL)" if direction_str == 'SELL' else "中性 (HOLD)")
    trade_action = "【批准开仓】" if decision_approved else ("【信号被拒】" if direction_str in ['BUY', 'SELL'] else "【无需操作】")

    report_data = [
        ('股票名称', f"{keyword} ({target_ticker})"),
        ('决策生成时间', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('信号方向', final_direction),
        ('信号强度', f"{fused_prediction:+.4%}"),
        ('交易动作', f"{trade_action} {final_direction if decision_approved else ''}"),
        ('备注', decision_notes),
        ('关联订单ID', order_id if order_id else 'N/A'),
    ]
    report_df = pd.DataFrame(report_data, columns=['项目', '内容']).set_index('项目')
    
    # 为了在非 Notebook 环境中也能清晰显示，我们使用 to_string()
    print(report_df.to_string())
    
    print(f"\n### 单点预测工作流已成功为 {keyword} ({target_ticker}) 执行完毕！ ###")


# --- 主执行函数 ---

# --- 1. 训练工作流 ---
def run_complete_training_workflow(
    config: dict, 
    modules: dict, 
    force_reprocess_l3=False, 
    run_hpo=False, 
    force_retrain_base=False, 
    force_retrain_fuser=False,
    run_fusion=True,
    run_evaluation=True,
    run_visualization=True
):
    """
    按顺序执行完整的、非交互式的训练流水线。
    """
    """
    (主函数1) 按顺序执行完整的、非交互式的训练工作流。
    """
    print("### 主工作流：启动完整模型训练 ###")
    try:
        run_all_data_pipeline(config, modules)
        
        global_data_cache = run_preprocess_l3_cache(config, modules, force_reprocess=force_reprocess_l3)
        if not global_data_cache:
            print("ERROR: L3 数据缓存为空，无法继续。工作流终止。")
            return

        if run_hpo:
            run_hpo_train(config, modules, global_data_cache)
        
        all_ic_history = run_all_models_train(
            config, modules, global_data_cache, 
            force_retrain_base=force_retrain_base, 
            force_retrain_fuser=force_retrain_fuser,
            run_fusion=run_fusion
        )
        
        if run_evaluation and all_ic_history:
            evaluation_summary, backtest_summary, final_eval_df = run_performance_evaluation(config, modules, all_ic_history)
            
            if run_visualization and (evaluation_summary is not None or backtest_summary is not None):
                run_results_visualization(config, modules, evaluation_summary, backtest_summary, final_eval_df)
                
        print("\n### 完整训练工作流已成功执行完毕！ ###")
    except Exception as e:
        print(f"\nFATAL: 训练工作流在执行过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

# --- 2. 批量预测 ---
def run_batch_prediction_workflow(config: dict, modules: dict):
    """
    为股票池中的所有股票执行单点预测。
    """
    print("### 主工作流：启动批量预测 ###")

    stocks_to_predict = config.get('stocks_to_process', [])
    if not stocks_to_predict:
        print("WARNNING: 配置文件中的 'stocks_to_process' 为空，没有可预测的股票。")
        return

    successful_predictions = 0
    for stock_info in stocks_to_predict:
        ticker = stock_info.get('ticker')
        if not ticker:
            continue
        
        try:
            # 复用我们已经写好的单点预测函数！
            run_single_stock_prediction(config, modules, target_ticker=ticker)
            successful_predictions += 1
        except Exception as e:
            keyword = stock_info.get('keyword', ticker)
            print(f"\n--- ERROR: 在为 {keyword} ({ticker}) 进行预测时发生严重错误 ---")
            print(e)
            # 即使一只股票失败，也继续处理下一只
            continue
            
    print(f"\n### 批量预测工作流执行完毕。成功预测 {successful_predictions} / {len(stocks_to_predict)} 只股票。 ###")

# --- 3. 自动化更新工作流 ---
def run_periodic_retraining_workflow(config: dict, modules: dict):
    """
    (主函数3) 执行周期性的、自动化的模型再训练工作流。
    """
    print("\n" + "#"*80)
    print("### 主工作流：启动周期性再训练 ###")
    print("#"*80)

    # 1. 动态更新配置
    print("\n--- 步骤1：动态更新配置 ---")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    config['strategy_config']['end_date'] = today_str
    print(f"SUCCESS: 配置中的 'end_date' 已动态更新为: {today_str}")

    # 2. 调用核心训练工作流
    print("\n--- 步骤2：启动完整再训练流水线 ---")
    run_complete_training_workflow(
        config=config, 
        modules=modules,
        force_reprocess_l3=True,        # 强制重建 L3 缓存以包含最新数据
        run_hpo=False,                  # 周期性运行时通常跳过耗时的 HPO
        force_retrain_base=True,        # 强制重训基础模型
        force_retrain_fuser=True,       # 强制重训融合模型
        run_evaluation=False            # 自动化运行时通常不需要生成图表
    )

# --- 命令行执行入口 ---

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="【量化模型核心引擎】一个集成了训练、预测和自动化更新功能的多功能工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('workflow', choices=['train', 'predict', 'retrain', 'batch_predict'], help="要执行的工作流")
    parser.add_argument('--ticker', type=str, help="（仅用于 'predict'）目标股票代码")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="指定配置文件路径")
    parser.add_argument('--models', nargs='+', help="（仅用于 'predict'）指定只使用哪些模型")
    parser.add_argument('--no-viz', action='store_true', help="（仅用于 'train'）执行训练，但不生成可视化图表")
    parser.add_argument('--no-fusion', action='store_true', help="（仅用于 'train'）执行基础模型训练，但不训练融合模型")
    
    args = parser.parse_args()

    print("--- 启动量化模型核心引擎 ---")
    config, modules = run_load_config_and_modules(config_path=args.config)
    
    if not (config and modules):
        print("FATAL: 环境初始化失败，无法执行任何工作流。")
        sys.exit(1)

    try:
        if args.workflow == 'train':
            run_complete_training_workflow(
                config=config, modules=modules,
                run_hpo=True,
                force_reprocess_l3=False,
                force_retrain_base=False,
                force_retrain_fuser=False,
                run_fusion=not args.no_fusion,
                run_evaluation=True,
                run_visualization=not args.no_viz
            )
        
        elif args.workflow == 'predict':
            ticker_to_predict = args.ticker or config.get('application_settings', {}).get('prophet_target_ticker')
            if ticker_to_predict:
                run_single_stock_prediction(config, modules, target_ticker=ticker_to_predict, use_specific_models=args.models)
            else:
                print("ERROR: 未指定要预测的股票。")

        elif args.workflow == 'retrain':
            run_periodic_retraining_workflow(config, modules)
        
        elif args.workflow == 'batch_predict':
            run_batch_prediction_workflow(config, modules)

    except Exception as e:
        print(f"\nFATAL: 在执行工作流 '{args.workflow}' 期间发生顶级异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- 引擎工作流执行完毕。 ---")