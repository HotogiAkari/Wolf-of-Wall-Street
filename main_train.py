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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 内部辅助函数 ---
def _find_latest_artifact_paths(model_dir: Path, model_type: str) -> dict:
    """
    在指定目录中，根据文件名中的日期版本号找到最新模型及其关联构件的路径。
    """
    os = __import__('os')
    pd = __import__('pandas')

    # 1. 根据模型类型确定文件后缀
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt', 'tabtransformer': '.pt'}
    model_suffix = file_suffixes.get(model_type, '.pkl')
    
    # 2. 查找所有匹配的模型文件
    model_files = list(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
    if not model_files:
        raise FileNotFoundError(f"未在目录 {model_dir} 中找到任何 {model_type.upper()} 的模型文件 (匹配 *{model_suffix})。")
    
    # 3. 通过解析文件名中的日期版本号进行排序，找到最新的文件
    try:
        latest_model_file = sorted(
            model_files, 
            key=lambda f: pd.to_datetime(f.stem.split('_')[-1], format='%Y%m%d')
        )[-1]
    except (IndexError, ValueError):
        # 如果排序失败（如文件名格式不符），则回退到使用文件修改时间
        print("WARNNING: 无法按日期版本号对模型文件进行排序，将回退到按文件修改时间。")
        latest_model_file = max(model_files, key=os.path.getctime)

    # 4. 从最终确定的最新文件名中提取日期版本
    try:
        version_date = latest_model_file.stem.split('_')[-1]
    except IndexError:
        raise ValueError(f"无法从文件名 '{latest_model_file.name}' 中解析出日期版本。")

    # 5. 根据日期版本，构建所有关联构件的精确文件路径
    paths = {
        'model': latest_model_file,
        'scaler': model_dir / f"{model_type}_scaler_{version_date}.pkl",
        'meta': model_dir / f"{model_type}_meta_{version_date}.json",
        'encoders': model_dir / f"{model_type}_encoders_{version_date}.pkl",
        'timestamp': version_date # 实际是 version_date
    }
    
    return paths

def _prophet_load_artifacts(config: dict, modules: dict, target_ticker: str, use_specific_models: list = None) -> dict:
    """
    为单点预测加载所有必需的、最新版本的构件。
    """
    print("\n--- 步骤1：加载所有已训练构件 (自动查找最新版本) ---")
    
    Path, os, joblib, torch, json = modules['Path'], __import__('os'), modules['joblib'], modules['torch'], modules['json']
    ModelFuser, RiskManager, LSTMModel, TabTransformerModel = modules['ModelFuser'], modules['RiskManager'], modules['LSTMModel'], modules.get('TabTransformerModel')
    
    artifacts = {'models': {}, 'scalers': {}, 'encoders': {}, 'feature_cols': None}
    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / target_ticker
    stock_info = next((s for s in config.get('stocks_to_process', []) if s['ticker'] == target_ticker), {})

    models_to_load = use_specific_models or config.get('global_settings', {}).get('models_to_train', [])
    if not models_to_load: 
        raise ValueError("没有指定要加载的模型。")

    # --- 1. 分别加载每种模型的构件 ---
    for model_type in models_to_load:
        print(f"  - 正在加载 {model_type.upper()} 的构件...")
        try:
            paths = _find_latest_artifact_paths(model_dir, model_type)
            
            if model_type == 'lgbm':
                artifacts['models'][model_type] = joblib.load(paths['model'])
            elif model_type in ['lstm', 'tabtransformer']:
                if not paths['meta'].exists(): raise FileNotFoundError(f"元数据文件丢失: {paths['meta']}")
                with open(paths['meta'], 'r') as f: metadata = json.load(f)
                model_structure = metadata.get('model_structure')
                if not model_structure: raise ValueError(f"{model_type.upper()} 元数据中缺少 'model_structure' 信息。")

                if model_type == 'lstm':
                    model_instance = LSTMModel(input_size=metadata['input_size'], **model_structure)
                elif model_type == 'tabtransformer' and TabTransformerModel:
                    model_instance = TabTransformerModel(num_continuous=metadata['input_size_cont'], cat_dims=metadata['cat_dims'], **model_structure)
                    if paths['encoders'].exists(): artifacts['encoders'][model_type] = joblib.load(paths['encoders'])
                
                model_instance.load_state_dict(torch.load(paths['model']))
                model_instance.eval()
                artifacts['models'][model_type] = model_instance
            
            artifacts['scalers'][model_type] = joblib.load(paths['scaler'])
            print(f"    - SUCCESS: {model_type.upper()} 版本 '{paths['timestamp']}' 已加载。")
        except Exception as e:
            print(f"    - ERROR: 加载 {model_type.upper()} 构件失败: {e}")

    if not artifacts.get('models'):
        raise RuntimeError("未能成功加载任何基础模型，预测流程无法继续。")

    # --- 2. 智能加载训练时的特征列表 ---
    print("  - INFO: 正在加载训练时使用的特征列表...")
    feature_cols_loaded = False
    load_priority = ['lgbm'] + [m for m in models_to_load if m != 'lgbm']
    for model_type_to_try in load_priority:
        if model_type_to_try in artifacts['models']:
            try:
                paths = _find_latest_artifact_paths(model_dir, model_type_to_try)
                if paths['meta'].exists():
                    with open(paths['meta'], 'r') as f: meta = json.load(f)
                    feature_cols = meta.get('feature_cols')
                    if feature_cols:
                        artifacts['feature_cols'] = feature_cols
                        print(f"    - SUCCESS: 已从 {model_type_to_try.upper()} 的元数据加载特征列表 ({len(feature_cols)}个)。")
                        feature_cols_loaded = True
                        break
            except Exception: continue
            
    if not feature_cols_loaded:
        raise RuntimeError("未能从任何一个模型的元数据中加载特征列表 (feature_cols)。")
    
    # --- 3. 加载通用构件 ---
    try:
        artifacts['fuser'] = ModelFuser(target_ticker, config)
        if not artifacts['fuser'].load(): print(f"  - WARNNING: 未能加载 ModelFuser，将回退到简单平均。")
        db_path = config.get('global_settings', {}).get('order_history_db', 'order_history.db')
        artifacts['risk_manager'] = RiskManager(db_path=db_path)
    except Exception as e:
        print(f"    - ERROR: 加载 Fuser 或 RiskManager 失败: {e}")

    print("--- 步骤1成功完成：所有构件已加载。 ---")
    return artifacts

def _prophet_get_latest_features(config: dict, modules: dict, target_ticker: str, keyword: str) -> pd.DataFrame:
    """
    为单点预测动态计算并获取最新的特征 DataFrame。
    内置了智能的每日全局数据缓存，避免不必要的重复下载。
    """
    print("\n--- 步骤2：准备预测所需的数据 ---")
    
    pd, Path, json = modules['pd'], modules['Path'], modules['json']
    initialize_apis, shutdown_apis, get_full_feature_df = modules['initialize_apis'], modules['shutdown_apis'], modules['get_full_feature_df']
    
    try:
        from data_process.get_data import (
            _get_us_stock_data_yf, _get_macroeconomic_data_cn, 
            _get_market_sentiment_data_ak, _generate_market_breadth_data, _get_fama_french_factors
        )
    except ImportError:
        print("WARNNING: 无法直接从 data_process.get_data 导入内部函数。")
        return None

    full_feature_df = None
    try:
        initialize_apis(config)
        
        # --- 1. 智能全局数据缓存 ---
        print("  - INFO: 检查全局数据缓存是否为最新...")
        cache_dir = Path(config.get('global_settings', {}).get("data_cache_dir", "data_cache"))
        global_cache_file = cache_dir / "_global_data_cache.pkl"
        global_meta_file = cache_dir / "_global_data_cache_meta.json"
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        market_breadth_df, external_market_df, market_sentiment_df, macro_df, factors_df = None, None, None, None, None
        
        # a. 检查缓存
        use_cache = False
        if global_meta_file.exists() and global_cache_file.exists():
            with open(global_meta_file, 'r') as f:
                meta = json.load(f)
            if meta.get('generation_date') == today_str:
                print("    - SUCCESS: 发现今天的全局数据缓存。正在从缓存加载...")
                cached_global_data = joblib.load(global_cache_file)
                market_breadth_df = cached_global_data.get('market_breadth_df')
                external_market_df = cached_global_data.get('external_market_df')
                market_sentiment_df = cached_global_data.get('market_sentiment_df')
                macro_df = cached_global_data.get('macro_df')
                factors_df = cached_global_data.get('factors_df')
                use_cache = True

        # b. 如果缓存不可用，则执行下载
        if not use_cache:
            print("    - INFO: 缓存不存在或已过期。将重新生成全局数据...")
            max_lookback_days = 365 * 2
            end_date_dt = pd.Timestamp.now()
            start_date_dt = end_date_dt - pd.DateOffset(days=max_lookback_days)
            start_date_str_global, end_date_str_global = start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d')
            
            strategy_config = config.get('strategy_config', {})
            
            market_breadth_df = _generate_market_breadth_data(start_date_str_global, end_date_str_global, cache_dir)
            
            all_external_dfs = []
            # ... (循环加载所有外部市场数据的逻辑不变)
            external_market_df = pd.concat(all_external_dfs, axis=1) if all_external_dfs else None
            
            market_sentiment_df = _get_market_sentiment_data_ak(start_date_str_global, end_date_str_global, cache_dir)
            macro_df = _get_macroeconomic_data_cn(start_date_str_global, end_date_str_global, config)
            factors_df = _get_fama_french_factors(start_date_str_global, end_date_str_global)

            # c. (新增) 保存新的全局数据缓存
            global_data_to_cache = {
                'market_breadth_df': market_breadth_df, 'external_market_df': external_market_df,
                'market_sentiment_df': market_sentiment_df, 'macro_df': macro_df, 'factors_df': factors_df
            }
            joblib.dump(global_data_to_cache, global_cache_file)
            with open(global_meta_file, 'w') as f:
                json.dump({'generation_date': today_str}, f)
            print("    - SUCCESS: 新的全局数据缓存已生成并保存。")
        
        # --- 2. 为目标股票获取特征 (使用已加载或新生成的全局数据) ---
        # 即使全局数据来自缓存，个股的最新数据仍然需要动态获取
        max_lookback_days_stock = 365 * 2
        end_date_dt_stock = pd.Timestamp.now()
        start_date_dt_stock = end_date_dt_stock - pd.DateOffset(days=max_lookback_days_stock)
        start_date_str_stock, end_date_str_stock = start_date_dt_stock.strftime('%Y-%m-%d'), end_date_dt_stock.strftime('%Y-%m-%d')

        full_feature_df = get_full_feature_df(
            ticker=target_ticker, config=config, 
            start_date_str=start_date_str_stock, end_date_str=end_date_str_stock,
            keyword=keyword, prediction_mode=True,
            market_breadth_df=market_breadth_df, external_market_df=external_market_df,
            market_sentiment_df=market_sentiment_df, macro_df=macro_df, factors_df=factors_df
        )
    finally:
        shutdown_apis()
    
    if full_feature_df is None or full_feature_df.empty:
        raise RuntimeError("获取最新特征数据失败。")
        
    print("--- 步骤2成功完成：最新特征数据已生成。 ---")
    return full_feature_df

def _prophet_generate_decision(config: dict, modules: dict, artifacts: dict, full_feature_df: pd.DataFrame, target_ticker: str, keyword: str):
    """
    (辅助函数) 执行预测、融合、风控、生成报告和可视化。
    """
    # --- 1. 提取模块和配置 ---
    pd, torch, np = modules['pd'], modules['torch'], __import__('numpy')
    
    # --- 2. (核心修复) 特征对齐 ---
    # a. 从构件中加载【训练时】最终使用的特征列表
    train_feature_cols = artifacts.get('feature_cols')
    if not train_feature_cols:
        raise RuntimeError("未能从构件中加载训练时的特征列表 (feature_cols)。请确保重新训练模型以生成包含此信息的 meta.json。")
        
    # b. 检查当前生成的特征数据是否包含了所有必需的特征
    current_cols = set(full_feature_df.columns)
    missing_features = set(train_feature_cols) - current_cols
    if missing_features:
        raise RuntimeError(f"预测时生成的特征数据缺少了训练时的关键特征: {missing_features}。这可能意味着特征计算逻辑已发生变化。")

    # c. 从当前数据中，精确地、按顺序地挑选出训练时使用的特征
    print(f"  - INFO: 成功加载训练时的 {len(train_feature_cols)} 个特征。将进行特征对齐...")
    feature_df_aligned = full_feature_df[train_feature_cols]

    # --- 3. 准备预测所需的最新数据片段 ---
    lstm_seq_len = config.get('default_model_params',{}).get('lstm_params',{}).get('sequence_length', 60)
    
    if len(feature_df_aligned) < lstm_seq_len:
        raise ValueError(f"对齐后的数据长度 ({len(feature_df_aligned)}) 不足以满足 LSTM 序列长度 ({lstm_seq_len})。")

    # 使用已对齐的数据
    historical_sequence = feature_df_aligned.iloc[-lstm_seq_len:]
    latest_features = feature_df_aligned.iloc[-1:]
    
    # --- 4. 独立模型预测 ---
    print("\n--- 步骤3：生成独立模型预测 ---")
    predictions = {}
    
    for model_type, model in artifacts['models'].items():
        try:
            if model_type == 'lgbm':
                X_scaled = artifacts['scalers']['lgbm'].transform(latest_features)
                pred = model['q_0.5'].predict(X_scaled)[0]
                predictions['lgbm'] = pred
                print(f"  - LGBM 预测值 (中位数): {pred:.6f}")
            
            elif model_type == 'lstm':
                X_scaled = artifacts['scalers']['lstm'].transform(historical_sequence)
                X_tensor = torch.from_numpy(X_scaled).unsqueeze(0).float()
                with torch.no_grad():
                    pred = model(X_tensor).item()
                predictions['lstm'] = pred
                print(f"  - LSTM 预测值 (基于真实序列): {pred:.6f}")
            
            elif model_type == 'tabtransformer':
                # TabTransformer 使用单行数据进行预测
                X_df_to_predict = latest_features
                
                # a. 使用训练时保存的编码器
                encoders = artifacts['encoders']['tabtransformer']
                X_df_encoded = X_df_to_predict.copy()
                cat_features = config.get('default_model_params',{}).get('tabtransformer_params',{}).get('categorical_features', [])
                for col in cat_features:
                    X_df_encoded[col] = encoders[col].transform(X_df_encoded[col].astype(str))
                
                # b. 标准化
                X_scaled = artifacts['scalers']['tabtransformer'].transform(X_df_encoded)
                X_df_scaled = pd.DataFrame(X_scaled, columns=X_df_encoded.columns, index=X_df_encoded.index)
                
                # c. 准备 Tensors
                cont_features = [c for c in train_feature_cols if c not in cat_features]
                X_cont = torch.from_numpy(X_df_scaled[cont_features].values).float()
                X_cat = torch.from_numpy(X_df_scaled[cat_features].values).long()

                with torch.no_grad():
                    pred = model(X_cont.to(artifacts['fuser'].device), X_cat.to(artifacts['fuser'].device)).item()
                predictions['tabtransformer'] = pred
                print(f"  - TabTransformer 预测值: {pred:.6f}")
        
        except Exception as e:
            print(f"  - WARNNING: 为模型 {model_type.upper()} 生成预测时失败: {e}")

    if not predictions:
        raise RuntimeError("未能生成任何有效的模型预测。")

    # --- 5. 模型融合 ---
    print("\n--- 步骤4：融合模型预测 ---")
    fuser_instance = artifacts['fuser']
    fused_prediction = None
    
    if fuser_instance and fuser_instance.is_trained:
        preds_dict_all = {f'pred_{mt}': p for mt, p in predictions.items()}
        # 确保只传入 Fuser 训练时用到的模型预测
        try:
            fuser_inputs = {k: v for k, v in preds_dict_all.items() if k in fuser_instance.meta_model.feature_names_in_}
            if len(fuser_inputs) == len(fuser_instance.meta_model.feature_names_in_):
                fused_prediction = fuser_instance.predict(fuser_inputs)
                print(f"  - SUCCESS: ModelFuser 已成功融合 {list(fuser_inputs.keys())}。")
            else:
                print("  - WARNNING: 提供的模型预测不全，无法使用 ModelFuser。将回退到简单平均。")
        except AttributeError:
             print("  - WARNNING: ModelFuser 的 meta_model 缺少 feature_names_in_ 属性。将回退到简单平均。")


    if fused_prediction is None:
        fused_prediction = np.mean(list(predictions.values()))
        print(f"  - INFO: 已回退到对所有可用预测进行简单平均。")
    
    print(f"    - 最终预测信号 (已平滑): {fused_prediction:.6f}")
    
    # --- 6. 风险审批与决策输出 ---
    print("\n--- 步骤5：风险审批与决策输出 ---")
    risk_manager = artifacts['risk_manager']
    signal_threshold = config.get('strategy_config', {}).get('signal_threshold', 0.005)
    direction_str = 'BUY' if fused_prediction > signal_threshold else ('SELL' if fused_prediction < -signal_threshold else 'HOLD')
    trade_price = latest_features['close'].iloc[0]

    decision_approved, order_id, decision_notes = False, None, "信号强度未达到开仓阈值。"

    if direction_str in ['BUY', 'SELL']:
        decision_approved, order_id = risk_manager.approve_trade(
            ticker=target_ticker, direction=direction_str, price=trade_price,
            latest_market_data=full_feature_df, config=config
        )
        decision_notes = "信号通过所有风险检查。" if decision_approved else "信号被 RiskManager 拒绝（详情见上方日志）。"
    
    # --- 6. 生成决策报告 ---
    print("\n--- 最终决策报告 ---")
    final_direction = "看涨 (BUY)" if direction_str == 'BUY' else ("看跌 (SELL)" if direction_str == 'SELL' else "中性 (HOLD)")
    trade_action = "【批准开仓】" if decision_approved else ("【信号被拒】" if direction_str in ['BUY', 'SELL'] else "【无需操作】")

    report_data = [
        ('股票名称', f"{keyword} ({target_ticker})"), ('决策生成时间', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('信号方向', final_direction), ('信号强度', f"{fused_prediction:+.4%}"),
        ('交易动作', f"{trade_action} {final_direction if decision_approved else ''}"), ('备注', decision_notes),
        ('关联订单ID', order_id or 'N/A'),
    ]
    report_df = pd.DataFrame(report_data, columns=['项目', '内容']).set_index('项目')
    print(report_df.to_string())
    
    # --- 7. (新增) 简单可视化 ---
    print("\n--- 步骤6：特征可视化 ---")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        key_features_to_plot = ['close', 'rsi_14', 'volatility_expansion_ratio', 'regime_is_uptrend']
        plot_features = [f for f in key_features_to_plot if f in full_feature_df.columns]
        
        if plot_features:
            plot_df = full_feature_df[plot_features].tail(100)
            fig, axes = plt.subplots(len(plot_features), 1, figsize=(15, 3 * len(plot_features)), sharex=True)
            fig.suptitle(f'{keyword} ({target_ticker}) - 关键特征观察', fontsize=16)
            for i, feature in enumerate(plot_features):
                ax = axes[i] if len(plot_features) > 1 else axes
                plot_df[feature].plot(ax=ax)
                ax.set_ylabel(feature)
                ax.axvline(plot_df.index[-1], color='r', linestyle='--', linewidth=2, label='当前预测点')
                ax.legend()
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
    except ImportError:
        print("WARNNING: Matplotlib 未安装，跳过可视化。")

def _encode_categorical_features(df_train: pd.DataFrame, df_val: pd.DataFrame, cat_features: list) -> tuple:
    """
    (公共工具函数) 对类别特征进行安全的标签编码。
    """
    encoders = {}
    df_train_encoded, df_val_encoded = df_train.copy(), df_val.copy()

    for col in cat_features:
        if col not in df_train_encoded.columns: continue
        le = LabelEncoder()
        df_train_encoded[col] = le.fit_transform(df_train_encoded[col].astype(str))
        known_classes = set(le.classes_)
        if '<unknown>' not in known_classes:
            le.classes_ = np.append(le.classes_, '<unknown>')
        df_val_encoded[col] = df_val_encoded[col].astype(str).apply(lambda x: x if x in known_classes else '<unknown>')
        df_val_encoded[col] = le.transform(df_val_encoded[col])
        encoders[col] = le
    return df_train_encoded, df_val_encoded, encoders

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
        from model.builders.utils import encode_categorical_features
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
        'encode_categorical_features': encode_categorical_features,
        'run_backtrader_backtest': run_backtrader_backtest,
        'pd': pd, 'torch': torch, 'joblib': joblib, 'tqdm': tqdm, 'StandardScaler': StandardScaler,
        'Path': Path, 'yaml': yaml, 'json': json
    }
    return config, modules

# --- 1. 阶段一：数据流水线 ---

def run_all_data_pipeline(config: dict, modules: dict, use_today_as_end_date=False):
    """
    执行数据准备与特征工程阶段
    """
    print("=== 阶段一：数据准备与特征工程 ===")
    pd = modules['pd']
    strategy_config = config.get('strategy_config', {})

    # --- (核心修改) 动态日期计算 ---
    if use_today_as_end_date:
        end_date_dt = pd.Timestamp.now()
        print(f"INFO: 已启用动态日期模式，将使用今天的日期 '{end_date_dt.strftime('%Y-%m-%d')}' 作为 end_date。")
    else:
        end_date_dt = pd.to_datetime(strategy_config['end_date'])
        print(f"INFO: 使用配置文件中的固定 end_date: '{end_date_dt.strftime('%Y-%m-%d')}'。")
    
    # 将最终确定的 end_date（字符串格式）注入/覆盖 config
    config['strategy_config']['end_date'] = end_date_dt.strftime('%Y-%m-%d')
    
    # 根据最终的 end_date 计算 start_date (逻辑不变)
    lookback_years = strategy_config.get('data_lookback_years', 10)
    earliest_start_date_dt = pd.to_datetime(strategy_config['earliest_start_date'])
    target_start_date_dt = end_date_dt - pd.DateOffset(years=lookback_years)
    start_date_dt = max(target_start_date_dt, earliest_start_date_dt)
    
    # 将计算出的 start_date 注入 config
    config['strategy_config']['start_date'] = start_date_dt.strftime('%Y-%m-%d')
    print(f"      计算得出的 start_date 为: {config['strategy_config']['start_date']}")
    
    # --- 后续的 API 调用和 run_data_pipeline 调用 (逻辑不变) ---
    try:
        modules['initialize_apis'](config)
        modules['run_data_pipeline'](config) # run_data_pipeline 现在接收一个已准备好日期的 config
    except Exception as e:
        print(f"ERROR: 数据处理阶段发生严重错误: {e}")
        raise
    finally:
        modules['shutdown_apis']()
    print("--- 阶段 1 成功完成。 ---")

# --- 2. 阶段二：模型流水线 ---

def run_preprocess_l3_cache(config: dict, modules: dict, force_reprocess=False) -> dict:
    """
    (已最终修复) 执行 L3 数据预处理与缓存。
    - 使用配置文件中定义的目录。
    - 采用分块保存策略，为每只股票单独保存 L3 缓存，以避免 MemoryError。
    - 在函数末尾重新加载所有缓存以供当次运行使用。
    """
    print("\n" + "="*80)
    print("=== 工作流阶段 2.1：为模型预处理数据 (L3 缓存) ===")
    print("="*80)
    
    # 提取模块和配置
    Path, joblib, tqdm, pd, torch, np = modules['Path'], modules['joblib'], modules['tqdm'], modules['pd'], modules['torch'], __import__('numpy')
    get_processed_data_path, walk_forward_split, LSTMBuilder = modules['get_processed_data_path'], modules['walk_forward_split'], modules['LSTMBuilder']
    encode_categorical_features = modules.get('encode_categorical_features')

    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    default_model_params = config.get('default_model_params', {})
    stocks_to_process = config.get('stocks_to_process', [])

    # --- 从配置中读取 L3 缓存目录 ---
    l3_cache_dir_path = global_settings.get('l3_cache_dir', 'data/processed/_l3_cache_by_stock')
    L3_CACHE_DIR = Path(l3_cache_dir_path)
    L3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if force_reprocess:
        print(f"INFO: 已启用强制重建，将清空 L3 缓存目录: {L3_CACHE_DIR}")
        for f in L3_CACHE_DIR.glob("*.joblib"):
            f.unlink()

    # --- 循环内处理和分块保存 ---
    print("INFO: 开始执行预处理流程 (分块保存模式)...\n")
    if config and stocks_to_process:
        for stock_info in tqdm(stocks_to_process, desc="正在预处理股票"):
            ticker, keyword = stock_info.get('ticker'), stock_info.get('keyword', stock_info.get('ticker'))
            if not ticker: continue
            
            stock_l3_cache_path = L3_CACHE_DIR / f"{ticker}.joblib"
            if not force_reprocess and stock_l3_cache_path.exists():
                continue
            
            data_path = get_processed_data_path(stock_info, config)
            if not data_path.exists():
                print(f"\n错误: 未找到 {keyword} ({ticker}) 的 L2 特征数据。跳过预处理。")
                continue
            
            df = pd.read_pickle(data_path)
            df.index.name = 'date'
            folds = walk_forward_split(df, strategy_config)
            if not folds: continue

            preprocessed_folds_lgbm, preprocessed_folds_lstm, preprocessed_folds_tabtransformer = [], [], []
            label_col = global_settings.get('label_column', 'label_return')
            features_for_model = [c for c in df.columns if c != label_col and not c.startswith('future_')]

            for i, (train_df, val_df) in enumerate(folds):
                y_train, y_val = train_df[label_col], val_df[label_col]
                X_train_raw, X_val_raw = train_df[features_for_model], val_df[features_for_model]
                
                train_mean, train_std = X_train_raw.mean(), X_train_raw.std() + 1e-8
                X_train_scaled = (X_train_raw - train_mean) / train_std
                X_val_scaled = (X_val_raw - train_mean) / train_std
                
                preprocessed_folds_lgbm.append({'X_train_scaled': X_train_scaled, 'y_train': y_train, 'X_val_scaled': X_val_scaled, 'y_val': y_val, 'feature_cols': features_for_model})

                use_tabtransformer = stock_info.get('use_tabtransformer', global_settings.get('use_tabtransformer_globally', True))
                if 'tabtransformer' in global_settings.get('models_to_train', []) and use_tabtransformer and encode_categorical_features:
                    try:
                        cat_features = default_model_params.get('tabtransformer_params', {}).get('categorical_features', [])
                        cont_features = [c for c in features_for_model if c not in cat_features]
                        train_encoded, val_encoded, _ = encode_categorical_features(X_train_scaled.copy(), X_val_scaled.copy(), cat_features)
                        cat_dims = [int(train_encoded[c].max()) + 1 for c in cat_features]
                        preprocessed_folds_tabtransformer.append({
                            'X_train_cont': torch.from_numpy(train_encoded[cont_features].values.copy()).float(), 'X_train_cat': torch.from_numpy(train_encoded[cat_features].values.copy()).long(),
                            'y_train_tensor': torch.from_numpy(y_train.values.copy()).float().unsqueeze(1), 'X_val_cont': torch.from_numpy(val_encoded[cont_features].values.copy()).float(),
                            'X_val_cat': torch.from_numpy(val_encoded[cat_features].values.copy()).long(), 'y_val_tensor': torch.from_numpy(y_val.values.copy()).float().unsqueeze(1),
                            'y_val': y_val, 'cat_dims': cat_dims, 'feature_cols': features_for_model
                        })
                    except Exception as e:
                        print(f"\nERROR (L3 Cache Gen): 为 {keyword} ({ticker}) 的 Fold {i+1} 生成 TabTransformer 数据时出错: {e}")
                
                use_lstm = stock_info.get('use_lstm', global_settings.get('use_lstm_globally', True))
                if 'lstm' in global_settings.get('models_to_train', []) and use_lstm:
                    try:
                        lstm_seq_len = default_model_params.get('lstm_params', {}).get('sequence_length', 60)
                        if len(X_train_scaled) < lstm_seq_len: continue
                        train_df_scaled_for_lstm = X_train_scaled.copy(); train_df_scaled_for_lstm[label_col] = y_train
                        val_df_scaled_for_lstm = X_val_scaled.copy(); val_df_scaled_for_lstm[label_col] = y_val
                        train_history_for_val = train_df_scaled_for_lstm.iloc[-lstm_seq_len:]
                        combined_df_for_lstm_val = pd.concat([train_history_for_val, val_df_scaled_for_lstm])
                        
                        lstm_builder_for_seq = LSTMBuilder(config)
                        X_train_seq, y_train_seq, _ = lstm_builder_for_seq._create_sequences(train_df_scaled_for_lstm.reset_index(), features_for_model)
                        X_val_seq, y_val_seq, dates_val_seq = lstm_builder_for_seq._create_sequences(combined_df_for_lstm_val.reset_index(), features_for_model)

                        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0: continue
                        preprocessed_folds_lstm.append({
                            'X_train_tensor': torch.from_numpy(X_train_seq.copy()).float(), 'y_train_tensor': torch.from_numpy(y_train_seq.copy()).float().unsqueeze(1),
                            'X_val_tensor': torch.from_numpy(X_val_seq.copy()).float(), 'y_val_tensor': torch.from_numpy(y_val_seq.copy()).float().unsqueeze(1),
                            'y_val_seq': y_val_seq, 'dates_val_seq': dates_val_seq, 'feature_cols': features_for_model
                        })
                    except Exception as e:
                        print(f"\nERROR (L3 Cache Gen): 为 {keyword} ({ticker}) 的 Fold {i+1} 生成 LSTM 数据时出错: {e}")

            stock_data_cache = {'full_df': df, 'lgbm_folds': preprocessed_folds_lgbm, 'lstm_folds': preprocessed_folds_lstm, 'tabtransformer_folds': preprocessed_folds_tabtransformer}
            try:
                joblib.dump(stock_data_cache, stock_l3_cache_path)
            except Exception as e:
                tqdm.write(f"ERROR: 为 {keyword} ({ticker}) 保存 L3 缓存时失败: {e}")

    # --- 重新加载所有分块缓存 ---
    print("\nINFO: 正在从分块文件重新加载所有 L3 缓存到内存...")
    global_data_cache = {}
    for stock_info in stocks_to_process:
        ticker = stock_info.get('ticker')
        if not ticker: continue
        stock_l3_cache_path = L3_CACHE_DIR / f"{ticker}.joblib"
        if stock_l3_cache_path.exists():
            try:
                global_data_cache[ticker] = joblib.load(stock_l3_cache_path)
            except Exception as e:
                print(f"WARNNING: 加载 {ticker} 的 L3 缓存文件时失败: {e}")

    if not global_data_cache:
        print("WARNNING: 未能成功加载任何 L3 缓存数据。")

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
        print(f"开始为模型 [{model_type_for_hpo.upper()}] 进行 HPO")
        
        hpo_results_list = []
        model_hpo_config = hpo_config.get(f'{model_type_for_hpo}_hpo_config', {})
        num_eval_folds = model_hpo_config.get('hpo_num_eval_folds', hpo_config.get('hpo_num_eval_folds', 2))

        for ticker in hpo_tickers:
            stock_info = next((s for s in stocks_to_process if s['ticker'] == ticker), None)
            if not stock_info:
                print(f"WARNNING: 未在 'stocks_to_process' 中找到 HPO 股票 {ticker} 的配置。跳过。")
                continue
            
            keyword = stock_info.get('keyword', ticker)

            use_model = True
            if model_type_for_hpo != 'lgbm':
                 is_enabled_flag = f"use_{model_type_for_hpo}_globally"
                 is_enabled_per_stock_flag = f"use_{model_type_for_hpo}"
                 use_model = stock_info.get(is_enabled_per_stock_flag, global_settings.get(is_enabled_flag, True))

            if not use_model:
                print(f"\nINFO: {keyword} ({ticker}) 已配置为不使用 {model_type_for_hpo.upper()}，跳过 HPO。")
                continue

            if ticker not in global_data_cache:
                print(f"ERROR: 预处理数据缓存中未找到 {keyword} ({ticker}) 的数据。跳过。")
                continue

            folds_key = f"{model_type_for_hpo}_folds"
            if model_type_for_hpo == 'tabtransformer' and (folds_key not in global_data_cache[ticker] or not global_data_cache[ticker][folds_key]):
                folds_key = 'lgbm_folds'
            
            all_preprocessed_folds = global_data_cache[ticker].get(folds_key, [])
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
                preprocessed_folds=hpo_folds_data, ticker=ticker,
                config=hpo_run_config, model_type=model_type_for_hpo
            )
            
            if best_params and best_value is not None:
                hpo_results_list.append({'ticker': ticker, 'keyword': keyword, 'best_score': best_value, **best_params})
        
        if hpo_results_list:
            hpo_log_dir_name = global_settings.get('hpo_log_dir', 'hpo_logs')
            hpo_log_dir = Path(hpo_log_dir_name)
            hpo_log_dir.mkdir(exist_ok=True)
            
            hpo_best_results_path = hpo_log_dir / f"hpo_best_results_{model_type_for_hpo}.csv"
            current_hpo_df = pd.DataFrame(hpo_results_list).set_index('ticker')

            if hpo_best_results_path.exists():
                historical_best_df = pd.read_csv(hpo_best_results_path).set_index('ticker')
                for ticker, current_row in current_hpo_df.iterrows():
                    if ticker not in historical_best_df.index or current_row['best_score'] > historical_best_df.loc[ticker, 'best_score']:
                        historical_best_df.loc[ticker] = current_row
                final_best_df = historical_best_df
            else:
                final_best_df = current_hpo_df
            final_best_df.to_csv(hpo_best_results_path)
            print(f"\nSUCCESS: 最新的 [{model_type_for_hpo.upper()}] HPO 冠军榜已保存至 {hpo_best_results_path}")
            
            param_cols_original = [c for c in hpo_results_list[0].keys() if c not in ['ticker', 'keyword', 'best_score']]
            final_hpo_params = final_best_df[param_cols_original].median().to_dict()
            
            # --- (核心修复) ---
            model_search_space = hpo_config.get(f'{model_type_for_hpo}_hpo_config', {}).get('search_space', {})
            
            # 找出所有被定义为 'int' 或 'categorical' 的参数，因为 categorical 的结果也应该是整数
            params_to_round = [
                param for param, definition in model_search_space.items() 
                if isinstance(definition, list) and len(definition) > 0 and definition[0] in ('int', 'categorical')
            ]
            
            print(f"INFO: 模型 [{model_type_for_hpo.upper()}] 需要取整的超参数为: {params_to_round}")

            # 对所有找到的参数进行取整
            for p in params_to_round:
                if p in final_hpo_params:
                    final_hpo_params[p] = int(round(final_hpo_params[p]))
            
            param_key_for_yaml = f"{model_type_for_hpo}_params:"
            yaml_string = yaml.dump(final_hpo_params, indent=4, allow_unicode=True, default_flow_style=False)
            
            output_text_block = (
                f"\n# --- HPO 建议参数 ({model_type_for_hpo.upper()}) ---\n"
                f"# (基于在 {list(final_best_df.index)} 上的优化结果，平均最佳 ICIR: {final_best_df['best_score'].mean():.4f})\n"
                f"# (请将以下内容复制到 config.yaml 的 default_model_params 部分)\n"
                f"{param_key_for_yaml}\n"
            )
            for line in yaml_string.splitlines():
                output_text_block += f"  {line}\n"
                
            hpo_params_file_path = hpo_log_dir / f"suggested_params_{model_type_for_hpo}.txt"
            with open(hpo_params_file_path, 'w', encoding='utf-8') as f:
                f.write(output_text_block)
                
            print(f"SUCCESS: HPO 建议参数已保存至: {hpo_params_file_path}")
            print("--- HPO 参数建议 ---")
            print(output_text_block)
            
            param_key_for_config = f"{model_type_for_hpo}_params"
            config['default_model_params'][param_key_for_config].update(final_hpo_params)
            print(f"INFO: 当前运行的 config 已动态更新为 HPO 建议的平均参数。")

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

def run_single_stock_prediction(config: dict, modules: dict, target_ticker: str = None, use_specific_models: list = None):
    """
    为单只股票执行完整的“加载-预测-决策-可视化”流程。
    """
    print("=== 主工作流：启动单点预测 ===")

    if target_ticker is None:
        target_ticker = config.get('application_settings', {}).get('prophet_target_ticker')
    if not target_ticker:
        print("ERROR: 未指定要预测的目标股票。"); return

    stock_info = next((s for s in config.get('stocks_to_process', []) if s['ticker'] == target_ticker), {})
    keyword = stock_info.get('keyword', target_ticker)
    
    try:
        # 1. 加载所有构件
        artifacts = _prophet_load_artifacts(config, modules, target_ticker, use_specific_models)
        
        # 2. 获取最新特征数据
        full_feature_df = _prophet_get_latest_features(config, modules, target_ticker, keyword)
        
        # 3. 执行预测、决策和可视化
        _prophet_generate_decision(config, modules, artifacts, full_feature_df, target_ticker, keyword)
        
        print(f"\n### 单点预测工作流已成功为 {keyword} ({target_ticker}) 执行完毕！ ###")
    
    except Exception as e:
        print(f"\nFATAL: 单点预测工作流失败: {e}")
        import traceback
        traceback.print_exc()

# --- 主执行函数 ---

# --- 1. 训练工作流 ---
def run_complete_training_workflow(
    config: dict, 
    modules: dict, 
    use_today_as_end_date=False,
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
        run_all_data_pipeline(config, modules, use_today_as_end_date=use_today_as_end_date)
        
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
def run_periodic_retraining_workflow(config: dict, modules: dict, full_retrain: False):
    """
    (主函数3) 执行周期性的、自动化的模型再训练工作流。
    """
    workflow_type = "全局再训练 (Full Retrain)" if full_retrain else "增量更新 (Incremental Update)"

    print("=== 主工作流：启动周期性再训练 ===")

        # --- (核心新增) 1. 智能前置检查 ---
    print("\n--- 步骤0：检查是否需要更新 ---")
    
    pd, Path, os, json = modules['pd'], modules['Path'], __import__('os'), modules['json']
    
    # a. 选择一个代表性的股票来进行检查 (通常是列表中的第一个)
    stocks_to_process = config.get('stocks_to_process', [])
    if not stocks_to_process:
        print("WARNNING: 股票池为空，跳过更新。"); return
    
    check_stock_info = stocks_to_process[0]
    check_ticker = check_stock_info['ticker']
    
    # b. 获取模型的最后训练时间
    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / check_ticker
    fuser_meta_path = model_dir / "fuser_meta.json"
    
    if not fuser_meta_path.exists():
        print(f"INFO: 未找到代表性股票 {check_ticker} 的融合模型元数据。将继续执行训练流程。")
    else:
        with open(fuser_meta_path, 'r') as f:
            fuser_meta = json.load(f)
        last_train_time = pd.to_datetime(fuser_meta.get('trained_at'))

        # c. 获取数据的最后更新时间 (从 L1 缓存)
        # 我们需要找到最新的 L1 缓存文件
        cache_dir = Path(config.get('global_settings', {}).get("data_cache_dir", "data_cache"))
        raw_ohlcv_dir = cache_dir / 'raw_ohlcv'
        
        # 找到该股票所有 L1 缓存文件中的最新一个
        raw_files = list(raw_ohlcv_dir.glob(f"raw_{check_ticker.replace('.', '_')}*.pkl"))
        if not raw_files:
             print(f"INFO: 未找到代表性股票 {check_ticker} 的 L1 缓存数据。将继续执行训练流程。")
        else:
            latest_raw_file = max(raw_files, key=os.path.getctime)
            last_data_update_time = pd.to_datetime(os.path.getctime(latest_raw_file), unit='s')
            
            print(f"  - 模型的最后训练时间: {last_train_time}")
            print(f"  - 数据的最后更新时间: {last_data_update_time}")
            
            # d. 对比时间
            # 我们增加一个小的 buffer (例如1分钟)，以避免因文件系统时间精度问题导致的误判
            if last_data_update_time <= (last_train_time + pd.Timedelta(minutes=1)) and not full_retrain:
                print("\nSUCCESS: 数据自上次成功训练以来未发生变化。无需执行更新。")
                print("### 周期性更新工作流已跳过。 ###")
                return # <-- 关键：提前退出

    # 1. 动态更新配置
    print("\n--- 步骤1：动态更新配置 ---")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    config['strategy_config']['end_date'] = today_str
    print(f"SUCCESS: 配置中的 'end_date' 已动态更新为: {today_str}")

    # 2. 调用核心训练工作流
    print(f"\n--- 步骤2：启动 [{workflow_type}] 流水线 ---")

    if full_retrain:
        # 模式一：全局重训练 (破坏性)
        print("INFO: 将执行完整的、破坏性的全局再训练。")
        run_complete_training_workflow(
            config=config, 
            modules=modules,
            use_today_as_end_date=True, # 使用今日日期
            force_reprocess_l3=True,    # 强制重建 L3 缓存以包含最新数据
            run_hpo=False,              # 跳过 HPO
            force_retrain_base=True,    # 重训基础模型
            force_retrain_fuser=True,   # 重训融合模型
            run_evaluation=True,        # 重训后需要记录完整性能
            run_visualization=True     # 生成图表
        ) 
    else:
        # 模式二：增量更新 (非破坏性，默认)
        print("INFO: 将执行非破坏性的增量更新。")
        run_complete_training_workflow(
            config=config, 
            modules=modules,
            use_today_as_end_date=True, # 使用今日日期
            force_reprocess_l3=True,    # 强制重建 L3 缓存以包含最新数据
            run_hpo=False,              # 跳过 HPO
            force_retrain_base=False,   # 使用追加模式
            force_retrain_fuser=True,   # 重训融合模型，以学习新的 OOF
            run_evaluation=True,        # 评估
            run_visualization=False     # 生成图表
        )

# --- 命令行执行入口 ---

if __name__ == '__main__':
    import argparse
    import sys

    # --- 1. 设置功能丰富、帮助信息清晰的命令行参数解析器 ---
    parser = argparse.ArgumentParser(
        description="【量化模型核心引擎】一个集成了训练、预测和自动化更新功能的多功能工具。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息中的换行格式
    )
    
    # 定义主命令 (工作流)
    parser.add_argument(
        'workflow', 
        choices=['train', 'predict', 'batch_predict', 'update', 'full_retrain'], 
        help=(
            "要执行的核心工作流:\n"
            "  train          - (研究) 执行一次完整的、可配置的训练与评估流程。\n"
            "                 默认使用缓存，不强制重训，用于日常研究和快速迭代。\n\n"
            "  predict        - (应用) 为单个目标股票生成即时交易决策。\n"
            "                 自动加载最新模型和最新数据。\n\n"
            "  batch_predict  - (应用) 为配置文件中的所有股票批量生成交易决策。\n\n"
            "  update         - (运维) 执行一次【增量更新】。只训练新数据，并重新生成最终模型。\n"
            "                 适合高频（如每日、每周）的自动化模型迭代。\n\n"
            "  full_retrain   - (运维) 执行一次【全局重训】。删除所有旧模型并从头开始。\n"
            "                 适合低频（如每月）或在代码有重大变更后执行。"
        )
    )
    
    # 定义可选参数
    parser.add_argument('--ticker', type=str, help="（仅用于 'predict'）要预测的目标股票代码，例如 '600519.SH'。")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="指定配置文件的路径 (默认为 'configs/config.yaml')。")
    parser.add_argument('--models', nargs='+', help="（仅用于 'predict'）指定只使用哪些模型进行预测，例如 --models lgbm lstm。")
    parser.add_argument('--no-hpo', action='store_true', help="（仅用于 'train'）执行训练，但跳过超参数优化步骤。")
    parser.add_argument('--no-viz', action='store_true', help="（仅用于 'train'）执行训练，但不生成可视化图表。")
    parser.add_argument('--no-fusion', action='store_true', help="（仅用于 'train'）执行基础模型训练，但不训练融合模型。")
    
    args = parser.parse_args()

    # --- 2. 在所有操作之前，首先加载环境 ---
    print("--- 启动量化模型核心引擎 ---")
    config, modules = run_load_config_and_modules(config_path=args.config)
    
    if not (config and modules):
        print("FATAL: 环境初始化失败，无法执行任何工作流。请检查配置文件路径和模块导入。")
        sys.exit(1)

    # --- 3. 根据命令行参数分发并执行工作流 ---
    try:
        if args.workflow == 'train':
            # 手动执行的完整训练，通常会运行 HPO 并使用已有的缓存来加速
            run_complete_training_workflow(
                config=config, modules=modules,
                use_today_as_end_date=args.latest if 'latest' in args else False,
                run_hpo=not args.no_hpo,
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
                print("ERROR: 未在命令行或配置文件中指定要预测的股票。")
                print("用法示例: python main_train.py predict --ticker 600519.SH")

        elif args.workflow == 'batch_predict':
            run_batch_prediction_workflow(config, modules)

        elif args.workflow == 'update':
            # 调用增量更新，full_retrain=False
            run_periodic_retraining_workflow(config, modules, full_retrain=False)
        
        elif args.workflow == 'full_retrain':
            # 调用全局重训，full_retrain=True
            run_periodic_retraining_workflow(config, modules, full_retrain=True)

    except Exception as e:
        print(f"\nFATAL: 在执行工作流 '{args.workflow}' 期间发生顶级异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- 引擎工作流执行完毕。 ---")