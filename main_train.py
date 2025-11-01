# 文件路径: main_train.py

import sys
import yaml
import json
import torch
import hydra
import joblib
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

torch.set_float32_matmul_precision('high')

# --- 内部辅助函数 ---
def _prophet_load_artifacts(config: dict, modules: dict, target_ticker: str, use_specific_models: list = None) -> dict:
    """
    为单点预测加载所有必需的、最新版本的构件。
    通过文件名对模型版本进行排序，并在加载后进行一致性检查。
    """
    print("\n--- 步骤1：加载所有已训练构件 (自动查找最新版本) ---")
    
    Path, joblib, torch, json = modules['Path'], modules['joblib'], modules['torch'], modules['json']
    ModelFuser, RiskManager, LSTMModel, TabTransformerModel = modules['ModelFuser'], modules['RiskManager'], modules['LSTMModel'], modules.get('TabTransformerModel')
    find_latest_artifact_paths = modules['find_latest_artifact_paths']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    artifacts = {'models': {}, 'scalers': {}, 'encoders': {}, 'feature_cols': None}
    model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / target_ticker
    
    models_to_load = use_specific_models or config.get('global_settings', {}).get('models_to_train', [])
    if not models_to_load: 
        raise ValueError("没有指定要加载的模型。")

    loaded_versions = {}

    # --- 1. 分别加载每种模型的构件 ---
    if 'lgbm' in models_to_load:
        print(f"  - 正在加载 LGBM 的构件...")
        try:
            paths = find_latest_artifact_paths(model_dir, 'lgbm')
            artifacts['models']['lgbm'] = joblib.load(paths['model'])
            artifacts['scalers']['lgbm'] = joblib.load(paths['scaler'])
            loaded_versions['lgbm'] = paths['timestamp']
            print(f"    - SUCCESS: LGBM 版本 '{paths['timestamp']}' 已加载。")
        except Exception as e:
            print(f"    - ERROR: 加载 LGBM 构件失败: {e}")

    if 'lstm' in models_to_load:
        print(f"  - 正在加载 LSTM 的构件...")
        try:
            paths = find_latest_artifact_paths(model_dir, 'lstm')
            with open(paths['meta'], 'r') as f: metadata = json.load(f)
            
            model_structure = metadata.get('model_structure')
            input_size = metadata.get('input_size')
            
            if not model_structure or input_size is None: 
                raise ValueError("元数据缺少 'model_structure' 或 'input_size'。")
            
            lstm_instance = LSTMModel(input_size=input_size, **model_structure)
            
            lstm_instance.load_state_dict(torch.load(paths['model'], map_location=device))
            lstm_instance.to(device).eval()
            
            artifacts['models']['lstm'] = lstm_instance
            artifacts['scalers']['lstm'] = joblib.load(paths['scaler'])
            loaded_versions['lstm'] = paths['timestamp']
            print(f"    - SUCCESS: LSTM 版本 '{paths['timestamp']}' 已加载。")
        except Exception as e:
            print(f"    - ERROR: 加载 LSTM 构件失败: {e}")

    if 'tabtransformer' in models_to_load and TabTransformerModel:
        print(f"  - 正在加载 TABTRANSFORMER 的构件...")
        try:
            paths = find_latest_artifact_paths(model_dir, 'tabtransformer')
            with open(paths['meta'], 'r') as f: metadata = json.load(f)

            model_structure = metadata.get('model_structure')
            cat_dims = metadata.get('cat_dims')

            if not model_structure or not cat_dims:
                raise ValueError("元数据缺少 'model_structure' 或 'cat_dims'。")

            tt_instance = TabTransformerModel(
                cat_dims=cat_dims,
                **model_structure
            )
            
            tt_instance.load_state_dict(torch.load(paths['model'], map_location=device))
            tt_instance.to(device).eval()

            artifacts['models']['tabtransformer'] = tt_instance
            artifacts['scalers']['tabtransformer'] = joblib.load(paths['scaler'])
            if paths['encoders'].exists(): artifacts['encoders']['tabtransformer'] = joblib.load(paths['encoders'])
            loaded_versions['tabtransformer'] = paths['timestamp']
            print(f"    - SUCCESS: TabTransformer 版本 '{paths['timestamp']}' 已加载。")
        except Exception as e:
            print(f"    - ERROR: 加载 TABTRANSFORMER 构件失败: {e}")

    if not artifacts.get('models'):
        raise RuntimeError("未能成功加载任何基础模型，预测流程无法继续。")

    # --- 2. 版本一致性检查 ---
    if len(set(loaded_versions.values())) > 1:
        print("!!! FATAL WARNING: 检测到加载了不同版本的模型构件 !!!")
        for model_type, version in loaded_versions.items():
            print(f"  - {model_type.upper()}: 版本 {version}")
        print("  - 这将导致预测结果不一致和不可靠。请检查您的模型文件。")
        print("  - 建议清理模型目录，或只保留一套版本匹配的模型文件。")
        raise RuntimeError("加载了不一致的模型版本，流程中止以保证安全。")

    # --- 2. 智能加载训练时的特征列表 ---
    print("  - INFO: 正在加载训练时使用的特征列表...")
    feature_cols_loaded = False
    load_priority = ['lgbm'] + [m for m in models_to_load if m != 'lgbm']
    for model_type_to_try in load_priority:
        if model_type_to_try in artifacts['models']:
            try:
                paths = find_latest_artifact_paths(model_dir, model_type_to_try)
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
        latest_fuser_paths = find_latest_artifact_paths(model_dir, 'fuser')
        fuser_version = latest_fuser_paths.get('timestamp')
        artifacts['fuser'] = ModelFuser(target_ticker, config, version=fuser_version)

        if not artifacts['fuser'].load(): 
            print(f"  - WARNNING: 未能加载 ModelFuser (版本 {fuser_version})，将回退到简单平均。")

        db_path = config.get('global_settings', {}).get('order_history_db', 'order_history.db')
        artifacts['risk_manager'] = RiskManager(db_path=db_path)
    except FileNotFoundError:
        print("  - WARNNING: 未找到任何版本的 ModelFuser 文件。将回退到简单平均。")
        artifacts['fuser'] = ModelFuser(target_ticker, config)
        artifacts['fuser'].use_fallback = True
    except Exception as e:
        print(f"    - ERROR: 加载 Fuser 或 RiskManager 失败: {e}")

    print("--- 步骤1成功完成：所有构件已加载。 ---")
    return artifacts

def _prophet_get_latest_features(config: dict, modules: dict, target_ticker: str, keyword: str) -> pd.DataFrame:
    """
    为单点预测动态计算并获取最新的特征 DataFrame。
    """
    print("\n--- 步骤2：准备预测所需的数据 ---")
    
    # 1. 从 modules 字典中获取所有需要的函数
    initialize_apis, shutdown_apis = modules['initialize_apis'], modules['shutdown_apis']
    get_full_feature_df = modules['get_full_feature_df']
    get_latest_global_data = modules['get_latest_global_data']

    full_feature_df = None
    try:
        initialize_apis(config)
        
        # 2. (核心修改) 用一行调用替换所有全局数据获取逻辑
        global_data = get_latest_global_data(config)

        # 3. 为目标股票获取特征
        max_lookback_days_stock = 365 * 2 # 可配置
        end_date_dt_stock = pd.Timestamp.now()
        start_date_dt_stock = end_date_dt_stock - pd.DateOffset(days=max_lookback_days_stock)
        start_date_str_stock, end_date_str_stock = start_date_dt_stock.strftime('%Y-%m-%d'), end_date_dt_stock.strftime('%Y-%m-%d')

        # 将获取到的全局数据通过 kwargs 传递给 get_full_feature_df
        full_feature_df = get_full_feature_df(
            ticker=target_ticker, 
            config=config, 
            start_date_str=start_date_str_stock, 
            end_date_str=end_date_str_stock,
            keyword=keyword, 
            prediction_mode=True,
            **global_data # 使用字典解包传递所有全局 DataFrame
        )
    finally:
        shutdown_apis()
    
    if full_feature_df is None or full_feature_df.empty:
        raise RuntimeError("获取最新特征数据失败。")
        
    print("--- 步骤2成功完成：最新特征数据已生成。 ---")
    return full_feature_df

def _prophet_generate_decision(config: dict, modules: dict, artifacts: dict, full_feature_df: pd.DataFrame, target_ticker: str, keyword: str):
    """
    执行预测、融合、风控、生成报告和可视化。
    确保所有模型被预测，融合模型被使用，并且可视化图表完整、正确、美观。
    """
    # --- 1. 提取模块和配置 ---
    pd, torch, np = modules['pd'], modules['torch'], __import__('numpy')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- 2. 特征对齐与数据准备 ---
    train_feature_cols = artifacts.get('feature_cols')
    if not train_feature_cols:
        raise RuntimeError("未能从构件中加载训练时的特征列表 (feature_cols)。")
    
    missing_features = set(train_feature_cols) - set(full_feature_df.columns)
    if missing_features:
        raise RuntimeError(f"预测时生成的特征数据缺少了训练时的关键特征: {missing_features}。")
    
    feature_df_aligned = full_feature_df[train_feature_cols]
    
    lstm_params = config.get('model', {}).get('lstm_params', {})
    lstm_seq_len = lstm_params.get('sequence_length', 60)
    
    if len(feature_df_aligned) < lstm_seq_len:
        raise ValueError(f"对齐后的数据长度 ({len(feature_df_aligned)}) 不足以满足 LSTM 序列长度 ({lstm_seq_len})。")
    
    historical_sequence = feature_df_aligned.iloc[-lstm_seq_len:]
    latest_features = feature_df_aligned.iloc[-1:]
    
    # --- 3. 独立模型预测 ---
    print("\n--- 步骤3：生成独立模型预测 ---")
    predictions, lgbm_preds = {}, {}
    
    for model_type, model in artifacts['models'].items():
        try:
            if model_type == 'lgbm':
                X_scaled_df = artifacts['scalers']['lgbm'].transform(latest_features)
                for name, quantile_model in model.items():
                    pred = quantile_model.predict(X_scaled_df)[0]
                    lgbm_preds[name] = pred
                    if name == 'q_0.5':
                        predictions['lgbm'] = pred
                print(f"  - LGBM 预测值 (中位数): {predictions.get('lgbm', 'N/A'):.6f}")
            
            elif model_type == 'lstm':
                scaler = artifacts['scalers']['lstm']
                X_scaled_df = pd.DataFrame(
                    scaler.transform(historical_sequence),
                    columns=historical_sequence.columns,
                    index=historical_sequence.index
                )
                
                # 从修复后的 DataFrame 中获取 NumPy 数组
                X_tensor = torch.from_numpy(X_scaled_df.values.copy()).unsqueeze(0).float()

                with torch.no_grad():
                    pred = model(X_tensor.to(device)).item()
                predictions['lstm'] = pred
                print(f"  - LSTM 预测值 (基于真实序列): {pred:.6f}")
            
            elif model_type == 'tabtransformer':
                X_df_to_predict = latest_features.copy()
                encoders = artifacts['encoders']['tabtransformer']
                scaler = artifacts['scalers']['tabtransformer']
                
                cat_features = config.get('default_model_params',{}).get('tabtransformer_params',{}).get('categorical_features', [])
                cont_features = [c for c in train_feature_cols if c not in cat_features]
                
                for col in cat_features:
                    known_classes = set(encoders[col].classes_)
                    X_df_to_predict[col] = X_df_to_predict[col].astype(str).apply(lambda x: x if x in known_classes else '<unknown>')
                    X_df_to_predict[col] = encoders[col].transform(X_df_to_predict[col])
                
                X_df_to_predict[cont_features] = scaler.transform(X_df_to_predict[cont_features])
                
                # 4. 准备 Tensors
                X_cont_tensor = torch.from_numpy(X_df_to_predict[cont_features].values.copy()).float().to(device)
                X_cat_tensor = torch.from_numpy(X_df_to_predict[cat_features].values.copy()).long().to(device)
                
                # 5. 预测
                with torch.no_grad():
                    pred = model(X_cont_tensor, X_cat_tensor).item()
                
                predictions['tabtransformer'] = pred
                print(f"  - TabTransformer 预测值: {pred:.6f}")
        
        except Exception as e:
            print(f"  - WARNNING: 为模型 {model_type.upper()} 生成预测时失败: {e}")

    if not predictions:
        raise RuntimeError("未能生成任何有效的模型预测。")

    # --- 4. 模型融合 ---
    print("\n--- 步骤4：融合模型预测 ---")
    fuser_instance = artifacts['fuser']
    fused_prediction = None
    if fuser_instance and fuser_instance.is_trained:
        preds_dict_all = {f'pred_{mt}': p for mt, p in predictions.items()}
        try:
            required_fuser_inputs = set(fuser_instance.meta_model.feature_names_in_)
            available_preds = set(preds_dict_all.keys())
            
            if required_fuser_inputs.issubset(available_preds):
                fuser_inputs = {k: preds_dict_all[k] for k in required_fuser_inputs}
                fused_prediction = fuser_instance.predict(fuser_inputs)
                print(f"  - SUCCESS: ModelFuser 已成功融合 {list(fuser_inputs.keys())}。")
            else:
                 print(f"  - WARNNING: 缺少 ModelFuser 所需的预测: {required_fuser_inputs - available_preds}。将回退。")
        except AttributeError:
             print("  - WARNNING: ModelFuser 的 meta_model 属性异常。将回退。")
    if fused_prediction is None:
        if predictions:
            fused_prediction = np.mean(list(predictions.values()))
            print(f"  - INFO: 已回退到对所有可用预测进行简单平均。")
        else:
            fused_prediction = 0
    print(f"    - 最终预测信号 (已平滑): {fused_prediction:.6f}")
    
    # --- 5. 计算涨跌概率 ---
    
    print("\n--- 步骤5：计算涨跌概率 ---")
    prob_up, prob_down = None, None
    '''
    # 暂时没有模型...
    if lgbm_preds and all(k in lgbm_preds for k in ['q_0.05', 'q_0.95']):
        try:
            from scipy.stats import norm
            lower_lgbm, upper_lgbm = lgbm_preds['q_0.05'], lgbm_preds['q_0.95']
            
            mu = fused_prediction
            
            sigma = (upper_lgbm - lower_lgbm) / (2 * 1.645)
            if sigma <= 1e-6: # 如果宽度过窄或为负，则使用一个小的默认值
                print(f"  - WARNNING: 计算出的 Sigma ({sigma:.4f}) 无效. 将使用默认的波动率。")
                sigma = np.std(list(predictions.values())) if len(predictions) > 1 else 0.01

            if sigma > 1e-6:
                dist = norm(loc=mu, scale=sigma)
                prob_up, prob_down = 1 - dist.cdf(0), dist.cdf(0)
                print(f"  - INFO: 基于融合信号(mu={mu:.4f})和LGBM波动(sigma={sigma:.4f})计算得到上涨概率 {prob_up:.2%}, 下跌概率 {prob_down:.2%}")
        except Exception as e:
            print(f"  - WARNNING: 计算涨跌概率时失败: {e}")
        '''

    # --- 6. 风险审批与决策 ---
    print("\n--- 步骤6：风险审批与决策输出 ---")
    risk_manager = artifacts['risk_manager']
    backtest_cfg = config.get('backtest', {})
    signal_threshold = backtest_cfg.get('live_trading_signal_threshold', 0.01)
    
    direction_str = 'BUY' if fused_prediction > signal_threshold else ('SELL' if fused_prediction < -signal_threshold else 'HOLD')
    trade_price = latest_features['close'].iloc[0]
    decision_approved, order_id, decision_notes = False, None, "信号强度未达到开仓阈值。"
    if direction_str in ['BUY', 'SELL']:
        decision_approved, order_id = risk_manager.approve_trade(ticker=target_ticker, direction=direction_str, price=trade_price, latest_market_data=full_feature_df, config=config)
        decision_notes = "信号通过所有风险检查。" if decision_approved else "信号被 RiskManager 拒绝（详情见上方日志）。"
    
    final_direction = "看涨 (BUY)" if direction_str == 'BUY' else ("看跌 (SELL)" if direction_str == 'SELL' else "中性 (HOLD)")
    trade_action = "【批准开仓】" if decision_approved else ("【信号被拒】" if direction_str in ['BUY', 'SELL'] else "【无需操作】")
    
    # --- 7. 生成最终决策报告 ---
    print("\n--- 最终决策报告 ---")
    report_data = [
        ('股票名称', f"{keyword} ({target_ticker})"), ('决策生成时间', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('信号方向', final_direction), ('信号强度', f"{fused_prediction:+.4%}"),
        ('上涨概率', f"{prob_up:.2%}" if prob_up is not None else 'N/A'),
        ('下跌概率', f"{prob_down:.2%}" if prob_down is not None else 'N/A'),
        ('交易动作', f"{trade_action} {final_direction if decision_approved else ''}"), ('备注', decision_notes),
        ('关联订单ID', order_id or 'N/A'),
    ]
    report_df = pd.DataFrame(report_data, columns=['项目', '内容']).set_index('项目')
    print(report_df.to_string())
    
    # --- 8. 可视化 ---
    print("\n--- 步骤7：决策支持可视化 ---")
    try:
        import matplotlib.pyplot as plt; import seaborn as sns; from matplotlib.patches import Patch
        
        plt.style.use('seaborn-v0_8-darkgrid')
        try: 
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except: print("WARNNING: 未能设置中文字体 'SimHei'。")
        
        plot_df = full_feature_df.tail(200).copy()
        rsi_col_name = next((c for c in plot_df.columns if 'rsi' in c.lower()), None)
        for col in ['close', 'volume'] + ([rsi_col_name] if rsi_col_name else []):
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

        fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True, gridspec_kw={'height_ratios': [5, 2, 2]})
        fig.suptitle(f'{keyword} ({target_ticker}) - 交易决策支持图表', fontsize=20, fontweight='bold')
        
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
        
        # 图 1: 价格 & 预测
        ax1.set_ylabel('价格', fontsize=12)
        ax1.plot(plot_df.index, plot_df['close'], color='black', linewidth=1.5, zorder=5)
        
        last_price, last_date = plot_df['close'].iloc[-1], plot_df.index[-1]
        labeling_cfg = config.get('labeling', {})
        pred_horizon = labeling_cfg.get('labeling_horizon', 45)
        future_date = last_date + pd.DateOffset(days=pred_horizon)
        
        handles = [plt.Line2D([0], [0], color='black', lw=2, label='收盘价')]
        
        if lgbm_preds and all(k in lgbm_preds for k in ['q_0.05', 'q_0.5', 'q_0.95']):
            ax1.fill_between([last_date, future_date], [last_price, last_price * (1+lgbm_preds['q_0.05'])], [last_price, last_price * (1+lgbm_preds['q_0.95'])], color='skyblue', alpha=0.5, zorder=2)
            handles.append(Patch(facecolor='skyblue', alpha=0.5, label='90% 置信区间 (LGBM)'))
        
        model_colors = {'lgbm': 'dodgerblue', 'lstm': 'orange', 'tabtransformer': 'green'}
        for mt, p_val in predictions.items():
            ax1.plot([last_date, future_date], [last_price, last_price * (1+p_val)], color=model_colors.get(mt, 'grey'), linestyle='--', linewidth=2, zorder=6)
            handles.append(plt.Line2D([0], [0], color=model_colors.get(mt, 'grey'), linestyle='--', label=f'{mt.upper()} 中位数预测 ({p_val:+.2%})'))

        ax1.plot([last_date, future_date], [last_price, last_price * (1+fused_prediction)], color='red', linestyle='--', marker='o', markersize=8, linewidth=2.5, zorder=7)
        handles.append(plt.Line2D([0], [0], color='red', linestyle='--', marker='o', label=f'最终融合预测 ({fused_prediction:+.2%})'))
        
        ax1.legend(handles=handles, loc='best'); ax1.grid(True, linestyle='--', alpha=0.6)
        
        prob_up_str, prob_down_str = (f"{p:.2%}" if p is not None else "N/A" for p in (prob_up, prob_down))
        text_box_content = (f"最终信号: {fused_prediction:+.2%}\n方向: {final_direction}\n行动: {trade_action}\n上涨概率: {prob_up_str}\n下跌概率: {prob_down_str}")
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, text_box_content, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props, zorder=10)

        # 图 2: RSI
        if rsi_col_name and not plot_df[rsi_col_name].isnull().all():
            ax2.plot(plot_df.index, plot_df[rsi_col_name], color='purple')
            ax2.axhline(70, color='r', linestyle='--', lw=1); ax2.axhline(30, color='g', linestyle='--', lw=1)
            ax2.legend([f'{rsi_col_name.upper()}', '超买线 (70)', '超卖线 (30)'], loc='upper left')
        ax2.set_ylabel('RSI'); ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 图 3: 成交量
        if 'volume' in plot_df.columns and not plot_df['volume'].isnull().all():
            ax3.bar(plot_df.index, plot_df['volume'], color='grey', width=0.6)
        ax3.set_ylabel('成交量'); ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.set_xlabel('日期', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
    except Exception as e:
        print(f"WARNNING: 绘图时发生错误: {e}"); import traceback; traceback.print_exc()

# --- 1. 阶段一：数据流水线 ---

def run_all_data_pipeline(config: dict, modules: dict, use_today_as_end_date=False):
    """
    执行数据准备与特征工程阶段。
    这是所有训练工作流中，日期解析的唯一入口。
    """
    print("=== 阶段一：数据准备与特征工程 ===")
    
    # --- 1. 从 modules 字典中获取所需模块/函数 ---
    pd = modules['pd']
    resolve_data_pipeline_dates = modules['resolve_data_pipeline_dates']

    if 'data' not in config:
        config['data'] = {}

    # --- 2. 使用工具函数来解析日期 ---
    # 如果指定了 use_today_as_end_date，则为工具函数准备一个临时键
    if use_today_as_end_date:
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        # 注入 dynamic_end_date，让 resolve_data_pipeline_dates 优先使用它
        config['data']['dynamic_end_date'] = today_str
        print(f"INFO: 已启用动态日期模式，将使用今天的日期 '{today_str}' 作为 end_date。")
    
    try:
        resolve_data_pipeline_dates(config)
    except KeyError:
        print("ERROR: 日期解析失败，数据流水线中止。")
        raise

    # --- 3. 调用核心的数据 API 和处理流程 ---
    try:
        # 初始化 API (如 Baostock, Tushare)
        modules['initialize_apis'](config)
        
        modules['run_data_pipeline'](config) 
        
    except Exception as e:
        print(f"ERROR: 数据处理阶段发生严重错误: {e}")
        raise
    finally:
        # 无论成功或失败，都确保 API 被安全登出
        modules['shutdown_apis']()
        
    print("--- 阶段 1 成功完成。 ---")

# --- 2. 阶段二：模型流水线 ---

def run_preprocess_l3_cache(config: dict, modules: dict, force_reprocess=False) -> dict:
    """
    执行 L3 数据预处理与缓存。
    - 使用配置文件中定义的目录。
    - 采用分块保存策略，为每只股票单独保存 L3 缓存，以避免 MemoryError。
    - 在函数末尾重新加载所有缓存以供当次运行使用。
    """
    print("=== 工作流阶段 2.1：为模型预处理数据 (L3 缓存) ===")
    
    # 提取模块和配置
    Path, joblib, tqdm, pd, torch = modules['Path'], modules['joblib'], modules['tqdm'], modules['pd'], modules['torch']
    get_processed_data_path, walk_forward_split = modules['get_processed_data_path'], modules['walk_forward_split']
    encode_categorical_features = modules['encode_categorical_features']
    LSTMBuilder = modules['LSTMBuilder']

    global_settings = config.get('global_settings', {})
    features_cfg = config.get('features', {}) # walk_forward_split 使用
    model_cfg = config.get('model', {}) # 所有模型参数的来源
    data_cfg = config.get('data', {}) # stocks_to_process 的来源
    stocks_to_process = data_cfg.get('stocks_to_process', [])

    l3_cache_dir_path = global_settings.get('l3_cache_dir', 'data/l3_cache')
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
            folds = walk_forward_split(df, features_cfg)
            if not folds: continue

            preprocessed_folds_lgbm, preprocessed_folds_lstm, preprocessed_folds_tabtransformer = [], [], []
            label_col = global_settings.get('label_column', 'label_return')
            features_for_model = [c for c in df.columns if c != label_col and not c.startswith('future_')]

            for i, (train_df, val_df) in enumerate(folds):
                y_train, y_val = train_df[label_col], val_df[label_col]
                X_train_raw, X_val_raw = train_df[features_for_model], val_df[features_for_model]
                
                # a. LGBM 和 LSTM 的通用标准化
                train_mean, train_std = X_train_raw.mean(), X_train_raw.std() + 1e-8
                X_train_scaled = (X_train_raw - train_mean) / train_std
                X_val_scaled = (X_val_raw - train_mean) / train_std
                preprocessed_folds_lgbm.append({'X_train_scaled': X_train_scaled, 'y_train': y_train, 'X_val_scaled': X_val_scaled, 'y_val': y_val, 'feature_cols': features_for_model})

                # b. TabTransformer 的专属数据准备
                use_tabtransformer = stock_info.get('use_tabtransformer', global_settings.get('use_tabtransformer_globally', True))
                if 'tabtransformer' in global_settings.get('models_to_train', []) and use_tabtransformer:
                    try:
                        tabt_params = model_cfg.get('tabtransformer_params', {})
                        cat_features = tabt_params.get('categorical_features', [])
                        if not cat_features:
                            print(f"WARNNING: 为 TabTransformer 配置的 categorical_features 为空，跳过。")
                            continue

                        cont_features = [c for c in features_for_model if c not in cat_features]
                        
                        # 步骤 1: 先在原始数据上进行类别编码
                        X_train_encoded, X_val_encoded, encoders = encode_categorical_features(X_train_raw.copy(), X_val_raw.copy(), cat_features)
                        
                        # 步骤 2: 然后只对连续特征计算均值/标准差，并进行标准化
                        train_mean_cont = X_train_encoded[cont_features].mean()
                        train_std_cont = X_train_encoded[cont_features].std() + 1e-8
                        
                        X_train_encoded[cont_features] = (X_train_encoded[cont_features] - train_mean_cont) / train_std_cont
                        X_val_encoded[cont_features] = (X_val_encoded[cont_features] - train_mean_cont) / train_std_cont
                        
                        # 步骤 3: 准备 Tensors
                        cat_dims = [len(encoders[col].classes_) for col in cat_features]
                        
                        preprocessed_folds_tabtransformer.append({
                            'X_train_cont': torch.from_numpy(X_train_encoded[cont_features].values.copy()).float(),
                            'X_train_cat': torch.from_numpy(X_train_encoded[cat_features].values.copy()).long(),
                            'y_train_tensor': torch.from_numpy(y_train.values.copy()).float().unsqueeze(1),
                            'X_val_cont': torch.from_numpy(X_val_encoded[cont_features].values.copy()).float(),
                            'X_val_cat': torch.from_numpy(X_val_encoded[cat_features].values.copy()).long(),
                            'y_val_tensor': torch.from_numpy(y_val.values.copy()).float().unsqueeze(1),
                            'y_val': y_val, 
                            'cat_dims': cat_dims, 
                            'feature_cols': features_for_model
                        })
                    except Exception as e:
                        print(f"\nERROR (L3 Cache Gen): 在为 {keyword} ({ticker}) 的 Fold {i+1} 生成 TabTransformer 数据时出错: {e}")

                # --- LSTM 数据准备 ---
                use_lstm = stock_info.get('use_lstm', global_settings.get('use_lstm_globally', True))
                if 'lstm' in global_settings.get('models_to_train', []) and use_lstm:
                    try:
                        lstm_params = model_cfg.get('lstm_params', {})
                        lstm_seq_len = lstm_params.get('sequence_length', 60)
                        
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
    print("--- INFO: L3 缓存文件已在磁盘上准备就绪。 ---")
    print("--- 阶段 2.1 成功完成。 ---")

def run_hpo_train(config: dict, modules: dict):
    """
    执行超参数优化 (对应 Train.ipynb 阶段 2.2)。
    """
    print("=== 阶段 2.2：超参数优化 ===")

    # --- 1. 提取模块和配置 ---
    Path, joblib, yaml = modules['Path'], modules['joblib'], modules['yaml']
    run_hpo_for_ticker = modules['run_hpo_for_ticker']

    hpo_config = config.get('hpo', {})
    global_settings = config.get('global_settings', {})
    
    models_for_hpo = hpo_config.get('models_for_hpo', [])
    hpo_tickers = hpo_config.get('tickers_for_hpo', [])
    stocks_to_process = config.get('data', {}).get('stocks_to_process', [])
    
    # --- 2. 获取 L3 缓存目录路径 ---
    l3_cache_dir_path = global_settings.get('l3_cache_dir', 'data/l3_cache')
    L3_CACHE_DIR = Path(l3_cache_dir_path)
    
    # --- 3. 前置检查 ---
    if not models_for_hpo:
        print("INFO: 在配置文件 hpo_config.models_for_hpo 中未指定要优化的模型，跳过此步骤。")
        print("--- 阶段 2.2 已跳过。 ---")
        return

    if not hpo_tickers:
        print("INFO: 在配置文件 hpo_config.tickers_for_hpo 中未指定用于 HPO 的股票，跳过此步骤。")
        print("--- 阶段 2.2 已跳过。 ---")
        return

    if not L3_CACHE_DIR.exists():
        print(f"ERROR: L3 缓存目录 '{L3_CACHE_DIR}' 不存在。请确保阶段 2.1 已成功运行。")
        print("--- 阶段 2.2 执行失败。 ---")
        return

    print(f"--- INFO: 开始为模型 {models_for_hpo} 和股票 {hpo_tickers} 进行超参数优化 ---\n")
    
    # --- 4. 主循环 ---
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

            # 检查该模型是否对该股票启用
            use_model = True
            if model_type_for_hpo != 'lgbm':
                 is_enabled_flag = f"use_{model_type_for_hpo}_globally"
                 is_enabled_per_stock_flag = f"use_{model_type_for_hpo}"
                 use_model = stock_info.get(is_enabled_per_stock_flag, global_settings.get(is_enabled_flag, True))

            if not use_model:
                print(f"\nINFO: {keyword} ({ticker}) 已配置为不使用 {model_type_for_hpo.upper()}，跳过 HPO。")
                continue

            # --- 5. (核心修改) Just-in-Time 加载 L3 缓存 ---
            stock_l3_cache_path = L3_CACHE_DIR / f"{ticker}.joblib"
            if not stock_l3_cache_path.exists():
                print(f"ERROR: 预处理数据缓存中未找到 {keyword} ({ticker}) 的数据文件。跳过 HPO。")
                continue
            
            try:
                cached_stock_data = joblib.load(stock_l3_cache_path)
            except Exception as e:
                print(f"ERROR: 加载 {keyword} ({ticker}) 的 L3 缓存失败: {e}。跳过 HPO。")
                continue

            folds_key = f"{model_type_for_hpo}_folds"
            if model_type_for_hpo == 'tabtransformer' and (folds_key not in cached_stock_data or not cached_stock_data[folds_key]):
                folds_key = 'lgbm_folds'
            
            all_preprocessed_folds = cached_stock_data.get(folds_key, [])
            if not all_preprocessed_folds:
                print(f"WARNNING: 缓存中未找到 {keyword} ({ticker}) 的 '{model_type_for_hpo}' 预处理数据。跳过 HPO。")
                continue
            
            hpo_folds_data = all_preprocessed_folds[-num_eval_folds:]
            
            print(f"\nINFO: 已为 {keyword} ({ticker}) 加载最后 {len(hpo_folds_data)} 个 folds 用于 {model_type_for_hpo.upper()} HPO。")
            
            best_params, best_value = run_hpo_for_ticker(
                preprocessed_folds=hpo_folds_data, 
                ticker=ticker,
                config=config,
                model_type=model_type_for_hpo
            )
            
            if best_params and best_value is not None:
                hpo_results_list.append({'ticker': ticker, 'keyword': keyword, 'best_score': best_value, **best_params})
        
        # --- 6. HPO 结果聚合与保存 (这部分逻辑完全不变) ---
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
            
            model_search_space = hpo_config.get(f'{model_type_for_hpo}_hpo_config', {}).get('search_space', {})
            
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

def run_all_models_train(config: dict, modules: dict, 
                     force_retrain_base=False, 
                     force_retrain_fuser=False, 
                     run_fusion=True) -> list:
    """
    执行基础模型和融合模型的训练。
    """
    print("=== 工作流阶段 2.3：训练所有模型 ===")

    # --- 1. 提取所需模块和配置 ---
    tqdm = modules['tqdm']
    run_training_for_ticker = modules['run_training_for_ticker']
    ModelFuser = modules['ModelFuser']
    Path = modules['Path']
    joblib = modules['joblib']

    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    default_model_params = config.get('default_model_params', {})
    stocks_to_process = config.get('data', {}).get('stocks_to_process', [])
    
    # --- 2. 获取 L3 缓存目录路径 ---
    l3_cache_dir_path = global_settings.get('l3_cache_dir', 'data/l3_cache')
    L3_CACHE_DIR = Path(l3_cache_dir_path)

    # --- 3. 前置检查 ---
    all_ic_history = []
    if not (config and stocks_to_process):
        print("ERROR: 配置为空或股票池为空，无法训练模型。")
        return all_ic_history
        
    if not L3_CACHE_DIR.exists():
        print(f"ERROR: L3 缓存目录 '{L3_CACHE_DIR}' 不存在。请确保阶段 2.1 已成功运行。")
        print("--- 阶段 2.3 执行失败。 ---")
        return all_ic_history

    models_to_train = global_settings.get('models_to_train', [])
    
    # --- 4. 单一的股票主循环 ---
    stock_iterator = tqdm(stocks_to_process, desc="处理股票模型")

    for stock_info in stock_iterator:
        ticker = stock_info.get('ticker')
        if not ticker:
            continue
        
        # --- 5. Just-in-Time 加载 L3 缓存 ---
        stock_l3_cache_path = L3_CACHE_DIR / f"{ticker}.joblib"
        if not stock_l3_cache_path.exists():
            tqdm.write(f"\nERROR: 未找到 {ticker} 的 L3 缓存文件，跳过该股票的训练。")
            continue
        try:
            cached_stock_data = joblib.load(stock_l3_cache_path)
        except Exception as e:
            tqdm.write(f"\nERROR: 加载 {ticker} 的L3缓存失败: {e}。跳过该股票的训练。")
            continue
        
        keyword = stock_info.get('keyword', ticker)
        stock_iterator.set_description(f"正在处理 {keyword} ({ticker})")
        
        # 从加载的数据中获取 full_df
        full_df = cached_stock_data['full_df']
        
        # --- 6. 基础模型训练 ---
        print(f"\n--- 2.3.1 为 {keyword} ({ticker}) 训练基础模型 ---")
        base_models_succeeded_count = 0
        for model_type in models_to_train:
            try:
                # 检查该模型是否被全局或个股配置启用
                use_model = True
                if model_type != 'lgbm':
                    is_enabled_flag = f"use_{model_type}_globally"
                    is_enabled_per_stock_flag = f"use_{model_type}"
                    use_model = stock_info.get(is_enabled_per_stock_flag, global_settings.get(is_enabled_flag, True))
                if not use_model:
                    print(f"INFO: {keyword} ({ticker}) 已配置为不使用 {model_type.upper()}, 跳过训练。")
                    base_models_succeeded_count += 1
                    continue

                folds_key = f"{model_type}_folds"
                if model_type == 'tabtransformer' and (folds_key not in cached_stock_data or not cached_stock_data[folds_key]):
                    folds_key = 'lgbm_folds'
                
                preprocessed_folds = cached_stock_data.get(folds_key)
                if not preprocessed_folds:
                    print(f"WARNNING: 未找到 {keyword} ({ticker}) 的 '{model_type}' 预处理 folds。跳过训练。")
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
                
                base_models_succeeded_count += 1

            except Exception as e:
                print(f"\nERROR: 为 {keyword} ({ticker}) 训练 {model_type.upper()} 模型时发生严重错误: {e}")
        
        # --- 7. 融合模型训练 ---
        if run_fusion and base_models_succeeded_count == len(models_to_train):
            print(f"\n--- 2.3.5 为 {keyword} ({ticker}) 训练融合模型 ---")
            try:
                end_date_str = config.get('data', {}).get('end_date', 'unknown_date')
                version_date = pd.to_datetime(end_date_str).strftime('%Y%m%d')

                fuser = ModelFuser(ticker, config, version=version_date)

                if not force_retrain_fuser and fuser.meta_path.exists():
                    print(f"INFO: {keyword} ({ticker}) 的融合模型元数据 (版本 {version_date}) 已存在。跳过训练。")
                else:
                    fuser.train()
            except Exception as e:
                print(f"\nERROR: 为 {keyword} ({ticker}) 训练融合模型时发生严重错误: {e}")
        elif run_fusion:
            print(f"INFO: 由于 {keyword} ({ticker}) 的基础模型训练未完全成功或被跳过，跳过其融合模型训练。")

    print("\n--- 阶段 2.3 成功完成。 ---")
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
    run_backtrader_backtest = modules.get('run_backtrader_backtest')

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
    
    print("=== 1. 模型预测能力评估 (ICIR) ===")
    print(evaluation_summary.to_string())

    # --- 2. 向量化回测 ---
    all_vectorized_results = []
    if fused_oof_preds_list and VectorizedBacktester:
        print("=== 2. 策略理论表现评估 (向量化回测) ===")
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
    
    return evaluation_summary, vectorized_summary, final_eval_df

def run_results_visualization(config: dict, modules: dict, evaluation_summary: pd.DataFrame, backtest_summary: pd.DataFrame, final_eval_df: pd.DataFrame):
    """
    对已计算的评估结果进行可视化。
    """
    print("=== 工作流阶段 2.4b：生成评估可视化图表 ===")
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

    if evaluation_summary is not None and not evaluation_summary.empty:
        print("\n--- 生成 ICIR 对比图 ---")
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.barplot(data=evaluation_summary, x='ticker_name', y='icir', hue='model_type', palette=custom_palette, ax=ax)
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
        
        print(f"\n=== 单点预测工作流已成功为 {keyword} ({target_ticker}) 执行完毕！ ===")
    
    except Exception as e:
        print(f"\nFATAL: 单点预测工作流失败: {e}")
        import traceback
        traceback.print_exc()

# --- 主执行函数 ---

# --- Hydra 主执行函数 ---
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    主执行函数。
    负责配置处理、模块加载和工作流分发。
    """
    print("--- 启动量化模型核心引擎 ---")
    
    # --- 1. 配置处理 ---
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # --- 2. 模块动态加载 ---
    # 将项目根目录添加到系统路径，确保所有模块都能被找到
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        # a. 导入核心业务逻辑模块
        from data_process.save_data import run_data_pipeline, get_processed_data_path
        from model.build_models import run_training_for_ticker
        from utils.hpo_utils import run_hpo_for_ticker
        from risk_management.risk_manager import RiskManager
        from backtest.backtester import VectorizedBacktester
        from backtest.event_driven_backtester import run_backtrader_backtest
        
        # b. 导入所有模型构建器及其依赖
        from model.builders.base_builder import builder_registry
        from model.builders.model_fuser import ModelFuser
        from model.builders.lgbm_builder import LGBMBuilder
        from model.builders.lstm_builder import LSTMBuilder, LSTMModel
        from model.builders.tabtransformer_builder import TabTransformerBuilder, TabTransformerModel
        
        # c. 导入所有需要暴露给工作流的公共接口
        from data_process.get_data import (
            initialize_apis, 
            shutdown_apis, 
            get_full_feature_df,
            get_latest_global_data
        )

        # d. 导入所有工具函数
        from utils.date_utils import resolve_data_pipeline_dates
        from utils.encoding_utils import encode_categorical_features
        from utils.file_utils import find_latest_artifact_paths
        from utils.ml_utils import walk_forward_split
        
        print("INFO: 项目模块导入成功。")
    except ImportError as e:
        print(f"FATAL: 模块导入失败: {e}")
        # 打印更详细的 trace 以便调试
        import traceback
        traceback.print_exc()
        return

    # 构建 modules 字典，作为所有工作流函数的统一依赖入口
    modules = {
        # 核心业务逻辑
        'initialize_apis': initialize_apis, 'shutdown_apis': shutdown_apis,
        'get_full_feature_df': get_full_feature_df, 'run_data_pipeline': run_data_pipeline,
        'get_processed_data_path': get_processed_data_path, 'run_training_for_ticker': run_training_for_ticker,
        'run_hpo_for_ticker': run_hpo_for_ticker,
        
        # 公共接口
        'get_latest_global_data': get_latest_global_data,
        
        # 类与构建器
        'ModelFuser': ModelFuser,
        'LGBMBuilder': LGBMBuilder, 'LSTMBuilder': LSTMBuilder, 'LSTMModel': LSTMModel,
        'TabTransformerBuilder': TabTransformerBuilder, 'TabTransformerModel': TabTransformerModel,
        'RiskManager': RiskManager, 'VectorizedBacktester': VectorizedBacktester,
        
        # 回测与其他
        'run_backtrader_backtest': run_backtrader_backtest,
        
        # 工具函数
        'resolve_data_pipeline_dates': resolve_data_pipeline_dates,
        'encode_categorical_features': encode_categorical_features,
        'find_latest_artifact_paths': find_latest_artifact_paths,
        'walk_forward_split': walk_forward_split,
        
        # 常用库的别名
        'pd': pd, 'torch': torch, 'joblib': joblib, 'tqdm': tqdm, 
        'StandardScaler': StandardScaler, 'Path': Path, 'yaml': yaml, 'json': json,
    }

    # --- 3. 工作流分发 ---
    workflow = cfg.get("workflow", "train")
    print(f"\nINFO: 检测到工作流: '{workflow}'")

    try:
        if workflow == 'train':
            run_complete_training_workflow(
                config=config, modules=modules,
                use_today_as_end_date=cfg.get("latest", False),
                run_hpo=not cfg.get("no_hpo", False),
                force_reprocess_l3=cfg.get("force_reprocess_l3", False),
                force_retrain_base=cfg.get("force_retrain_base", False),
                force_retrain_fuser=cfg.get("force_retrain_fuser", False),
                run_fusion=not cfg.get("no_fusion", False),
                run_evaluation=cfg.get("evaluation", True),
                run_visualization=not cfg.get("no_viz", False)
            )
        
        elif workflow == 'predict':
            app_cfg = config.get('application', {}).get('application_settings', {})
            ticker_to_predict = cfg.get("ticker", app_cfg.get('prophet_target_ticker'))
            
            if ticker_to_predict:
                run_single_stock_prediction(config, modules, target_ticker=ticker_to_predict, use_specific_models=cfg.get('models'))
            else:
                print("ERROR: 未在命令行或配置文件中指定要预测的股票。用法: python main_train.py workflow=predict ticker=...")

        elif workflow == 'batch_predict':
            run_batch_prediction_workflow(config, modules)

        elif workflow == 'update':
            run_periodic_retraining_workflow(config, modules, full_retrain=False)
        
        elif workflow == 'full_retrain':
            run_periodic_retraining_workflow(config, modules, full_retrain=True)

    except Exception as e:
        print(f"\nFATAL: 在执行工作流 '{workflow}' 期间发生顶级异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- 引擎工作流执行完毕。 ---")

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
    按顺序执行完整的、非交互式的训练工作流。
    """
    print("=== 主工作流：启动完整模型训练 ===")
    try:
        run_all_data_pipeline(config, modules, use_today_as_end_date=use_today_as_end_date)
        
        run_preprocess_l3_cache(config, modules, force_reprocess=force_reprocess_l3)
        
        if run_hpo:
            run_hpo_train(config, modules)
        
        all_ic_history = run_all_models_train(
            config, modules, 
            force_retrain_base=force_retrain_base, 
            force_retrain_fuser=force_retrain_fuser,
            run_fusion=run_fusion
        )
        
        if run_evaluation and all_ic_history:
            evaluation_summary, backtest_summary, final_eval_df = run_performance_evaluation(config, modules, all_ic_history)
            
            if run_visualization and (evaluation_summary is not None or backtest_summary is not None):
                run_results_visualization(config, modules, evaluation_summary, backtest_summary, final_eval_df)
                
        print("\n=== 完整训练工作流已成功执行完毕！ ===")
    except Exception as e:
        print(f"\nFATAL: 训练工作流在执行过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

# --- 2. 批量预测 ---
def run_batch_prediction_workflow(config: dict, modules: dict):
    """
    为股票池中的所有股票执行单点预测。
    """
    print("=== 主工作流：启动批量预测 ===")

    stocks_to_predict = config.get('data', {}).get('stocks_to_process', [])
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
            print(f"\n--- ERROR: 在为 {keyword} ({ticker}) 进行预测时发生错误 ---")
            print(e)
            continue
            
    print(f"\n=== 批量预测工作流执行完毕。成功预测 {successful_predictions} / {len(stocks_to_predict)} 只股票。 ===")

# --- 3. 自动化更新工作流 ---
def run_periodic_retraining_workflow(config: dict, modules: dict, full_retrain: False):
    """
    执行周期性的、自动化的模型再训练工作流。
    """
    workflow_type = "全局再训练 (Full Retrain)" if full_retrain else "增量更新 (Incremental Update)"

    print("=== 主工作流：启动周期性再训练 ===")

        # --- 1. 智能前置检查 ---
    print("\n--- 步骤0：检查是否需要更新 ---")
    
    pd, Path, os, json = modules['pd'], modules['Path'], __import__('os'), modules['json']
    
    # a. 选择一个股票来进行检查
    stocks_to_process = config.get('data', {}).get('stocks_to_process', [])
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
            if last_data_update_time <= (last_train_time + pd.Timedelta(minutes=1)) and not full_retrain:
                print("\nSUCCESS: 数据自上次成功训练以来未发生变化。无需执行更新。")
                print("=== 周期性更新工作流已跳过。 ===")
                return

    # 1. 动态更新配置
    print("\n--- 步骤1：动态更新配置 ---")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    config['data']['end_date'] = today_str
    print(f"SUCCESS: 配置中的 'end_date' 已动态更新为: {today_str}")

    # 2. 调用核心训练工作流
    print(f"\n--- 步骤2：启动 [{workflow_type}] 流水线 ---")

    if full_retrain:
        # 模式一：全局重训练
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
            run_visualization=True      # 生成图表
        ) 
    else:
        # 模式二：增量更新
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
    main()