# 文件路径: data_process/save_data.py
'''
保存数据
'''
import json
import yaml
import sys
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import hashlib

try:
    from data_process.get_data import process_all_from_config
except ImportError:
    print("错误: 无法找到 'get_data.py'。请确保它与本脚本在同一目录中或路径正确。")
    sys.exit(1)

def get_config_hash_for_ticker(global_settings: Dict, strategy_config: Dict, stock_specific_config: Dict) -> str:
    """
    为单只股票计算其完整配置的哈希值。
    现在 stock_specific_config 已经反映了运行时的真实情况。
    """
    # 合并所有配置，stock_specific_config 会覆盖其他同名键
    relevant_config = {**global_settings, **strategy_config, **stock_specific_config}
    
    # 移除不影响数据生成的元数据
    items_to_pop = [
        'keyword', 'ticker', 'tushare_api_token', 'model_dir', 
        'ic_history_file', 'order_history_db', 'num_model_versions_to_keep',
        'models_to_train' # 这些不影响单个数据集的生成
    ]
    for item in items_to_pop:
        relevant_config.pop(item, None)
    
    config_string = json.dumps(relevant_config, sort_keys=True)
    return hashlib.sha256(config_string.encode('utf-8')).hexdigest()[:12]

def get_processed_data_path(stock_info: dict, config: dict) -> Path:
    """
    根据配置，为指定股票生成其处理后数据文件的确定性路径。
    这是定位数据文件的唯一官方途径。
    """
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    
    config_hash = get_config_hash_for_ticker(global_settings, strategy_config, stock_info)
    
    output_dir_base = global_settings.get('output_dir', 'data/processed')
    start_date = strategy_config.get('start_date')
    end_date = strategy_config.get('end_date')
    date_range_str = f"{start_date}_to_{end_date}"
    ticker = stock_info['ticker']

    target_dir = Path(output_dir_base) / ticker / date_range_str
    return target_dir / f"features_{config_hash}.pkl"

def save_processed_data(processed_data: Dict[str, pd.DataFrame], runtime_configs: Dict[str, Dict], config: Dict):
    """
    将处理好的数据字典以确定性的方式保存到磁盘。
    """
    if not processed_data:
        print("没有需要保存的数据。")
        return
        
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    
    print("\n" + "="*60)
    print("开始以确定性方式保存处理好的数据...")
    
    for ticker, df in processed_data.items():
        try:
            # --- 核心修正 1：使用运行时配置来计算哈希和路径 ---
            runtime_stock_info = runtime_configs.get(ticker)
            if not runtime_stock_info:
                continue

            # --- 核心修正 3：使用新的统一路径函数来获取保存路径 ---
            target_file_path = get_processed_data_path(runtime_stock_info, config)
            target_dir = target_file_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            if target_file_path.exists():
                print(f"  - INFO: 数据文件已存在于 {target_file_path}，跳过保存 {ticker}。")
                continue
            
            print(f"\n  - 正在保存 {ticker} 的数据...")
            df.to_pickle(target_file_path)
            print(f"    - SUCCESS: 数据已成功保存至: {target_file_path}")

        except Exception as e:
            print(f"    - ERROR: 保存 {ticker} 数据时发生错误: {e}")
            
    print("\n" + "="*60)
    print("所有数据保存完毕。")
    print("="*60)

def run_data_pipeline(config_path: str):
    """主执行函数：调用数据处理管道，然后将结果保存到指定目录。"""
    print("="*60); print("开始执行数据管道协调任务..."); print(f"将使用配置文件: {config_path}"); print("="*60)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"错误: 无法加载或解析配置文件 '{config_path}': {e}")
        return
    
    processed_data, runtime_configs = process_all_from_config(config_path)

    if processed_data:
        save_processed_data(processed_data, runtime_configs, config)
    else:
        print("未能从数据管道获取任何数据，跳过保存步骤。")
    
    print("\n数据管道协调任务执行完毕。")

if __name__ == '__main__':
    # 建议使用 YAML 配置文件以获得更好的可读性
    DEFAULT_CONFIG_PATH = 'configs/system_config.yaml'
    run_data_pipeline(config_path=DEFAULT_CONFIG_PATH)