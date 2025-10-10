# --- save_data.py ---

import json
import sys
from typing import Dict
from pathlib import Path
import datetime
import pandas as pd
import hashlib

try:
    from get_data import process_all_from_config
except ImportError:
    print("错误: 无法找到 'get_data.py'。请确保它与本脚本在同一目录中。")
    sys.exit(1)

def get_config_hash(config: Dict) -> str:
    """计算配置字典的SHA256哈希值，以唯一标识数据生成逻辑。"""
    # 我们只关心影响数据内容的 global_settings
    relevant_config = config.get('global_settings', {})
    
    # 转换为排序后的JSON字符串以确保哈希的确定性
    config_string = json.dumps(relevant_config, sort_keys=True)
    return hashlib.sha256(config_string.encode('utf-8')).hexdigest()[:10] # 取前10位即可

def save_processed_data(processed_data: Dict[str, pd.DataFrame], config: Dict):
    """
    将处理好的数据字典以确定性的方式保存到磁盘。
    """
    if not processed_data:
        print("没有需要保存的数据。")
        return
        
    global_settings = config.get('global_settings', {})
    output_dir_base = global_settings.get('output_dir', 'data/processed')
    start_date = global_settings.get('start_date')
    end_date = global_settings.get('end_date')
    config_hash = get_config_hash(config)
    
    print("\n" + "="*60)
    print("开始以确定性方式保存处理好的数据...")
    
    for ticker, df in processed_data.items():
        try:
            # 1. 构建确定性的目录和文件路径
            date_range_str = f"{start_date}_to_{end_date}"
            # 目录结构: base_dir / ticker / date_range /
            target_dir = Path(output_dir_base) / ticker / date_range_str
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = f"features_{config_hash}.pkl"
            target_file_path = target_dir / file_name

            # 2. 检查文件是否已存在 (高效断点续存)
            if target_file_path.exists():
                print(f"  - INFO: 数据文件已存在于 {target_file_path}，跳过保存 {ticker}。")
                continue
            
            # 3. 如果不存在，则保存
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
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析配置文件 '{config_path}': {e}")
        return
    
    # 1. 调用 get_data.py 的主函数获取数据
    processed_data = process_all_from_config(config_path)

    # 2. 将返回的数据字典以确定性的方式保存到磁盘
    if processed_data:
        save_processed_data(processed_data, config)
    else:
        print("未能从数据管道获取任何数据，跳过保存步骤。")
    
    print("\n数据管道协调任务执行完毕。")

if __name__ == '__main__':
    DEFAULT_CONFIG_PATH = 'config.json'
    run_data_pipeline(config_path=DEFAULT_CONFIG_PATH)