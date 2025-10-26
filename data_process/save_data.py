# 文件路径: data_process/save_data.py
'''
保存数据
'''
import sys
import json
import hashlib
import pandas as pd
from typing import Dict
from pathlib import Path

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
    路径的生成现在只依赖于静态配置，与动态计算的 start_date 解耦。
    """
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    
    # 1. 计算配置哈希
    config_hash = get_config_hash_for_ticker(global_settings, strategy_config, stock_info)
    
    # 2. 构建路径时不使用 start_date
    output_dir_base = global_settings.get('output_dir', 'data/processed')
    ticker = stock_info['ticker']
    
    # 新的目录结构：{output_dir}/{ticker}/{config_hash}/
    # 文件名固定为 features.pkl
    target_dir = Path(output_dir_base) / ticker / config_hash
    return target_dir / "features.pkl"

def save_processed_data(processed_data: Dict[str, pd.DataFrame], config: Dict):
    """
    将处理好的数据字典以确定性的方式保存到磁盘。
    同时保存一个 meta.json 文件来记录本次生成的动态信息。
    """
    if not processed_data:
        print("INFO: 没有需要保存的数据。")
        return
        
    stocks_config_map = {s['ticker']: s for s in config.get('stocks_to_process', [])}
    
    # 从 config 中获取本次运行的日期范围
    start_date = config['strategy_config'].get('start_date', 'N/A')
    end_date = config['strategy_config'].get('end_date', 'N/A')
    
    print("--- 正在保存处理好的数据... ---")
    
    for ticker, df in processed_data.items():
        try:
            stock_specific_config = stocks_config_map.get(ticker)
            if not stock_specific_config: continue
            
            keyword = stock_specific_config.get('keyword', ticker)

            target_file_path = get_processed_data_path(stock_specific_config, config)
            target_dir = target_file_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            if target_file_path.exists():
                print(f"  - INFO: 数据文件已存在于 {target_file_path}，跳过保存 {keyword} ({ticker})。")
                continue
            
            print(f"\n  - 正在保存 {keyword} ({ticker}) 的数据...")
            
            # 1. 保存数据文件
            df.to_pickle(target_file_path)
            
            # 2. (新增) 保存元数据文件
            meta_data = {
                'ticker': ticker,
                'keyword': keyword,
                'generated_at': pd.Timestamp.now().isoformat(),
                'data_start_date': start_date,
                'data_end_date': end_date,
                'config_hash': target_dir.name,
                'file_path': str(target_file_path)
            }
            meta_file_path = target_dir / "meta.json"
            with open(meta_file_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4)

            print(f"    - SUCCESS: 数据已成功保存至: {target_file_path}")
            print(f"    - INFO: 元信息已保存至: {meta_file_path}")

        except Exception as e:
            print(f"    - ERROR: 保存 {keyword} ({ticker}) 数据时发生错误: {e}")
            
    print("\n--- 所有数据保存完毕。 ---")

def run_data_pipeline(config: dict):
    """
    主执行函数：智能地处理数据。
    """
    print("开始执行数据管道协调任务...")
    
    strategy_config = config.get('strategy_config', {})
    
    # 动态计算 start_date
    print("INFO: 正在根据 'end_date' 和 'data_lookback_years' 动态计算 'start_date'...")
    try:
        end_date_dt = pd.to_datetime(strategy_config['end_date'])
        lookback_years = strategy_config.get('data_lookback_years', 10)
        earliest_start_date_dt = pd.to_datetime(strategy_config['earliest_start_date'])
        
        target_start_date_dt = end_date_dt - pd.DateOffset(years=lookback_years)
        start_date_dt = max(target_start_date_dt, earliest_start_date_dt)
        
        calculated_start_date = start_date_dt.strftime('%Y-%m-%d')
        
        # 将计算结果“注入”回传入的 config 字典中
        config['strategy_config']['start_date'] = calculated_start_date
        print(f"      计算得出的 start_date 为: {calculated_start_date}，已更新到本次运行的全局配置中。")

    except KeyError as e:
        print(f"错误: 动态计算 start_date 失败，因为缺少关键配置项: {e}。")
        return

    stocks_to_process = config.get('stocks_to_process', [])
    if not stocks_to_process:
        print("WARNNING: 股票池为空，无需处理。")
        return
        
    tickers_to_generate = []
    for stock_info in stocks_to_process:
        # 使用官方路径函数检查文件是否存在
        target_file_path = get_processed_data_path(stock_info, config)
        if target_file_path.exists():
            keyword = stock_info.get('keyword', stock_info.get('ticker'))
            ticker = stock_info.get('ticker')
            print(f"INFO: 特征文件已存在于 {target_file_path}，跳过 {keyword} ({ticker}) 的数据处理。")
        else:
            # 如果文件不存在，则将该股票加入待处理列表
            tickers_to_generate.append(stock_info['ticker'])

    if not tickers_to_generate:
        print("\n所有股票的特征文件均已存在。无需执行数据处理流水线。")
        return
    
    ticker_to_keyword_map = {s['ticker']: s.get('keyword', s['ticker']) for s in stocks_to_process}
    display_list = [f"{ticker_to_keyword_map.get(t, t)} ({t})" for t in tickers_to_generate]    
    print(f"\n需要为以下 {len(tickers_to_generate)} 只股票生成新数据: {display_list}")

    processed_data = process_all_from_config(config, tickers_to_generate=tickers_to_generate)

    if processed_data:
        save_processed_data(processed_data, config)
    else:
        print("未能从数据管道获取任何新数据，跳过保存步骤。")
    
    print("\n数据管道协调任务执行完毕。")

if __name__ == '__main__':
    # 建议使用 YAML 配置文件以获得更好的可读性
    DEFAULT_CONFIG_PATH = 'configs/system_config.yaml'
    run_data_pipeline(config_path=DEFAULT_CONFIG_PATH)