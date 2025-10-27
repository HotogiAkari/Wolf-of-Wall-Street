# 文件路径: utils/date_utils.py

import pandas as pd
from typing import Dict

def resolve_data_pipeline_dates(config: Dict) -> None:
    """
    根据配置文件中的 end_date 和回溯期，动态计算并注入 start_date。
    这是一个工具函数，用于确保日期计算逻辑在整个项目中的一致性。

    Args:
        config (Dict): 项目的全局配置字典，函数会直接修改此字典。
    """
    try:
        strategy_config = config['strategy_config']
        
        end_date_str = strategy_config.get('dynamic_end_date', strategy_config['end_date'])
        
        end_date_dt = pd.to_datetime(end_date_str)
        lookback_years = strategy_config.get('data_lookback_years', 10)
        earliest_start_date_dt = pd.to_datetime(strategy_config['earliest_start_date'])
        
        target_start_date_dt = end_date_dt - pd.DateOffset(years=lookback_years)
        start_date_dt = max(target_start_date_dt, earliest_start_date_dt)
        
        calculated_start_date = start_date_dt.strftime('%Y-%m-%d')
        
        config['strategy_config']['start_date'] = calculated_start_date
        
        print(f"INFO: 日期已解析. Start Date: {calculated_start_date}, End Date: {end_date_str}")

    except KeyError as e:
        print(f"ERROR: 动态计算 start_date 失败，因为缺少关键配置项: {e}。")
        raise