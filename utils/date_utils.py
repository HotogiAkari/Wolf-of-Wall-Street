# 文件路径: utils/date_utils.py

import pandas as pd
from typing import Dict

def resolve_data_pipeline_dates(config: Dict) -> None:
    """
    根据配置文件中的 end_date 和回溯期，动态计算并注入 start_date。
    这个函数现在只从 'data' 配置组读取参数。
    """
    try:
        # 所有参数都从 'data' 组获取
        data_config = config['data']
        
        end_date_str = data_config.get('dynamic_end_date', data_config['end_date'])
        
        end_date_dt = pd.to_datetime(end_date_str)
        lookback_years = data_config.get('data_lookback_years', 10)
        earliest_start_date_dt = pd.to_datetime(data_config['earliest_start_date'])
        
        target_start_date_dt = end_date_dt - pd.DateOffset(years=lookback_years)
        start_date_dt = max(target_start_date_dt, earliest_start_date_dt)
        
        calculated_start_date = start_date_dt.strftime('%Y-%m-%d')
        
        # 将计算出的 start_date 写回到 'data' 组
        config['data']['start_date'] = calculated_start_date
        
        print(f"INFO: 日期已解析. Start Date: {calculated_start_date}, End Date: {end_date_str}")

    except KeyError as e:
        print(f"ERROR: 动态计算 start_date 失败，因为 'data' 配置组中缺少关键项: {e}。")
        raise