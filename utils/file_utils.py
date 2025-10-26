import os
import time
import pandas as pd
from typing import Dict
from pathlib import Path

def find_latest_artifact_paths(model_dir: Path, model_type: str) -> dict:
    """
    在指定目录中，根据文件名中的日期版本号找到最新模型及其关联构件的路径。
    """
    file_suffixes = {'lgbm': '.pkl', 'lstm': '.pt', 'tabtransformer': '.pt'}
    model_suffix = file_suffixes.get(model_type, '.pkl')
    
    model_files = list(model_dir.glob(f"{model_type}_model_*{model_suffix}"))
    if not model_files:
        raise FileNotFoundError(f"未在目录 {model_dir} 中找到任何 {model_type.upper()} 的模型文件 (匹配 *{model_suffix})。")
    
    try:
        latest_model_file = sorted(model_files, key=lambda f: f.stem.split('_')[-1])[-1]
    except IndexError:
        raise ValueError("无法从模型文件名中解析出日期版本以进行排序。")

    version_date = latest_model_file.stem.split('_')[-1]
    
    paths = {
        'model': latest_model_file,
        'scaler': model_dir / f"{model_type}_scaler_{version_date}.pkl",
        'meta': model_dir / f"{model_type}_meta_{version_date}.json",
        'encoders': model_dir / f"{model_type}_encoders_{version_date}.pkl",
        'timestamp': version_date
    }
    
    return paths

def download_with_retry(api_call_func, max_retries=3, initial_delay=0.5):
    """
    一个通用的下载重试包装器，处理网络错误。
    """
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            result = api_call_func()
            return result
        except Exception as e:
            retries += 1
            print(f"  - WARNNING: API call failed (Attempt {retries}/{max_retries}). Retrying in {delay:.2f} seconds. Error: {e}")
            time.sleep(delay)
            delay *= 2
    
    print(f"  - ERROR: API call failed after {max_retries} retries.")
    class FailedResponse:
        def __init__(self):
            self.error_code = '-1'
            self.error_msg = 'Max retries exceeded'
        def get_data(self):
            return pd.DataFrame()
    return FailedResponse()

def get_l1_cache_path(ticker: str, start_date: str, end_date: str, config: Dict) -> Path:
    """
    为 L1 原始 OHLCV 数据生成确定性的缓存文件路径。
    将文件名构建逻辑集中在此处。
    """
    cache_dir = Path(config.get('global_settings', {}).get("data_cache_dir", "data_cache"))
    raw_ohlcv_dir = cache_dir / config.get('global_settings', {}).get('raw_ohlcv_cache_dir', 'raw_ohlcv')
    
    ticker_filename_safe = ticker.replace('.', '_')
    
    return raw_ohlcv_dir / f"raw_{ticker_filename_safe}_{start_date}_{end_date}.pkl"