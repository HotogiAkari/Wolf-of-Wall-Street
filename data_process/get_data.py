# 文件路径: data_process/get_data.py
'''
用于获取数据并计算特征
'''

import pandas as pd
import numpy as np
import baostock as bs
import tushare as ts
import time
from typing import Dict, Optional, Tuple
import json
import yaml
from pathlib import Path
import re
from data_process import feature_calculators
from data_process import data_contracts

# --- 全局 API 实例 ---
pro: Optional['ts.ProApi'] = None
bs_logged_in: bool = False

# 公共 API 生命周期管理函数

def initialize_apis(config: Dict):
    """
    (公共接口) 初始化所有数据 API。应在任何数据处理任务开始前调用。
    """
    global pro, bs_logged_in
    
    if not bs_logged_in:
        print("INFO: 尝试登陆 Baostock...")
        lg = bs.login()
        if lg.error_code != '0':
            raise ConnectionError(f"Baostock 登录失败: {lg.error_msg}")
        bs_logged_in = True
        print(f"INFO: Baostock API 登录成功。SDK版本: {bs.__version__}")
    else:
        print("INFO: Baostock API 已登陆.")
    
    if pro is None:
        token = config.get('global_settings', {}).get('tushare_api_token')
        if token and "TOKEN" not in token.upper():
            try:
                ts.set_token(token)
                pro = ts.pro_api()
                print("INFO: Tushare Pro API 初始化成功。将尝试获取宏观数据。")
            except Exception as e:
                print(f"WARNING: Tushare API 初始化失败: {e}。将跳过宏观数据获取。")
                pro = None
        else:
            print("INFO: 未在配置中提供有效的 Tushare Token。将跳过宏观数据获取。")

def shutdown_apis():
    """
    (公共接口) 安全地登出所有数据 API，并重置状态。应在所有数据处理任务结束后调用。
    """
    global bs_logged_in
    
    try:
        if bs_logged_in:
            bs.logout()
            bs_logged_in = False # 重置状态标志
            print("INFO: Baostock API 已成功登出.")
        else:
            print("INFO: Baostock API 未登录, 未执行任何操作.")
    except Exception as e:
        print(f"WARNNING: 在 Baostock 登出时发生错误: {e}")

# 内部辅助函数

def _download_with_retry(api_call_func, max_retries=3, initial_delay=0.5):
    """
    一个通用的下载重试包装器，处理网络错误。
    :param api_call_func: 要执行的 API 调用函数 (例如 bs.query_history_k_data_plus)
    :param max_retries: 最大重试次数
    :param initial_delay: 初始延迟秒数，每次重试后会加倍
    :return: API 调用的结果
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

def _get_api_ticker(ticker_from_config: str) -> str:
    if ticker_from_config is None: return ""
    ticker = ticker_from_config.lower().strip()
    match = re.match(r'(?:(sh|sz)[.\\s]*)?(\d{6})(?:[.\\s]*(sh|sz))?', ticker)
    if not match: return ticker
    groups = match.groups()
    market, code = groups[0] or groups[2], groups[1]
    if not market or not code: return ticker
    return f"{market}.{code}"

# 核心初始化与数据获取函数

def _initialize_apis(config: Dict):
    """初始化API。"""
    global pro, bs_logged_in
    if not bs_logged_in:
        lg = bs.login()
        if lg.error_code != '0':
            raise ConnectionError(f"Baostock 登录失败: {lg.error_msg}")
        bs_logged_in = True
        print(f"INFO: Baostock API 登录成功。SDK版本: {bs.__version__}")
    
    if pro is None:
        token = config.get('global_settings', {}).get('tushare_api_token')
        if token and "TOKEN" not in token.upper():
            try:
                ts.set_token(token)
                pro = ts.pro_api()
                print("INFO: Tushare Pro API 初始化成功。将尝试获取宏观数据。")
            except Exception as e:
                print(f"WARNING: Tushare API 初始化失败: {e}。将跳过宏观数据获取。")
                pro = None
        else:
            print("INFO: 未在配置中提供有效的 Tushare Token。将跳过宏观数据获取。")

def _get_ohlcv_data_bs(ticker: str, start_date: str, end_date: str, run_config: dict, keyword: str = None) -> Optional[pd.DataFrame]:
    """[BS] 从 Baostock 获取日线行情数据，优先使用本地缓存 (含延迟和重试)。"""
    display_name = keyword if keyword else ticker
    cache_base_dir = Path(run_config.get("data_cache_dir", "data_cache"))
    raw_cache_dir = cache_base_dir / run_config.get("raw_ohlcv_cache_dir", "raw_ohlcv")
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_filename = f"raw_{ticker}_{start_date}_{end_date}.pkl"
    cache_file_path = raw_cache_dir / cache_filename

    if cache_file_path.exists():
        print(f"  - [1/7] 正在从本地缓存加载 {display_name} 的原始日线数据...")
        return pd.read_pickle(cache_file_path)

    print(f"  - [1/7] 正在从 Baostock 下载 {display_name} 的日线行情...")
    start_fmt, end_fmt = pd.to_datetime(start_date).strftime('%Y-%m-%d'), pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    api_call = lambda: bs.query_history_k_data_plus(ticker, "date,open,high,low,close,volume", start_date=start_fmt, end_date=end_fmt, frequency="d", adjustflag="2")
    rs = _download_with_retry(api_call)
    
    if rs.error_code != '0':
        print(f"  - WARNING [BS]: 获取 {display_name} 数据失败: {rs.error_msg}")
        return None
        
    df = rs.get_data()
    if df.empty:
        print(f"  - WARNING [BS]: 未能获取到 {display_name} 在指定日期范围的数据。")
        return None
        
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={'close': 'close'}, inplace=True) 
    df.columns = df.columns.str.lower()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    try:
        df.to_pickle(cache_file_path)
        print(f"  - INFO: 已将 {display_name} 的原始数据缓存至 {cache_file_path}")
    except Exception as e:
        print(f"  - WARNING: 无法将 {display_name} 的原始数据写入缓存: {e}")
        
    return df[['open', 'high', 'low', 'close', 'volume']]

def _get_macroeconomic_data_cn(start_date: str, end_date: str, config: dict) -> Optional[pd.DataFrame]:
    """[TS] (可选) 从 Tushare 获取中国宏观经济指标。"""
    if pro is None: return None
    print("  - [2/7] 正在尝试从 Tushare 获取中国宏观经济数据...")
    try:
        start_m = (pd.to_datetime(start_date) - pd.DateOffset(days=100)).strftime('%Y%m')
        end_m = pd.to_datetime(end_date).strftime('%Y%m')
        
        all_series_df = []
        cpi_df = pro.cn_cpi(start_m=start_m, end_m=end_m, fields='month,nt_val'); time.sleep(0.3)
        cpi_df['date'] = pd.to_datetime(cpi_df['month'], format='%Y%m')
        all_series_df.append(cpi_df.set_index('date')[['nt_val']].rename(columns={'nt_val': 'cpi'}))
        
        m2_df = pro.cn_m(start_m=start_m, end_m=end_m, fields='month,m2'); time.sleep(0.3)
        m2_df['date'] = pd.to_datetime(m2_df['month'], format='%Y%m')
        all_series_df.append(m2_df.set_index('date')[['m2']])
        
        macro_data = pd.concat(all_series_df, axis=1).sort_index()
        for col, lag in config.get("macro_release_lag_days", {}).items():
            if col in macro_data.columns: 
                macro_data[col] = macro_data[col].shift(lag)
        print(f"    - SUCCESS: 已成功获取并处理宏观数据。")
        return macro_data
    except Exception as e: 
        print(f"    - WARNING: 获取宏观数据失败: {e}。将跳过此步骤。")
        return None

def _add_relative_performance_features(df: pd.DataFrame, benchmark_df: pd.DataFrame, industry_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """[计算] 添加相对于市场和行业的表现特征。"""
    print("  - [4/7] 正在添加相对表现特征...")
    df = df.join(benchmark_df, how='left')
    df = df.join(industry_df, how='left')
    df['relative_strength_vs_benchmark'] = df['close'] / df['benchmark_close']
    df['relative_strength_vs_industry'] = df['close'] / df['industry_close']
    window = config.get("correlation_window", 30)
    df['correlation_vs_benchmark'] = df['close'].pct_change().rolling(window).corr(df['benchmark_close'].pct_change())
    df['correlation_vs_industry'] = df['close'].pct_change().rolling(window).corr(df['industry_close'].pct_change())
    return df

def _make_features_stationary(df: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    """[计算] 对价格类特征进行平稳化处理。"""
    print("  - [5/7] 正在对特征进行平稳化...")
    cols_to_log_return = df.columns.intersection(['open', 'high', 'low', 'close', 'benchmark_close', 'industry_close', 'cpi', 'm2'])
    for col in cols_to_log_return:
        if col in df.columns: 
            df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
    return df

def _create_and_clean_labels(df: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    """[计算] 创建用于预测的标签，并进行降噪处理。"""
    print("  - [6/7] 正在创建并降噪预测标签...")
    horizon = run_config.get("labeling_horizon", 30)
    label_col = run_config.get('label_column', 'label_return')
    df[label_col] = df['close'].pct_change(periods=horizon).shift(-horizon)
    lower_bound, upper_bound = df[label_col].quantile(0.01), df[label_col].quantile(0.99)
    df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
    return df

def _initial_feature_selection(df: pd.DataFrame, run_config: dict) -> pd.DataFrame:
    """[计算] 进行初步的特征筛选，剔除高度共线性的特征。"""
    print("  - [7/7] 正在进行初步特征筛选...")
    core_features = {'open', 'high', 'low', 'close', 'volume'}
    numeric_df = df.select_dtypes(include=np.number)
    features_to_check = [col for col in numeric_df.columns if col not in core_features]
    if not features_to_check:
        print("    - 没有非核心特征需要检查相关性。"); return df
    corr_matrix = numeric_df[features_to_check].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    threshold = run_config.get("correlation_threshold", 0.95)
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        print(f"    - 移除了 {len(to_drop)} 个高相关性特征: {to_drop}")
    else:
        print("    - 未发现高相关性特征需要移除。")
    return df

# 公共 API 函数

def get_full_feature_df(ticker: str, config: Dict, keyword: str = None, prediction_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    (最终版) 为单个股票执行完整的特征生成流程。
    - prediction_mode=False (默认): 使用配置文件中的长期日期设置，用于训练。
    - prediction_mode=True: 忽略配置文件的日期，自动获取近期数据，用于预测。
    """
    display_name = keyword if keyword else ticker
    print(f"\n--- Generating features for {display_name} ({ticker}) ---")
    
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    stock_info = next((s for s in config.get('stocks_to_process', []) if s['ticker'] == ticker), {})
    run_config = {**global_settings, **strategy_config, **stock_info}

    # --- 核心修正：根据模式选择日期计算逻辑 ---
    if prediction_mode:
        # 预测模式：只需要获取最近的一段数据即可计算所有特征
        print("  - Running in Prediction Mode: Fetching recent data.")
        end_date_dt = pd.Timestamp.now()
        # 回溯大约 200 个交易日，足以计算所有常用指标
        start_date_dt = end_date_dt - pd.DateOffset(days=300) 
    else:
        # 训练模式：使用配置文件中的混合日期策略
        print("  - Running in Training Mode: Fetching historical data based on config.")
        end_date_dt = pd.to_datetime(run_config.get('end_date'))
        lookback_years = run_config.get('data_lookback_years', 10)
        earliest_start_date_dt = pd.to_datetime(run_config.get('earliest_start_date', '2010-01-01'))
        
        if not end_date_dt or not earliest_start_date_dt:
             print("ERROR: 'end_date' or 'earliest_start_date' not found in config for training mode.")
             return None

        target_start_date_dt = end_date_dt - pd.DateOffset(years=lookback_years)
        start_date_dt = max(target_start_date_dt, earliest_start_date_dt)
    
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    
    print(f"  - Data window: Requesting data from {start_date_str} to {end_date_str}.")

    # --- 2. 准备 API 参数 ---
    benchmark_ticker = run_config.get('benchmark_ticker')
    industry_etf_ticker = run_config.get('industry_etf')

    if not benchmark_ticker or not industry_etf_ticker:
        print(f"ERROR: Missing 'benchmark_ticker' or 'industry_etf' for {display_name}.")
        return None

    api_ticker = _get_api_ticker(ticker)
    api_benchmark = _get_api_ticker(benchmark_ticker)
    api_industry = _get_api_ticker(industry_etf_ticker)

    # --- 3. 数据获取 ---
    df = _get_ohlcv_data_bs(api_ticker, start_date_str, end_date_str, run_config)
    if df is None or df.empty:
        print(f"  - WARNNING: No data returned for {display_name} in the requested window. Skipping.")
        return None
    
    print(f"  - INFO: Received data for {display_name} from {df.index.min().date()} to {df.index.max().date()}.")

    # --- 4. 特征工程流水线 (所有后续步骤保持不变) ---
    
    # 4.1 宏观数据
    macro_df = _get_macroeconomic_data_cn(start_date_str, end_date_str, run_config)
    if macro_df is not None and not macro_df.empty:
        df = pd.merge_asof(df.sort_index(), macro_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    # 4.2 技术/日历特征
    df = feature_calculators.run_all_feature_calculators(df, run_config)
    
    # 4.3 相对表现特征
    bench_df_raw = _get_ohlcv_data_bs(api_benchmark, start_date_str, end_date_str, run_config)
    if bench_df_raw is None:
        print(f"ERROR: Could not get benchmark data for {display_name}. Aborting."); return None
    bench_df = bench_df_raw['close'].rename('benchmark_close')

    ind_df_raw = _get_ohlcv_data_bs(api_industry, start_date_str, end_date_str, run_config)
    if ind_df_raw is None:
        print(f"WARNNING: Could not get industry data for {display_name}. Using benchmark as fallback.")
        ind_df = bench_df_raw['close'].rename('industry_close')
    else:
        ind_df = ind_df_raw['close'].rename('industry_close')
        
    df = _add_relative_performance_features(df, bench_df, ind_df, run_config)
    
    # 4.4 平稳化、标签创建、初步筛选
    df = _make_features_stationary(df, run_config)
    df = _create_and_clean_labels(df, run_config)
    df = _initial_feature_selection(df, run_config)
    
    df.columns = df.columns.str.lower()
    
    # --- 5. 数据清洗 ---
    ffill_limit = run_config.get("ffill_limit", 5)
    df.ffill(inplace=True, limit=ffill_limit)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    label_col = run_config.get('label_column', 'label_return')
    if label_col in df.columns: df.dropna(subset=[label_col], inplace=True)
    
    feature_cols = [col for col in df.columns if col != label_col]
    df.dropna(subset=feature_cols, inplace=True)
    
    if df.empty: 
        print(f"WARNNING: DataFrame is empty for {display_name} after all processing."); return None
    
    # --- 6. 数据校验 ---
    validator = data_contracts.DataValidator(run_config)
    if not validator.validate_schema(df):
        print(f"ERROR: Final data validation failed for {display_name}. Aborting."); return None
        
    print(f"--- SUCCESS: Features generated for {display_name}. Shape: {df.shape} ---")
    return df

def process_all_from_config(config_path: str, tickers_to_generate: list = None) -> Dict[str, pd.DataFrame]:
    """
    (已重构) 根据配置文件，为指定的股票列表生成特征。
    如果 tickers_to_generate 为 None，则处理所有股票。
    """
    print("--- 开始批量特征生成 ---")
    if tickers_to_generate:
        print(f"针对特定股票: {len(tickers_to_generate)} 生成特征.")
    else:
        print("针对配置文件中的所有股票代码生成特征.")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) if config_path.endswith(('.yaml', '.yml')) else json.load(f)
    except Exception as e:
        print(f"ERROR: Config file {config_path} not found or failed to parse: {e}"); return {}
    
    # API 初始化应该由更高层（如 Notebook）管理，这里不再调用
    
    results_df = {}
    stocks_to_process = config.get('stocks_to_process', [])
    if not stocks_to_process:
        print("WARNNING: 'stocks_to_process' list is empty. No data will be processed."); return {}
    
    # 如果指定了目标列表，就只循环这个列表
    if tickers_to_generate:
        # 从完整的股票池中筛选出需要处理的 stock_info
        target_stocks = [s for s in stocks_to_process if s['ticker'] in tickers_to_generate]
    else:
        # 否则，处理全部
        target_stocks = stocks_to_process

    for i, stock_info in enumerate(target_stocks, 1):
        ticker, keyword = stock_info.get('ticker'), stock_info.get('keyword', stock_info.get('ticker'))
        if not ticker: 
            print(f"  - WARNNING: Skipping invalid config entry at index {i-1} (missing ticker)."); continue
        
        df = get_full_feature_df(ticker, config, keyword)
        if df is not None: results_df[ticker] = df
    
    print("--- Batch Feature Generation Process Finished ---")
    return results_df