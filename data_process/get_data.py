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

# ------------------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------------------
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
            # 尝试执行 API 调用
            result = api_call_func()
            # 如果成功，立即返回结果
            return result
        except Exception as e:
            # 捕获任何异常，特别是网络相关的
            retries += 1
            print(f"  - WARNNING: API call failed (Attempt {retries}/{max_retries}). Retrying in {delay:.2f} seconds. Error: {e}")
            # 等待一段时间
            time.sleep(delay)
            # 增加下一次的等待时间 (指数退避)
            delay *= 2
    
    # 如果所有重试都失败了
    print(f"  - ERROR: API call failed after {max_retries} retries.")
    # 返回一个模拟的失败结果，以便上游函数处理
    class FailedResponse:
        def __init__(self):
            self.error_code = '-1'
            self.error_msg = 'Max retries exceeded'
        def get_data(self):
            return pd.DataFrame()
            
    return FailedResponse()

def _get_api_ticker(ticker_from_config: str) -> str:
    """从配置文件中的ticker (e.g., '600519.SH') 提取出API调用所需的格式 (e.g., 'sh.600519')。"""
    if ticker_from_config is None: return ""
    ticker = ticker_from_config.lower().strip()
    match = re.match(r'(?:(sh|sz)[.\\s]*)?(\d{6})(?:[.\\s]*(sh|sz))?', ticker)
    if not match: return ticker
    groups = match.groups()
    market = groups[0] or groups[2]
    code = groups[1]
    if not market or not code: return ticker
    return f"{market}.{code}"

# ------------------------------------------------------------------------------
# 核心初始化与数据获取函数
# ------------------------------------------------------------------------------

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

def _get_ohlcv_data_bs(ticker: str, start_date: str, end_date: str, run_config: dict) -> Optional[pd.DataFrame]:
    """[BS] 从 Baostock 获取日线行情数据，优先使用本地缓存 (含延迟和重试)。"""
    
    cache_base_dir = Path(run_config.get("global_settings", {}).get("data_cache_dir", "data_cache"))
    raw_cache_dir = cache_base_dir / run_config.get("global_settings", {}).get("raw_ohlcv_cache_dir", "raw_ohlcv")
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_filename = f"raw_{ticker}_{start_date}_{end_date}.pkl"
    cache_file_path = raw_cache_dir / cache_filename

    if cache_file_path.exists():
        print(f"  - [1/7] 正在从本地缓存加载 {ticker} 的原始日线数据...")
        return pd.read_pickle(cache_file_path)

    print(f"  - [1/7] 正在从 Baostock 下载 {ticker} 的日线行情...")
    start_fmt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_fmt = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # --- 核心修正：使用带重试的下载逻辑 ---
    # 定义一个无参数的 lambda 函数来包裹 API 调用
    api_call = lambda: bs.query_history_k_data_plus(
        ticker, 
        "date,open,high,low,close,volume", 
        start_date=start_fmt, 
        end_date=end_fmt, 
        frequency="d", 
        adjustflag="2"
    )
    
    # 执行带重试的下载
    rs = _download_with_retry(api_call)
    # --- 修正结束 ---
    
    if rs.error_code != '0':
        print(f"  - WARNING [BS]: 获取 {ticker} 数据失败: {rs.error_msg}")
        return None
        
    df = rs.get_data()
    if df.empty:
        print(f"  - WARNING [BS]: 未能获取到 {ticker} 在指定日期范围的数据。")
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
        print(f"  - INFO: 已将 {ticker} 的原始数据缓存至 {cache_file_path}")
    except Exception as e:
        print(f"  - WARNING: 无法将 {ticker} 的原始数据写入缓存: {e}")
        
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

def _make_features_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """[计算] 对价格类特征进行平稳化处理。"""
    print("  - [5/7] 正在对特征进行平稳化...")
    # 从配置中获取需要平稳化的列
    cols_to_log_return = df.columns.intersection(
        ['open', 'high', 'low', 'close', 'benchmark_close', 'industry_close', 'cpi', 'm2']
    )
    for col in cols_to_log_return:
        df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
    return df

def _create_and_clean_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """[计算] 创建用于预测的标签，并进行降噪处理。"""
    print("  - [6/7] 正在创建并降噪预测标签...")
    horizon = config.get("labeling_horizon", 30)
    label_col = config.get('label_column', 'label_return')
    
    df[label_col] = df['close'].pct_change(periods=horizon).shift(-horizon)
    lower_bound = df[label_col].quantile(0.01)
    upper_bound = df[label_col].quantile(0.99)
    df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
    return df

def _initial_feature_selection(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """[计算] 进行初步的特征筛选，剔除高度共线性的特征。"""
    print("  - [7/7] 正在进行初步特征筛选...")
    
    # --- 核心修正：保护核心列不被移除 ---
    core_features = {'open', 'high', 'low', 'close', 'volume'}
    
    numeric_df = df.select_dtypes(include=np.number)
    
    # 从相关性计算中排除核心列，以避免它们被错误识别
    features_to_check = [col for col in numeric_df.columns if col not in core_features]
    if not features_to_check:
        print("    - 没有非核心特征需要检查相关性。")
        return df

    corr_matrix = numeric_df[features_to_check].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = config.get("strategy_config",{}).get("correlation_threshold", 0.95)
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        print(f"    - 移除了 {len(to_drop)} 个高相关性特征: {to_drop}")
    else:
        print("    - 未发现高相关性特征需要移除。")
    return df

# ------------------------------------------------------------------------------
# 公共 API 函数
# ------------------------------------------------------------------------------
def get_full_feature_df(ticker: str, config: Dict, keyword: str = None) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    为单个股票执行完整的特征生成流程。
    返回一个元组: (处理好的 DataFrame 或 None, 实际使用的配置字典)
    """
    display_name = keyword if keyword else ticker
    print(f"\n--- Generating features for {display_name} ({ticker}) ---")
    
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    stock_info = next((s for s in config.get('stocks_to_process', []) if s['ticker'] == ticker), {})
    run_config = {**global_settings, **strategy_config, **stock_info}
    
    # 在函数开始时就准备好 stock_info_runtime，以便任何退出路径都能返回它
    stock_info_runtime = stock_info.copy()

    start_date = run_config.get('start_date')
    end_date = run_config.get('end_date')
    benchmark_ticker = run_config.get('benchmark_ticker')
    industry_etf_ticker = run_config.get('industry_etf')

    if not benchmark_ticker or not industry_etf_ticker:
        print(f"ERROR: Missing 'benchmark_ticker' or 'industry_etf' for {display_name}.")
        return None, stock_info_runtime # <--- 修正 1

    api_ticker = _get_api_ticker(ticker)
    api_benchmark = _get_api_ticker(benchmark_ticker)
    api_industry = _get_api_ticker(industry_etf_ticker)

    df = _get_ohlcv_data_bs(api_ticker, start_date, end_date, run_config)
    if df is None: 
        return None, stock_info_runtime # <--- 修正 2

    macro_df = _get_macroeconomic_data_cn(start_date, end_date, run_config)
    if macro_df is not None and not macro_df.empty:
        print("INFO: Merging macroeconomic data.")
        df = pd.merge_asof(df.sort_index(), macro_df.sort_index(), left_index=True, right_index=True, direction='backward')
    else:
        print("INFO: No macroeconomic data to merge. Proceeding without it.")
    
    df = feature_calculators.run_all_feature_calculators(df, run_config)
    
    bench_df_raw = _get_ohlcv_data_bs(api_benchmark, start_date, end_date, run_config)
    if bench_df_raw is None:
        print(f"ERROR: Could not get benchmark data for {display_name}. Aborting feature generation.")
        return None, stock_info_runtime # <--- 修正 3
    bench_df = bench_df_raw['close'].rename('benchmark_close')

    ind_df_raw = _get_ohlcv_data_bs(api_industry, start_date, end_date, run_config)
    if ind_df_raw is None:
        print(f"WARNNING: Could not get industry data for {display_name}. Using benchmark data as a fallback for industry.")
        ind_df = bench_df_raw['close'].rename('industry_close')
        stock_info_runtime['industry_etf'] = benchmark_ticker
    else:
        ind_df = ind_df_raw['close'].rename('industry_close')
        
    df = _add_relative_performance_features(df, bench_df, ind_df, run_config)
    
    df = _make_features_stationary(df)
    df = _create_and_clean_labels(df, run_config)
    df = _initial_feature_selection(df, run_config)
    
    df.columns = df.columns.str.lower()
    
    ffill_limit = run_config.get("ffill_limit", 5)
    df.ffill(inplace=True, limit=ffill_limit)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    label_col = run_config.get('label_column', 'label_return')
    if label_col in df.columns:
        df.dropna(subset=[label_col], inplace=True)
    
    feature_cols = [col for col in df.columns if col != label_col]
    df.dropna(subset=feature_cols, inplace=True)
    
    if df.empty: 
        print(f"WARNNING: DataFrame is empty for {display_name} after all processing.")
        return None, stock_info_runtime
    
    validator = data_contracts.DataValidator(run_config)
    if not validator.validate_schema(df):
        print(f"ERROR: Final data validation failed for {display_name}. Aborting.")
        return None, stock_info_runtime
        
    print(f"--- SUCCESS: Features generated for {display_name}. Shape: {df.shape} ---")
    return df, stock_info_runtime

def process_all_from_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """
    根据配置文件，为所有股票生成特征并返回一个字典。
    """
    print("\n" + "="*80)
    print("--- Starting Batch Feature Generation Process ---")
    print(f"Using config file: {config_path}")
    print("="*80)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"ERROR: Config file {config_path} not found or failed to parse: {e}"); return {}
    
    try:
        _initialize_apis(config)
    except ConnectionError as e:
        print(f"ERROR: API initialization failed: {e}"); return {}

    results_df = {}
    results_runtime_config = {} # 新增一个字典来存储运行时的配置
    stocks_to_process = config.get('stocks_to_process', [])
    
    for i, stock_info in enumerate(stocks_to_process, 1):
        ticker = stock_info.get('ticker')
        keyword = stock_info.get('keyword', ticker)
        
        if not ticker: 
            print("  - WARNNING: Skipping invalid config entry (missing ticker)."); continue
        
        df, runtime_stock_info = get_full_feature_df(ticker, config, keyword)
        
        if df is not None:
            results_df[ticker] = df
            results_runtime_config[ticker] = runtime_stock_info
    
    print("--- Batch Feature Generation Process Finished ---")
    return results_df, results_runtime_config