# --- get_data.py---

import pandas as pd
import numpy as np
import baostock as bs
import tushare as ts
import pandas_ta as ta
import time
import os
from typing import List, Dict, Optional, Any
import hashlib
import json
from pathlib import Path
import re

# --- 全局 API 实例 ---
pro: Optional['ts.ProApi'] = None
bs_logged_in: bool = False

# --- 默认配置 ---
DEFAULT_CONFIG = {
    "macro_keys": {'cn_cpi': 'cpi', 'cn_m2': 'm2'},
    "macro_release_lag_days": {'cpi': 40, 'm2': 40},
    "correlation_window": 30, "labeling_horizon": 30, "ffill_limit": 5, 
    "correlation_threshold": 0.95, "cache_dir": "data_cache"
}

# ------------------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------------------

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
    """初始化API。Baostock是必须的，Tushare是可选的。"""
    global pro, bs_logged_in
    if not bs_logged_in:
        lg = bs.login()
        if lg.error_code != '0':
            raise ConnectionError(f"Baostock 登录失败: {lg.error_msg}")
        bs_logged_in = True
        print(f"INFO: Baostock API 登录成功。SDK版本: {bs.__version__}")
    
    if pro is None:
        token = config.get('global_settings', {}).get('tushare_api_token')
        if token and token != "在这里粘贴您从Tushare官网获取的API密钥" and "TOKEN" not in token.upper():
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
    """1. [BS] 从 Baostock 获取日线行情数据，优先使用本地缓存。"""
    
    # 1. 构建缓存路径
    cache_base_dir = Path(run_config["cache_dir"])
    raw_cache_dir = cache_base_dir / run_config["raw_ohlcv_cache_dir"]
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_filename = f"raw_{ticker}_{start_date}_{end_date}.pkl"
    cache_file_path = raw_cache_dir / cache_filename

    # 2. 检查缓存是否存在
    if cache_file_path.exists():
        try:
            print(f"  - [1/7] 正在从本地缓存加载 {ticker} 的原始日线数据...")
            return pd.read_pickle(cache_file_path)
        except Exception as e:
            print(f"  - WARNING: 读取缓存文件 {cache_file_path} 失败: {e}。将重新下载。")

    # 3. 如果缓存不存在或读取失败，则从API下载
    print(f"  - [1/7] 正在从 Baostock 下载 {ticker} 的日线行情...")
    start_fmt, end_fmt = pd.to_datetime(start_date).strftime('%Y-%m-%d'), pd.to_datetime(end_date).strftime('%Y-%m-%d')
    rs = bs.query_history_k_data_plus(ticker, "date,open,high,low,close,volume", start_date=start_fmt, end_date=end_fmt, frequency="d", adjustflag="2")
    
    if rs.error_code != '0':
        print(f"  - WARNING [BS]: 获取 {ticker} 数据失败: {rs.error_msg}")
        return None
        
    df = rs.get_data()
    if df.empty:
        print(f"  - WARNING [BS]: 未能获取到 {ticker} 在指定日期范围的数据。")
        return None
        
    # 数据清洗和格式化
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={'close': 'close_adj'}, inplace=True)
    for col in ['open', 'high', 'low', 'close_adj', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # 4. 将新下载的数据存入缓存
    try:
        df.to_pickle(cache_file_path)
        print(f"  - INFO: 已将 {ticker} 的原始数据缓存至 {cache_file_path}")
    except Exception as e:
        print(f"  - WARNING: 无法将 {ticker} 的原始数据写入缓存: {e}")
        
    return df[['open', 'high', 'low', 'close_adj', 'volume']]

def _get_macroeconomic_data_cn(start_date: str, end_date: str, config: dict) -> Optional[pd.DataFrame]:
    """2. [TS] (可选) 从 Tushare 获取中国宏观经济指标。"""
    if pro is None: return None
    print("  - [2/7] 正在尝试从 Tushare 获取中国宏观经济数据...")
    try:
        start_m, end_m = (pd.to_datetime(start_date)-pd.DateOffset(days=100)).strftime('%Y%m'), pd.to_datetime(end_date).strftime('%Y%m')
        all_series_df = []
        cpi_df = pro.cn_cpi(start_m=start_m, end_m=end_m, fields='month,nt_val'); time.sleep(0.3)
        cpi_df['date'] = pd.to_datetime(cpi_df['month'], format='%Y%m'); all_series_df.append(cpi_df.set_index('date')[['nt_val']].rename(columns={'nt_val': 'cpi'}))
        m2_df = pro.cn_m(start_m=start_m, end_m=end_m, fields='month,m2'); time.sleep(0.3)
        m2_df['date'] = pd.to_datetime(m2_df['month'], format='%Y%m'); all_series_df.append(m2_df.set_index('date')[['m2']])
        macro_data = pd.concat(all_series_df, axis=1).sort_index()
        for col, lag in config["macro_release_lag_days"].items():
            if col in macro_data.columns: macro_data[col] = macro_data[col].shift(lag)
        print(f"    - SUCCESS: 已成功获取并处理宏观数据。"); return macro_data
    except Exception as e: print(f"    - WARNING: 获取宏观数据失败: {e}。将跳过此步骤。"); return None

def _add_technical_and_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """3. [计算] 添加技术分析指标和日历特征。"""
    print("  - [3/7] 正在添加技术和日历特征...")
    df.ta.ema(length=10, append=True)
    df.ta.ema(length=30, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index)
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    return df

def _add_relative_performance_features(df: pd.DataFrame, benchmark_df: pd.DataFrame, industry_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """4. [计算] 添加相对于市场和行业的表现特征。"""
    print("  - [4/7] 正在添加相对表现特征...")
    df = df.join(benchmark_df, how='left')
    df = df.join(industry_df, how='left')
    df['relative_strength_vs_benchmark'] = df['close_adj'] / df['benchmark_close']
    df['relative_strength_vs_industry'] = df['close_adj'] / df['industry_close']
    window = config["correlation_window"]
    df['correlation_vs_benchmark'] = df['close_adj'].pct_change().rolling(window).corr(df['benchmark_close'].pct_change())
    df['correlation_vs_industry'] = df['close_adj'].pct_change().rolling(window).corr(df['industry_close'].pct_change())
    return df

def _make_features_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """5. [计算] 对价格类特征进行平稳化处理。"""
    print("  - [5/7] 正在对特征进行平稳化...")
    cols_to_log_return = ['open', 'high', 'low', 'close_adj', 'benchmark_close', 'industry_close']
    if 'cpi' in df.columns: cols_to_log_return.append('cpi')
    if 'm2' in df.columns: cols_to_log_return.append('m2')
    for col in cols_to_log_return:
        if col in df.columns: df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
    return df

def _create_and_clean_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """6. [计算] 创建用于预测的标签，并进行降噪处理。"""
    print("  - [6/7] 正在创建并降噪预测标签...")
    horizon = config.get("labeling_horizon", 30)
    df['label_return'] = df['close_adj'].pct_change(periods=horizon).shift(-horizon)
    lower_bound, upper_bound = df['label_return'].quantile(0.01), df['label_return'].quantile(0.99)
    df['label_return'] = df['label_return'].clip(lower=lower_bound, upper=upper_bound)
    return df

def _initial_feature_selection(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """7. [计算] 进行初步的特征筛选，剔除高度共线性的特征。"""
    print("  - [7/7] 正在进行初步特征筛选...")
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = config.get("correlation_threshold", 0.95)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    print(f"    - 移除了 {len(to_drop)} 个高相关性特征。")
    return df

# ------------------------------------------------------------------------------
# 公共 API 函数
# ------------------------------------------------------------------------------
def get_features_for_ticker(
    ticker: str, start_date: str, end_date: str, benchmark_ticker: str, industry_etf_ticker: str,
    config: Dict, shared_data: Dict
) -> Optional[pd.DataFrame]:
    """为单个A股股票执行完整流程，自动选择数据源并使用统一缓存。"""
    run_config = {**DEFAULT_CONFIG, **config.get('global_settings', {})}
    cache_dir = run_config["cache_dir"]; os.makedirs(cache_dir, exist_ok=True)
    
    # L2 缓存：最终特征数据的缓存
    params_hash = hashlib.md5(str((ticker, start_date, end_date, benchmark_ticker, industry_etf_ticker, json.dumps(run_config, sort_keys=True))).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"features_{ticker}_{params_hash[:10]}.pkl"

    if cache_file.exists():
        print(f"INFO: 从L2缓存加载 {ticker} 的最终特征数据...")
        return pd.read_pickle(cache_file)
    
    # --- 核心修改点：将 run_config 传递给 _get_ohlcv_data_bs ---
    df = _get_ohlcv_data_bs(ticker, start_date, end_date, run_config)
    if df is None: return None
    
    if 'macro' not in shared_data:
        shared_data['macro'] = _get_macroeconomic_data_cn(start_date, end_date, run_config)
    
    macro_df = shared_data.get('macro')
    if macro_df is not None and not macro_df.empty:
        df = pd.merge_asof(df.sort_index(), macro_df.sort_index(), left_index=True, right_index=True, direction='backward')

    df = _add_technical_and_calendar_features(df)
    
    if benchmark_ticker not in shared_data:
        bench_df = _get_ohlcv_data_bs(benchmark_ticker, start_date, end_date, run_config)
        shared_data[benchmark_ticker] = bench_df['close_adj'].rename('benchmark_close') if bench_df is not None else None
    
    if industry_etf_ticker not in shared_data:
        ind_df = _get_ohlcv_data_bs(industry_etf_ticker, start_date, end_date, run_config)
        shared_data[industry_etf_ticker] = ind_df['close_adj'].rename('industry_close') if ind_df is not None else None
        
    if shared_data.get(benchmark_ticker) is not None and shared_data.get(industry_etf_ticker) is not None:
        df = _add_relative_performance_features(df, shared_data[benchmark_ticker], shared_data[industry_etf_ticker], run_config)
    
    df = _make_features_stationary(df)
    df = _create_and_clean_labels(df, run_config)
    df = _initial_feature_selection(df, run_config)
    
    df.ffill(inplace=True, limit=run_config["ffill_limit"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    if df.empty: print(f"  - WARNING: 处理完成后，{ticker} 的数据集为空。"); return None
    
    df.to_pickle(cache_file)
    print(f"SUCCESS: 已为 {ticker} 生成特征数据集并缓存至 {cache_file}")
    return df

def process_all_from_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """根据配置文件，加载配置并处理所有股票数据。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
    except FileNotFoundError: print(f"错误: 配置文件 {config_path} 未找到。"); return {}
    
    try: _initialize_apis(config)
    except ConnectionError as e: print(f"错误: 核心API初始化失败: {e}"); return {}

    global_settings = config.get('global_settings', {}); stocks_to_process = config.get('stocks_to_process', [])
    results = {}; shared_data = {}
    
    for i, stock_info in enumerate(stocks_to_process, 1):
        config_ticker = stock_info.get('ticker')
        industry_etf = stock_info.get('industry_etf')
        if not config_ticker or not industry_etf: 
            print(f"\n({i}/{len(stocks_to_process)}) 跳过无效配置: {stock_info}"); continue
        
        print(f"\n({i}/{len(stocks_to_process)}) 正在处理股票: {config_ticker}...")
        
        # 仅在调用底层函数时进行临时格式转换
        api_ticker = _get_api_ticker(config_ticker)
        api_industry_etf = _get_api_ticker(industry_etf)
        api_benchmark = _get_api_ticker(global_settings['benchmark_ticker'])

        df = get_features_for_ticker(
            ticker=api_ticker, 
            start_date=global_settings['start_date'], 
            end_date=global_settings['end_date'],
            benchmark_ticker=api_benchmark, 
            industry_etf_ticker=api_industry_etf,
            config=config, 
            shared_data=shared_data
        )
        if df is not None:
            # --- 核心修复点 ---
            # *** 严格使用 config.json 中的原始 ticker 作为字典的键 ***
            results[config_ticker] = df
    
    bs.logout(); print("\n所有股票处理完毕，Baostock 已登出。")
    return results

# --- 示例用法 ---
if __name__ == "__main__":
    CONFIG_FILE_PATH = 'config.json'
    all_features_data = process_all_from_config(CONFIG_FILE_PATH)
    
    if all_features_data:
        print("\n--- 最终处理结果摘要 ---")
        for ticker, df in all_features_data.items():
            print(f"\n股票: {ticker}")
            print(f"  - 数据维度: {df.shape}")
            if not df.empty:
                print(f"  - 起始日期: {df.index.min().date()}")
                print(f"  - 结束日期: {df.index.max().date()}")