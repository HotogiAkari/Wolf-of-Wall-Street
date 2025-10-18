# 文件路径: data_process/get_data.py
import re
import json
import time
import yaml
import numpy as np
import pandas as pd
import tushare as ts
import akshare as ak
import baostock as bs
import yfinance as yf
from pathlib import Path
from typing import Dict, Optional
from tqdm.autonotebook import tqdm
from data_process import data_contracts
from data_process import feature_calculators
from data_process.feature_postprocessors import run_all_feature_postprocessors

# --- 全局 API 实例 ---
pro: Optional['ts.ProApi'] = None
bs_logged_in: bool = False

# 公共 API 生命周期管理函数
def initialize_apis(config: Dict):
    """(公共接口) 初始化所有数据 API。"""
    global pro, bs_logged_in
    if not bs_logged_in:
        print("INFO: 正在尝试登录 Baostock...")
        lg = bs.login()
        if lg.error_code != '0': raise ConnectionError(f"Baostock 登录失败: {lg.error_msg}")
        bs_logged_in = True; print(f"INFO: Baostock API 登录成功。")
    
    if pro is None:
        token = config.get('global_settings', {}).get('tushare_api_token')
        if token and "TOKEN" not in token.upper():
            try:
                print("INFO: 正在尝试初始化 Tushare Pro API...")
                pro = ts.pro_api(token)
                pro.trade_cal(exchange='', start_date='20200101', end_date='20200101')
                print("INFO: Tushare Pro API 初始化并验证成功。")
            except Exception as e:
                print(f"WARNNING: Tushare Pro API 初始化或验证失败: {e}。")
                pro = None
        else:
            print("INFO: 未在配置中提供有效的 Tushare Token，将跳过 Tushare 相关数据。")

def shutdown_apis():
    """
    安全地登出所有数据 API，并重置状态。应在所有数据处理任务结束后调用。
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

def _generate_market_breadth_data(start_date: str, end_date: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """
    计算并缓存全市场的广度指标。
    应该在处理任何个股之前运行一次。
    """
    print("--- 开始生成市场广度数据 ---")
    cache_file = cache_dir / f"market_breadth_{start_date}_{end_date}.pkl"
    if cache_file.exists():
        print(f"  - 正在从缓存加载市场广度数据: {cache_file}")
        return pd.read_pickle(cache_file)

    print("  - 正在获取指数成分股列表...")
    # 此处以沪深300为例
    rs_stocks = bs.query_hs300_stocks()
    if rs_stocks.error_code != '0':
        print(f"  - 错误: 无法获取沪深300成分股: {rs_stocks.error_msg}")
        return None
    
    stock_codes = rs_stocks.get_data()['code'].tolist()
    
    all_closes = []
    for code in tqdm(stock_codes, desc="下载成分股日线数据"):
        df_stock = _get_ohlcv_data_bs(code, start_date, end_date, cache_dir)
        if df_stock is not None and not df_stock.empty:
            all_closes.append(df_stock['close'].rename(code))
        time.sleep(0.1)
            
    if not all_closes:
        print("  - 错误: 未能下载任何成分股数据。")
        return None
        
    closes_df = pd.concat(all_closes, axis=1)
    returns_df = closes_df.pct_change().fillna(0)
    
    # 1. 计算 A/D Line (上涨/下跌线)
    # 这是最经典的广度指标，衡量每日上涨股票数与下跌股票数的差值累加。
    advances = (returns_df > 0).sum(axis=1)
    declines = (returns_df < 0).sum(axis=1)
    ad_line = (advances - declines).cumsum()
    
    # 2. 计算 NH-NL (新高-新低)
    # 衡量创出N日新高的股票数与创出N日新低的股票数的差值。
    rolling_high = closes_df.rolling(52*5).max() # 52周新高
    rolling_low = closes_df.rolling(52*5).min()  # 52周新低
    
    new_highs = (closes_df >= rolling_high.shift(1)).sum(axis=1)
    new_lows = (closes_df <= rolling_low.shift(1)).sum(axis=1)
    nh_nl = (new_highs - new_lows).rolling(10).mean() # 取10日移动平均使其平滑

    breadth_df = pd.DataFrame({
        'market_ad_line': ad_line,
        'market_nh_nl_10d_ma': nh_nl
    })
    
    breadth_df.to_pickle(cache_file)
    print(f"--- 市场广度数据已生成并缓存至 {cache_file} ---")
    return breadth_df

# 核心初始化与数据获取函数
def _get_ohlcv_data_bs(ticker: str, start_date: str, end_date: str, cache_dir: Path, keyword: str = None) -> Optional[pd.DataFrame]:
    """
    (已简化) 从 Baostock 获取日线行情数据，只依赖直接的路径参数。
    """
    display_name = keyword if keyword else ticker
    raw_cache_dir = cache_dir / "raw_ohlcv"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = raw_cache_dir / f"raw_{ticker}_{start_date}_{end_date}.pkl"

    if cache_file_path.exists():
        print(f"  - 正在从本地缓存加载 {display_name} 的原始日线数据...")
        return pd.read_pickle(cache_file_path)

    print(f"  - 正在从 Baostock 下载 {display_name} 的日线行情...")
    start_fmt, end_fmt = pd.to_datetime(start_date).strftime('%Y-%m-%d'), pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    api_call = lambda: bs.query_history_k_data_plus(ticker, "date,open,high,low,close,volume", start_date=start_fmt, end_date=end_date, frequency="d", adjustflag="2")
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
        print(f"  - INFO: 已将 {display_name} 的数据缓存至 {cache_file_path}")
    except Exception as e:
        print(f"  - WARNING: 无法缓存 {display_name} 的数据: {e}")
        
    return df[['open', 'high', 'low', 'close', 'volume']]

def _get_index_data_bs(index_code: str, start_date: str, end_date: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """
    (新增) 优先从 Baostock 获取指数数据。如果失败，则获取成分股并在本地计算等权重指数。
    """
    print(f"  - INFO: 正在为指数 {index_code} 获取数据...")
    
    # 尝试直接获取指数数据
    index_df = _get_ohlcv_data_bs(index_code, start_date, end_date, cache_dir, keyword=f"指数-{index_code}")
    if index_df is not None and not index_df.empty:
        print(f"    - SUCCESS: 已直接从 Baostock 获取到指数 {index_code} 的数据。")
        return index_df

    # 如果直接获取失败，则尝试在本地合成
    print(f"    - WARNNING: 无法直接获取指数 {index_code}。将尝试在本地合成等权重指数...")
    try:
        # 1. 获取指数成分股
        rs = bs.query_hs300_stocks() # 示例：获取沪深300成分股，可扩展
        if rs.error_code != '0':
            print(f"    - ERROR: 无法获取指数 {index_code} 的成分股: {rs.error_msg}")
            return None
        
        constituents = rs.get_data()['code'].tolist()
        print(f"    - INFO: 成功获取 {len(constituents)} 只成分股。开始下载成分股数据...")

        # 2. 批量获取成分股的收盘价
        all_closes = []
        for stock_code in tqdm(constituents, desc=f"下载 {index_code} 成分股", leave=False):
            stock_df = _get_ohlcv_data_bs(stock_code, start_date, end_date, cache_dir, keyword=stock_code)
            if stock_df is not None:
                all_closes.append(stock_df['close'].rename(stock_code))
        
        if not all_closes:
            print("    - ERROR: 未能获取到任何成分股的数据。")
            return None
            
        # 3. 合并所有收盘价，并计算等权重收益率
        all_closes_df = pd.concat(all_closes, axis=1)
        daily_returns = all_closes_df.pct_change().fillna(0)
        
        # 计算每日的等权重平均收益率
        equal_weighted_return = daily_returns.mean(axis=1)
        
        # 4. 根据收益率序列，构建一个虚拟的指数净值序列
        initial_value = 1000
        index_series = initial_value * (1 + equal_weighted_return).cumprod()
        
        # 构建一个符合格式的 DataFrame
        synthetic_index_df = pd.DataFrame({'close': index_series})
        # 补全 OHLCV
        synthetic_index_df['open'] = synthetic_index_df['high'] = synthetic_index_df['low'] = synthetic_index_df['close']
        synthetic_index_df['volume'] = 0
        
        print(f"    - SUCCESS: 已在本地成功合成 {index_code} 的等权重指数。")
        return synthetic_index_df

    except Exception as e:
        print(f"    - ERROR: 在本地合成指数时发生未知错误: {e}")
        return None

def _get_us_stock_data_yf(ticker: str, start_date: str, end_date: str, cache_dir: Path, 
                          max_retries: int = 3, initial_delay: float = 0.5) -> Optional[pd.DataFrame]:
    """
    使用 yfinance 库从 Yahoo Finance 获取美股等海外市场数据。
    集成了缓存逻辑和带指数退避的重试机制。
    
    :param ticker: 股票代码 (例如 "SPY")
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :param cache_dir: 缓存目录
    :param max_retries: 最大重试次数
    :param initial_delay: 初始等待时间（秒），每次重试后会加倍
    :return: 包含 OHLCV 数据的 DataFrame 或 None
    """
    print(f"  - 正在为 {ticker} 获取美股数据...")
    raw_cache_dir = cache_dir / "raw_us_ohlcv"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = raw_cache_dir / f"raw_{ticker}_{start_date}_{end_date}.pkl"

    if cache_file_path.exists():
        print(f"  - 正在从本地缓存加载 {ticker} 的美股数据...")
        try:
            return pd.read_pickle(cache_file_path)
        except Exception as e:
            print(f"  - 警告: 从缓存加载 {ticker} 数据失败 ({e})，将尝试重新下载。")
            # 如果缓存文件损坏，删除并重新下载
            cache_file_path.unlink(missing_ok=True) 

    print(f"  - 正在从 Yahoo Finance 下载 {ticker} 的数据...")
    
    retries = 0
    delay = initial_delay
    df = pd.DataFrame() # 初始化一个空 DataFrame

    while retries < max_retries:
        try:
            # yfinance 的 end 日期是“不包含”的，所以我们需要加一天来确保获取到 end_date 当天的数据
            end_date_inclusive = (pd.to_datetime(end_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            
            # 核心下载调用
            df = yf.download(ticker, start=start_date, end=end_date_inclusive, progress=False)
            
            if df.empty:
                # yfinance 可能返回空 DataFrame，这也被视为下载失败
                raise ValueError(f"yfinance returned an empty DataFrame for {ticker}.")
            
            # 如果成功，跳出重试循环
            break 

        except Exception as e:
            retries += 1
            print(f"  - 警告 [YF]: 下载 {ticker} 数据失败 (尝试 {retries}/{max_retries}). "
                  f"将在 {delay:.2f} 秒后重试. 错误: {e}")
            time.sleep(delay)
            delay *= 2 # 指数退避

    if df.empty:
        print(f"  - 错误 [YF]: 下载 {ticker} 数据在 {max_retries} 次尝试后仍失败。")
        return None
        
    # 格式化数据以匹配项目内部标准
    df.columns = df.columns.str.lower()
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    try:
        df.to_pickle(cache_file_path)
        print(f"  - 信息: 已将 {ticker} 的数据缓存至 {cache_file_path}")
    except Exception as e:
        print(f"  - 警告: 无法缓存 {ticker} 的数据: {e}")
        
    return df

def _get_fama_french_factors(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """(内部函数) 从 Tushare Pro 获取真实的法马-佛伦奇三因子数据。"""
    if pro is None: return None
    print("  - INFO: 正在从 Tushare Pro 加载法马-佛伦奇三因子数据...")
    try:
        start_date_ts = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_date_ts = pd.to_datetime(end_date).strftime('%Y%m%d')
        factors_df = pro.fama_f3_factor(start_date=start_date_ts, end_date=end_date_ts)
        if factors_df.empty:
            print("    - WARNNING: Tushare 未返回任何因子数据。"); return None
        
        factors_df['date'] = pd.to_datetime(factors_df['trade_date'], format='%Y%m%d')
        factors_df.set_index('date', inplace=True)
        for col in ['mktrf', 'smb', 'hml']: factors_df[col] = factors_df[col] / 100
        factors_df['rf'] = 0.02 / 252 # 假设年化无风险利率为 2%
        factors_df = factors_df[['mktrf', 'smb', 'hml', 'rf']].rename(columns={'mktrf': 'mkt_rf'}).sort_index()
        print(f"    - SUCCESS: 成功加载 {len(factors_df)} 条真实的因子数据。")
        return factors_df
    except Exception as e:
        print(f"    - ERROR: 从 Tushare 获取因子数据失败: {e}"); return None

def _get_macroeconomic_data_cn(start_date: str, end_date: str, config: dict) -> Optional[pd.DataFrame]:
    """[TS] (可选) 从 Tushare 获取中国宏观经济指标。"""
    if pro is None: return None
    print("  - 正在尝试从 Tushare 获取中国宏观经济数据...")
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

def _get_market_sentiment_data_ak(start_date: str, end_date: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """
    (新增) 使用 akshare 获取中国市场的情绪指标，如中国波指(iVIX)。
    """
    print("--- 正在获取市场情绪数据 (恐慌指数) ---")
    cache_file = cache_dir / f"market_sentiment_{start_date}_{end_date}.pkl"
    if cache_file.exists():
        print(f"  - 正在从缓存加载市场情绪数据...")
        return pd.read_pickle(cache_file)

    try:
        # akshare 的日期格式是 YYYYMMDD
        start_ak = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_ak = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        # 获取中国波动率指数 (iVIX)
        vix_df = ak.idx_ivix(start_date=start_ak, end_date=end_ak)
        if vix_df.empty:
            print("  - 警告: 未能从 akshare 获取到中国波指(iVIX)数据。")
            return None

        vix_df['date'] = pd.to_datetime(vix_df['日期'])
        vix_df.set_index('date', inplace=True)
        
        # 我们只关心收盘价
        sentiment_df = vix_df[['收盘']].rename(columns={'收盘': 'china_vix'})
        
        sentiment_df.to_pickle(cache_file)
        print(f"--- 市场情绪数据已生成并缓存。")
        return sentiment_df
        
    except Exception as e:
        print(f"  - 错误 [akshare]: 获取市场情绪数据失败: {e}")
        return None

# 公共 API 函数

def get_full_feature_df(ticker: str, config: Dict, keyword: str = None, prediction_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    为单个股票执行完整的特征生成流程，实现了：
    - 智能的、解耦 Tushare 的指数数据获取。
    - 基于多因子模型的 Alpha 标签计算，并能优雅地回退。
    - 灵活的日期窗口策略（训练/预测/增量模式）。
    """
    display_name = keyword if keyword else ticker
    print(f"\n--- 为 {display_name} ({ticker}) 生成特征 ---")
    
    global_settings = config.get('global_settings', {})
    strategy_config = config.get('strategy_config', {})
    stock_info = next((s for s in config.get('stocks_to_process', []) if s['ticker'] == ticker), {})
    run_config = {**global_settings, **strategy_config, **stock_info}

    runtime_start_date = run_config.get('earliest_start_date'); runtime_end_date = run_config.get('end_date')

    if pd.notna(pd.to_datetime(runtime_start_date, errors='coerce')) and pd.notna(pd.to_datetime(runtime_end_date, errors='coerce')):
        start_date_dt, end_date_dt = pd.to_datetime(runtime_start_date), pd.to_datetime(runtime_end_date)
    elif prediction_mode:
        end_date_dt, start_date_dt = pd.Timestamp.now(), pd.Timestamp.now() - pd.DateOffset(days=300)
    else:
        end_date_dt = pd.to_datetime(run_config.get('end_date'))
        lookback_years, earliest_start_date_dt = run_config.get('data_lookback_years', 10), pd.to_datetime(run_config.get('earliest_start_date'))
        if not end_date_dt or not earliest_start_date_dt: 
            print("ERROR: 'end_date' or 'earliest_start_date' not found."); return None
        target_start_date_dt = end_date_dt - pd.DateOffset(years=lookback_years)
        start_date_dt = max(target_start_date_dt, earliest_start_date_dt)
    
    start_date_str, end_date_str = start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d')
    print(f"  - 数据窗口: {start_date_str} to {end_date_str}")
    
    # --- 1. 数据获取 ---
    cache_dir = Path(run_config.get("data_cache_dir", "data_cache"))

    breadth_df = _generate_market_breadth_data(start_date_str, end_date_str, cache_dir)
    sentiment_df = _get_market_sentiment_data_ak(start_date_str, end_date_str, cache_dir)

    external_market_df = None
    # 从 run_config 中读取外部市场列表，如果不存在则默认为空列表
    external_tickers = run_config.get('external_market_tickers', [])
    if not external_tickers:
        print("  - 信息: 未在配置中找到 'external_market_tickers'，将跳过跨市场特征计算。")
    else:
        # 为了保持当前逻辑不变，我们暂时只使用列表中的第一个 ticker
        # 未来可以扩展为循环处理所有 tickers
        ext_ticker = external_tickers[0]
        print(f"  - 信息: 正在根据配置获取外部市场数据: {ext_ticker}")
        
        external_market_df = _get_us_stock_data_yf(ext_ticker, start_date_str, end_date_str, cache_dir)
        if external_market_df is not None:
            # 重命名列以避免与主数据冲突
            external_market_df.rename(columns={'close': 'close_ext', 'open': 'open_ext', 'volume': 'volume_ext'}, inplace=True)
    
    df = _get_ohlcv_data_bs(_get_api_ticker(ticker), start_date_str, end_date_str, cache_dir, keyword=display_name)
    if df is None: return None

    benchmark_df = _get_index_data_bs(run_config.get('benchmark_ticker'), start_date_str, end_date_str, cache_dir)
    industry_df = _get_index_data_bs(run_config.get('industry_etf'), start_date_str, end_date_str, cache_dir)
    
    factors_df = _get_fama_french_factors(start_date_str, end_date_str)
    macro_df = _get_macroeconomic_data_cn(start_date_str, end_date_str)
    
    # --- 2. 基础特征计算 ---
    if breadth_df is not None:
        df = df.join(breadth_df, how='left')
    if macro_df is not None: 
        df = pd.merge_asof(df, macro_df, left_index=True, right_index=True, direction='backward')
    if benchmark_df is not None:
        df = df.join(benchmark_df['close'].rename('benchmark_close'), how='left')
    if industry_df is not None:
        df = df.join(industry_df['close'].rename('industry_close'), how='left')
    if sentiment_df is not None:
        df = df.join(sentiment_df, how='left')
        df['china_vix'] = df['china_vix'].ffill()
        
    run_config_with_api = {**run_config, 'tushare_pro_instance': pro, 'ticker': ticker}

    extra_data_for_calc = {'external_market_df': external_market_df}
    df = feature_calculators.run_all_feature_calculators(df, run_config_with_api, **extra_data_for_calc)
    
    # --- 3. 特征后处理 ---
    # 运行通用的后处理器 (平稳化, 特征选择等)
    df = run_all_feature_postprocessors(df, run_config, factors_df=factors_df)
    df.columns = df.columns.str.lower()
    
    # 4. --- 数据清洗和校验 ---
    ffill_limit = run_config.get("ffill_limit", 5)
    df.ffill(inplace=True, limit=ffill_limit)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    label_col = run_config.get('label_column', 'label_alpha')
    if label_col in df.columns: df.dropna(subset=[label_col], inplace=True)
    
    feature_cols = [col for col in df.columns if col != label_col and not col.startswith('future_')]
    df.dropna(subset=feature_cols, inplace=True)
    
    if df.empty: 
        print(f"警告: {display_name} 的 DataFrame 在处理后为空。"); return None
    
    validator = data_contracts.DataValidator(run_config)
    if not validator.validate_schema(df): 
        print(f"错误: {display_name} 的最终数据校验失败。"); return None
        
    print(f"--- 成功为 {display_name} 生成特征. 维度: {df.shape} ---")
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