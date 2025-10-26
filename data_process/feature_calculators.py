# 文件路径: data_process/feature_calculators.py
'''
自定义特征规则
'''
import numpy as np
import pandas as pd
import pandas_ta as ta
import statsmodels.api as sm
from abc import ABC, abstractmethod
from statsmodels.regression.rolling import RollingOLS

# --- 1. 定义所有特征计算器都必须遵循的标准接口 ---
class FeatureCalculator(ABC):
    """
    所有具体特征计算器的抽象基类。
    每个子类负责计算一类特定的特征。
    """
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        接收一个DataFrame，计算特征，并将结果作为新列添加后返回。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """返回计算器的名称，用于日志记录。"""
        pass

# --- 2. 实现具体的特征计算器 ---

class CandlestickPatternCalculator(FeatureCalculator):
    """
    根据配置计算指定的K线模式特征。
    """
    @property
    def name(self) -> str:
        return "Candlestick Patterns"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(df.columns):
            print(f"WARNNING: [{self.name}] DataFrame is missing required columns. Skipping.")
            return df

        patterns_to_calc = self.config.get('candlestick_patterns', [])
        if not patterns_to_calc:
            print(f"  - [Calculating Features] INFO: No candlestick patterns specified in config. Skipping {self.name}.")
            return df

        print(f"  - [Calculating Features] Running: {self.name}...")
        # pandas-ta 会自动将生成的列添加到 df 中
        df.ta.cdl_pattern(name=patterns_to_calc, append=True)
        return df

class TechnicalIndicatorCalculator(FeatureCalculator):
    """
    根据配置动态计算技术分析指标。
    """
    @property
    def name(self) -> str:
        return "Technical Indicators"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        indicators_to_calc = self.config.get('technical_indicators', [])
        if not indicators_to_calc:
            print(f"  - [Calculating Features] INFO: No technical indicators specified in config. Skipping {self.name}.")
            return df
            
        print(f"  - [Calculating Features] Running: {self.name}...")
        for indicator in indicators_to_calc:
            name = indicator.get('name')
            params = indicator.get('params', {})
            if not name:
                continue
                
            try:
                # 使用 getattr 动态调用 pandas-ta 的方法
                # 例如, df.ta.ema(length=10, append=True)
                getattr(df.ta, name)(**params, append=True)
                print(f"    - Calculated: {name} with params {params}")
            except AttributeError:
                print(f"    - WARNNING: Indicator '{name}' is not a valid pandas-ta function.")
            except Exception as e:
                print(f"    - ERROR: Failed to calculate indicator '{name}': {e}")
                
        df.columns = df.columns.str.lower()
        return df

class CalendarFeatureCalculator(FeatureCalculator):
    """
    计算日历相关的特征。
    """
    @property
    def name(self) -> str:
        return "Calendar Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [Calculating Features] Running: {self.name}...")
        df.index = pd.to_datetime(df.index)
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        return df
    
class StatisticalFeatureCalculator(FeatureCalculator):
    """
    统计特征计算器: 提取滑动窗口内的统计特征 (滑动均值, 标准差, 偏度, 峰度, 与收盘价, 成交量等)
    """
    @property
    def name(self):
        return "Statistical Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [Calculating Features] Running: {self.name}...")
        window_sizes = self.config.get('stat_windows', [5, 10, 20])
        for w in window_sizes:
            df[f'return_{w}'] = df['close'].pct_change(w)
            df[f'volatility_{w}'] = df['close'].pct_change().rolling(w).std()
            df[f'zscore_{w}'] = (df['close'] - df['close'].rolling(w).mean()) / df['close'].rolling(w).std()
        return df

class PriceStructureCalculator(FeatureCalculator):
    """
    价格结构特征计算器: 提取价格行为中的结构性信息: 
    高低价差（high - low）
    实体长度（|close - open|）
    上下影线长度
    """
    @property
    def name(self):
        return "Price Structure Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [Calculating Features] Running: {self.name}...")
        df['body'] = (df['close'] - df['open']).abs()
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['range'].replace(0, 1e-8)
        return df

class VolumeFeatureCalculator(FeatureCalculator):
    """
    成交量行为特征: 补充量价配合类特征，比如：
    成交量变化率
    价格变化与成交量变化的相关系数
    OBV, VWAP
    """
    @property
    def name(self):
        return "Volume Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [Calculating Features] Running: {self.name}...")
        if 'volume' not in df.columns:
            print(f"    - WARNNING: Missing 'volume' column. Skipping {self.name}.")
            return df

        df['volume_change'] = df['volume'].pct_change()
        df['price_volume_corr'] = (
            df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
        )
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
class TrendRegimeFeatureCalculator(FeatureCalculator):
    """
    趋势与均衡特征: 使用线性回归/多项式回归拟合价格趋势斜率。
    可判断当前处于上升, 下降还是震荡阶段。
    """
    @property
    def name(self):
        return "Trend Regime Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print(f"  - [Calculating Features] Running: {self.name} (Vectorized)...")
        
        window = self.config.get('strategy_config', {}).get('trend_window', 30)
        
        # 创建一个代表时间流逝的序列 [0, 1, 2, ...]，用于线性回归的 X 轴
        df['time_index'] = np.arange(len(df))
        
        # 1. 计算 X 的滚动方差 Var(X)
        x_for_var = np.arange(window)
        var_x = np.var(x_for_var, ddof=0) 
        
        # 如果方差为0（例如 window=1），则返回0以避免除以零错误
        if var_x == 0:
            df['trend_slope'] = 0.0
        else:
            # 2. 计算 Y ('close') 与 X ('time_index') 的滚动协方差 Cov(X, Y)
            rolling_cov = df['close'].rolling(window=window).cov(df['time_index'])
            
            # 3. 计算斜率
            df['trend_slope'] = rolling_cov / var_x
        
        # 清理计算过程中使用的辅助列
        df.drop(columns=['time_index'], inplace=True)
        return df

class MomentumVolatilityCalculator(FeatureCalculator):
    """
    计算动量、波动率与风险调整收益类因子。
    """
    @property
    def name(self) -> str:
        return "Momentum & Volatility Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print(f"  - [Calculating Features] Running: {self.name}...")
        
        # 从配置中获取要计算的窗口期（以月为单位）
        periods_in_month = 21 # 平均每月约21个交易日
        windows = self.config.get('strategy_config', {}).get('momentum_windows', [3, 6, 12]) # 默认计算3,6,12个月的因子
        
        daily_ret = df['close'].pct_change().fillna(0)
        
        for m in windows:
            period = m * periods_in_month
            if len(df) < period: continue

            # 1. 动量因子
            df[f'momentum_{m}m'] = df['close'].pct_change(periods=period)
            
            # 2. 波动率因子
            df[f'volatility_{m}m'] = daily_ret.rolling(window=period).std() * np.sqrt(252)

            # 3. 风险调整后动量 (夏普比率)
            rolling_mean = daily_ret.rolling(window=period).mean()
            rolling_std = daily_ret.rolling(window=period).std()
            df[f'sharpe_{m}m'] = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252)

            # 4. 下行波动率
            neg_ret = daily_ret.copy()
            neg_ret[neg_ret > 0] = 0
            df[f'downside_vol_{m}m'] = neg_ret.rolling(window=period).std() * np.sqrt(252)
            
        return df

class FundamentalCalculator(FeatureCalculator):
    """
    从 Tushare Pro 获取基本面因子，如 PE, PB, ROE, 市值等。
    此计算器依赖于有效的 Tushare Token 和足够的积分。
    """
    @property
    def name(self) -> str:
        return "Fundamental Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print(f"  - [Calculating Features] Running: {self.name}...")

        # 检查 Tushare API 是否可用
        pro = self.config.get('tushare_pro_instance')
        if pro is None:
            if self.config.get('global_settings', {}).get('verbose', True):
                print(f"    - WARNNING: Tushare Pro API not available. Skipping {self.name}.")
            return df
        
        # 将 Baostock 代码转换为 Tushare 代码
        ticker_bs = self.config.get('ticker')
        if ticker_bs is None: return df
        ticker_ts = f"{ticker_bs.split('.')[1]}.{ticker_bs.split('.')[0].upper()}"
        
        start_date = df.index.min().strftime('%Y%m%d')
        end_date = df.index.max().strftime('%Y%m%d')

        try:
            # 调用 Tushare 的 'daily_basic' 接口
            funda_df = pro.daily_basic(
                ts_code=ticker_ts, 
                start_date=start_date, 
                end_date=end_date,
                fields='trade_date,turnover_rate,pe_ttm,pb,total_mv'
            )
            if funda_df.empty: return df

            funda_df['date'] = pd.to_datetime(funda_df['trade_date'], format='%Y%m%d')
            funda_df.set_index('date', inplace=True)
            
            # 选择并重命名因子
            funda_df.rename(columns={
                'turnover_rate': 'turnover_rate',
                'pe_ttm': 'pe_ttm',
                'pb': 'pb_ratio',
                'total_mv': 'log_market_cap' # 我们将获取总市值
            }, inplace=True)
            
            # 对市值取对数，使其分布更接近正态
            funda_df['log_market_cap'] = np.log(funda_df['log_market_cap'])
            
            # 合并到主 DataFrame
            df = df.join(funda_df[['turnover_rate', 'pe_ttm', 'pb_ratio', 'log_market_cap']], how='left')
            
            if self.config.get('global_settings', {}).get('verbose', True):
                print(f"    - SUCCESS: Successfully merged fundamental features.")

        except Exception as e:
            if self.config.get('global_settings', {}).get('verbose', True):
                print(f"    - ERROR: Failed to get fundamental data from Tushare: {e}")
        
        return df

class RelativeStrengthCalculator(FeatureCalculator):
    """
    计算个股相对于基准/行业的相对强度和风险暴露(Beta)。
    """
    @property
    def name(self) -> str:
        return "Relative Strength & Beta Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print(f"  - [Calculating Features] Running: {self.name}...")
        
        # 确保基准和行业数据存在
        if 'benchmark_close' not in df.columns or 'industry_close' not in df.columns:
            print(f"    - WARNNING: Missing 'benchmark_close' or 'industry_close' column. Skipping {self.name}.")
            return df

        window_sizes = self.config.get('stat_windows', [5, 10, 20])

        # 1. 计算相对价格比率
        # 用 fillna(method='ffill') 填充周末或节假日可能出现的缺失
        df['price_vs_benchmark'] = (df['close'] / df['benchmark_close'].ffill()).fillna(method='ffill')
        df['price_vs_industry'] = (df['close'] / df['industry_close'].ffill()).fillna(method='ffill')

        # 2. 计算相对强度的动量 (RSI, Momentum)
        # 这个指标衡量的是“超额表现”的趋势
        for w in window_sizes:
            df[f'relative_strength_momentum_bench_{w}d'] = df['price_vs_benchmark'].pct_change(periods=w)
            df[f'relative_strength_momentum_ind_{w}d'] = df['price_vs_industry'].pct_change(periods=w)

        # 3. 计算滚动的 Beta
        # Beta衡量了股票相对于大盘的波动性风险。高Beta股在牛市可能涨得更多，熊市也跌得更狠。
        stock_ret = df['close'].pct_change().fillna(0)
        bench_ret = df['benchmark_close'].pct_change().fillna(0)
        
        # 使用过去约半年的数据计算 Beta
        rolling_window = 120 
        
        # 准备 OLS 输入. X 是自变量(市场收益), Y 是因变量(个股收益)
        # 我们需要处理好索引对齐问题
        Y_reg = stock_ret
        X_reg = sm.add_constant(bench_ret) # add_constant 增加截距项
        
        if len(df) > rolling_window:
            rols = RollingOLS(endog=Y_reg, exog=X_reg, window=rolling_window, min_nobs=int(rolling_window * 0.8))
            rres = rols.fit()
            # rres.params 中包含 'const' (截距, 即Alpha) 和 'benchmark_close' (斜率, 即Beta)
            df['beta_to_benchmark'] = rres.params['benchmark_close']
        else:
            df['beta_to_benchmark'] = np.nan

        return df

class MarketRegimeCalculator(FeatureCalculator):
    """
    根据基准指数的波动率来定义市场的宏观状态（例如，高波动 vs 低波动）。
    """
    @property
    def name(self) -> str:
        return "Market Regime Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print(f"  - [Calculating Features] Running: {self.name}...")

        if 'benchmark_close' not in df.columns:
            print(f"    - WARNNING: Missing 'benchmark_close' column. Skipping {self.name}.")
            return df
        
        # 1. 计算基准指数的短期和长期波动率
        bench_ret = df['benchmark_close'].pct_change()
        df['benchmark_vol_20d'] = bench_ret.rolling(20).std() * np.sqrt(252) # 短期 (月度)
        df['benchmark_vol_60d'] = bench_ret.rolling(60).std() * np.sqrt(252) # 中期 (季度)

        # 2. 定义市场状态
        # 方法一: 是否处于高波动状态 (与长期中位数相比)
        long_term_median_vol = df['benchmark_vol_60d'].rolling(252).median()
        df['regime_is_high_vol'] = (df['benchmark_vol_20d'] > long_term_median_vol).astype(int)

        # 方法二: 波动率的趋势 (短期波动率是否在上升)
        df['regime_vol_trend_up'] = (df['benchmark_vol_20d'] > df['benchmark_vol_60d']).astype(int)

        return df

class AdvancedRiskCalculator(FeatureCalculator):
    """
    计算更高级的风险指标，如最大回撤 (Max Drawdown)。
    """
    @property
    def name(self) -> str:
        return "Advanced Risk Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.config.get('global_settings', {}).get('verbose', True):
            print(f"  - [Calculating Features] Running: {self.name}...")

        window_sizes = self.config.get('stat_windows', [20, 60, 120]) # 月度、季度、半年度

        for w in window_sizes:
            # 1. 计算滚动窗口内的最高价
            rolling_max = df['close'].rolling(window=w, min_periods=1).max()
            
            # 2. 计算当前价格相对于滚动最高价的回撤
            # 公式: (当前价格 / 峰值) - 1。结果是一个负数或零。
            daily_drawdown = (df['close'] / rolling_max) - 1.0
            
            # 3. 找到滚动窗口内的最大回撤
            # 我们在一个新的滚动窗口内寻找之前计算的 daily_drawdown 的最小值
            df[f'max_drawdown_{w}d'] = daily_drawdown.rolling(window=w, min_periods=1).min()

        return df

class CrossoverSignalCalculator(FeatureCalculator):
    """
    计算技术指标的交叉信号，如金叉和死叉。
    这个计算器必须在 TechnicalIndicatorCalculator 之后运行。
    """
    @property
    def name(self) -> str:
        return "交叉信号特征 (金叉/死叉)"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        crossover_rules = self.config.get('crossover_signals', [])
        if not crossover_rules:
            print(f"  - [计算特征] 信息: 未在配置中指定交叉信号规则，跳过 {self.name}。")
            return df
        
        print(f"  - [计算特征] 正在运行: {self.name}...")
        
        for rule in crossover_rules:
            fast_col = rule.get('fast')
            slow_col = rule.get('slow')
            rule_name = rule.get('name')

            if not all([fast_col, slow_col, rule_name]):
                print(f"    - 警告: 交叉规则配置不完整，缺少 fast, slow 或 name。跳过此规则。")
                continue

            # 确保计算金叉/死叉所依赖的均线已经存在于DataFrame中
            if fast_col not in df.columns or slow_col not in df.columns:
                print(f"    - 警告: 无法计算 '{rule_name}'，因为依赖的列 '{fast_col}' 或 '{slow_col}' 不存在。")
                print(f"    -      请确保它们已在 'technical_indicators' 配置中计算。")
                continue
            
            # 计算信号
            # 金叉条件: t-1 时 fast <= slow, 且 t 时 fast > slow
            # 死叉条件: t-1 时 fast >= slow, 且 t 时 fast < slow
            signal = np.zeros(len(df))
            
            # 使用 .iloc 避免 SettingWithCopyWarning
            golden_cross_mask = (df[fast_col].iloc[1:] > df[slow_col].iloc[1:]) & (df[fast_col].iloc[:-1].values <= df[slow_col].iloc[:-1].values)
            death_cross_mask = (df[fast_col].iloc[1:] < df[slow_col].iloc[1:]) & (df[fast_col].iloc[:-1].values >= df[slow_col].iloc[:-1].values)
            
            # np.where(condition, x, y)
            # 将信号设置在交叉发生的当天
            signal[1:] = np.where(golden_cross_mask, 1, np.where(death_cross_mask, -1, 0))

            df[f'signal_{rule_name}'] = signal.astype(int)
            print(f"    - 已计算: {rule_name} (金叉/死叉信号)")

        return df

class CandleQuantCalculator(FeatureCalculator):
    """
    对K线本身进行量化解构，提取更多维度的特征。
    """
    @property
    def name(self) -> str:
        return "K线量化解构特征"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [计算特征] 正在运行: {self.name}...")

        # 1. K线在布林带中的位置
        # 这个特征衡量了收盘价相对于近期波动区间的极端程度。
        # 值接近1代表接近上轨（超买），接近0代表接近下轨（超卖）。
        
        try:
            # 1. 先执行计算
            # 使用 append=True 会直接将计算结果添加到 df 中，更简洁
            df.ta.bbands(length=20, append=True)
            
            # --- 修改开始: 动态查找列名 ---
            # 2. 动态查找上轨(BBU)和下轨(BBL)的列名
            bbu_col = next((col for col in df.columns if col.startswith('BBU_')), None)
            bbl_col = next((col for col in df.columns if col.startswith('BBL_')), None)
            
            if bbu_col and bbl_col:
                # 3. 如果找到了，就执行计算
                bandwidth = (df[bbu_col] - df[bbl_col]).replace(0, 1e-9)
                df['candle_pos_in_bbands'] = (df['close'] - df[bbl_col]) / bandwidth
                print(f"    - 已计算: K线在布林带中的位置")
            else:
                print(f"    - 警告: 未能动态找到布林带的上轨或下轨列。跳过此特征。")
            
            # 4. 清理所有由 bbands 生成的辅助列
            bbands_cols_to_drop = [col for col in df.columns if col.startswith(('BBU_', 'BBM_', 'BBL_', 'BBB_', 'BBP_'))]
            df.drop(columns=bbands_cols_to_drop, inplace=True, errors='ignore')
            # --- 修改结束 ---
            
        except Exception as e:
            print(f"    - 警告: 计算布林带特征时发生意外错误: {e}")

        # 2. 波动率收缩/扩张
        # 衡量今天的振幅（high-low）相对于过去一段时间平均振幅的大小。
        # 正值表示波动放大，负值表示波动收缩。这常用于识别突破前的“蓄力”状态。
        if 'range' not in df.columns: # 依赖 PriceStructureCalculator
             df['range'] = df['high'] - df['low']
        
        avg_range = df['range'].rolling(10).mean().replace(0, 1e-9)
        df['volatility_expansion_ratio'] = (df['range'] / avg_range) - 1.0
        print(f"    - 已计算: 波动率扩张比率")
        
        # 3. 跳空缺口大小
        # 衡量开盘价相对于昨日收盘价的跳空幅度。
        # 正值是向上跳空，负值是向下跳空。强大的跳空通常预示着趋势的开始或延续。
        gap = df['open'] - df['close'].shift(1)
        df['gap_size_ratio'] = gap / df['close'].shift(1)
        print(f"    - 已计算: 跳空缺口大小")

        return df

class IntermarketCorrelationCalculator(FeatureCalculator):
    """
    (已重构) 计算与一个或多个外部市场的滚动相关性。
    """
    @property
    def name(self) -> str:
        return "跨市场关联特征"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        external_market_df = kwargs.get('external_market_df')
        if external_market_df is None or external_market_df.empty:
            print(f"  - [计算特征] 信息: 未提供外部市场数据，跳过 {self.name}。")
            return df
            
        print(f"  - [计算特征] 正在运行: {self.name}...")
        
        stock_ret = df['close'].pct_change()
        
        # --- (核心修改) 循环处理所有外部市场的 'close' 列 ---
        # 我们查找所有以 'close_' 开头的列，例如 'close_SPY', 'close_QQQ'
        close_cols_external = [col for col in external_market_df.columns if col.startswith('close_')]

        if not close_cols_external:
            print(f"    - WARNNING: 在提供的 external_market_df 中未找到任何 'close_' 开头的列。跳过。")
            return df

        for ext_close_col in close_cols_external:
            # 从列名中提取 Ticker, 例如 'close_SPY' -> 'SPY'
            ext_ticker = ext_close_col.replace('close_', '')
            
            # 创建一个临时的 DataFrame 用于计算，只包含当前外部市场的数据
            df_merged = pd.DataFrame({'stock_ret': stock_ret}).join(external_market_df[[ext_close_col]])
            
            # 计算该外部市场的日收益率
            df_merged['external_ret'] = df_merged[ext_close_col].pct_change()
            
            # 计算滚动相关系数，并在新特征的列名中包含 Ticker
            corr_col_name = f'corr_with_{ext_ticker}_60d'
            df[corr_col_name] = df_merged['stock_ret'].rolling(60).corr(df_merged['external_ret'])
            
            print(f"    - 已计算: 与 {ext_ticker} 的60日滚动相关性 (列名: {corr_col_name})")
        
        return df

class LiquidityAndFlowCalculator(FeatureCalculator):
    """
    计算流动性与资金流冲击相关的特征。
    """
    @property
    def name(self) -> str:
        return "流动性与资金流特征"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [计算特征] 正在运行: {self.name}...")

        # 1. 计算 Amihud 非流动性指标
        daily_return_abs = df['close'].pct_change().abs()
        # 使用成交额 (volume * close) 更准确，避免低价股的成交量失真
        turnover = df['volume'] * df['close']
        
        # 加上一个极小值防止除以零
        df['amihud_illiquidity'] = daily_return_abs / (turnover + 1e-9)
        # 取30日移动平均使其更平滑，观察趋势
        df['amihud_illiquidity_30d_ma'] = df['amihud_illiquidity'].rolling(30).mean()
        print(f"    - 已计算: Amihud 非流动性指标")

        # 2. 计算成交量 Z-Score
        vol_mean_60d = df['volume'].rolling(60).mean()
        vol_std_60d = df['volume'].rolling(60).std().replace(0, 1e-9) # 防止除以零
        df['volume_zscore_60d'] = (df['volume'] - vol_mean_60d) / vol_std_60d
        print(f"    - 已计算: 60日成交量 Z-Score")

        return df
    
class ContextualFeatureCalculator(FeatureCalculator):
    """
    情景感知与交叉特征计算器。
    负责定义市场的宏观状态（牛/熊市，高/低波动），并创建交叉特征。
    这个计算器应该在大多数基础技术指标计算之后运行。
    """
    @property
    def name(self) -> str:
        return "Contextual & Cross-Sectional Features"

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"  - [Calculating Features] Running: {self.name}...")

        if 'benchmark_close' not in df.columns:
            print(f"    - WARNNING: Missing 'benchmark_close' column. Skipping {self.name}.")
            return df
        
        # --- 1. 定义宏观市场状态 (Regimes) ---
        
        # a) 市场趋势状态 (牛市 vs 熊市)
        # 使用长期均线来定义，例如 200 日线
        df['regime_is_uptrend'] = (df['benchmark_close'] > df['benchmark_close'].rolling(200).mean()).astype(int)
        print("    - Calculated: Market Trend Regime (Uptrend/Downtrend)")

        # b) 市场波动率状态 (高波 vs 低波)
        # 这个在 MarketRegimeCalculator 中已经计算了，我们在这里可以直接使用或重新计算以保证独立性
        if 'regime_is_high_vol' not in df.columns:
            bench_ret = df['benchmark_close'].pct_change()
            vol_60d = bench_ret.rolling(60).std()
            long_term_median_vol = vol_60d.rolling(252).median()
            df['regime_is_high_vol'] = (vol_60d > long_term_median_vol).astype(int)
            print("    - Calculated: Market Volatility Regime (High/Low Vol)")

        # --- 2. 创建交叉特征 ---
        # 我们的假设是：某些指标在不同的市场状态下，含义是不同的。
        
        # a) 将 RSI 与趋势状态交叉
        if 'rsi_14' in df.columns: # 假设 RSI 列名是 pandas-ta 默认生成的
            df['rsi_x_uptrend'] = df['rsi_14'] * df['regime_is_uptrend']
            print("    - Created Cross-Feature: RSI x Trend Regime")
            
        # b) 将动量指标 (以 ROC 为例) 与波动率状态交叉
        if 'roc_10' in df.columns:
            df['roc_x_high_vol'] = df['roc_10'] * df['regime_is_high_vol']
            print("    - Created Cross-Feature: ROC x Volatility Regime")

        # c) 将成交量变化与趋势状态交叉
        if 'volume_change' in df.columns:
            df['vol_change_x_uptrend'] = df['volume_change'] * df['regime_is_uptrend']
            print("    - Created Cross-Feature: Volume Change x Trend Regime")
            
        return df

# 创建注册表和运行器
# 注意: ALL_CALCULATORS 列表中的顺序为数据处理顺序。
# 1. 基础指标 (TechnicalIndicatorCalculator, PriceStructureCalculator) 必须先运行。
# 2. 依赖于基础指标的计算器 (CrossoverSignalCalculator, CandleQuantCalculator) 必须在后面。
# 3. 依赖于外部数据 (如 benchmark_close) 的计算器 (RelativeStrengthCalculator, ContextualFeatureCalculator)
#    应该在数据合并之后，但通常没有严格的先后顺序。
# 4. ContextualFeatureCalculator (交叉特征) 应该在它所依赖的基础特征计算完毕后运行。
ALL_CALCULATORS = [
    TechnicalIndicatorCalculator,
    CalendarFeatureCalculator,
    CandlestickPatternCalculator,
    StatisticalFeatureCalculator,
    PriceStructureCalculator,
    VolumeFeatureCalculator,
    TrendRegimeFeatureCalculator,
    MomentumVolatilityCalculator,
    FundamentalCalculator,
    RelativeStrengthCalculator,
    MarketRegimeCalculator,
    AdvancedRiskCalculator,
    CrossoverSignalCalculator, 
    CandleQuantCalculator,
    IntermarketCorrelationCalculator,
    LiquidityAndFlowCalculator,
    ContextualFeatureCalculator,
]

def run_all_feature_calculators(df: pd.DataFrame, config: dict, **kwargs) -> pd.DataFrame:
    """
    实例化并按顺序运行所有已注册的特征计算器。
    """
    print("INFO: 开始特征计算流水线...")
    df_copy = df.copy()
    for calculator_class in ALL_CALCULATORS:
        calculator = calculator_class(config)
        # 将 kwargs 传递给每个 calculate 方法
        df_copy = calculator.calculate(df_copy, **kwargs) 
    print("INFO: 特征计算流水线结束。")
    return df_copy