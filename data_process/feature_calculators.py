# 文件路径: data_process/feature_calculators.py
'''
自定义特征规则
'''
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from data_process.get_data import pro as tushare_pro_instance

# --- 1. 定义所有特征计算器都必须遵循的标准接口 ---
class FeatureCalculator(ABC):
    """
    所有具体特征计算器的抽象基类。
    每个子类负责计算一类特定的特征。
    """
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
# 创建注册表和运行器
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
]

def run_all_feature_calculators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    实例化并按顺序运行所有已注册的特征计算器。
    """
    print("INFO: Starting feature calculation pipeline...")
    df_copy = df.copy()
    for calculator_class in ALL_CALCULATORS:
        calculator = calculator_class(config)
        df_copy = calculator.calculate(df_copy)
    print("INFO: Feature calculation pipeline finished.")
    return df_copy