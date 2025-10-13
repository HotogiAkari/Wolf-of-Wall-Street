# 文件路径: data_process/feature_calculators.py
'''
自定义特征规则
'''
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

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
        print(f"  - [Calculating Features] Running: {self.name}...")
        window = self.config.get('trend_window', 20)
        slopes = []
        x = np.arange(window).reshape(-1, 1)
        model = LinearRegression()
        for i in range(len(df)):
            if i < window:
                slopes.append(0)
            else:
                y = df['close'].iloc[i-window:i].values.reshape(-1, 1)
                model.fit(x, y)
                slopes.append(model.coef_[0][0])
        df['trend_slope'] = slopes
        return df
    
class TargetFeatureCalculator(FeatureCalculator):
    """
    目标工程特征: 提前计算未来收益, 未来方向（可做监督标签或辅助特征）
    """
    @property
    def name(self):
        return "Target/Forecast Features"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon = self.config.get('target_horizon', [1, 5, 10])
        for h in horizon:
            df[f'future_return_{h}'] = df['close'].shift(-h) / df['close'] - 1
        return df


# --- 3. 创建一个注册表和运行器 ---
# 在这里注册所有想要运行的计算器
ALL_CALCULATORS = [
    TechnicalIndicatorCalculator,
    CalendarFeatureCalculator,
    CandlestickPatternCalculator,
    StatisticalFeatureCalculator,
    PriceStructureCalculator,
    VolumeFeatureCalculator,
    TrendRegimeFeatureCalculator,
    TargetFeatureCalculator,
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