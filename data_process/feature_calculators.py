# 文件路径: data_process/feature_calculators.py
'''
自定义特征规则
'''

import pandas as pd
import pandas_ta as ta
from abc import ABC, abstractmethod

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

# --- 3. 创建一个注册表和运行器 ---
# 在这里注册所有您想要运行的计算器
ALL_CALCULATORS = [
    TechnicalIndicatorCalculator,
    CalendarFeatureCalculator,
    CandlestickPatternCalculator,
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