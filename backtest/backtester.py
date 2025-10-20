# backtest/backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class VectorizedBacktester:
    """
    一个基于 Pandas 和 NumPy 的向量化回测引擎。
    它接收模型在样本外（OOF）的预测，并根据预设的交易规则模拟策略表现。
    """
    def __init__(self, oof_predictions: pd.DataFrame, config: dict):
        """
        初始化回测器。
        :param oof_predictions: DataFrame，必须包含 'y_pred' (模型预测) 和 'y_true' (未来真实收益) 列。
        :param config: 完整的项目配置字典。
        """
        if not all(col in oof_predictions.columns for col in ['y_pred', 'y_true']):
            raise ValueError("输入的 DataFrame 必须包含 'y_pred' 和 'y_true' 列。")
        
        self.predictions = oof_predictions.sort_index()
        self.config = config.get('backtest_settings', {})
        self.results = {}
        self.ticker = oof_predictions['ticker'].iloc[0] if 'ticker' in oof_predictions.columns else 'UNKNOWN'
        
        # 从配置加载回测参数
        self.signal_threshold = self.config.get('signal_threshold', 0.005)
        self.transaction_cost = self.config.get('transaction_cost', 0.001) # 0.1% a round trip

    def _generate_signals(self):
        """根据模型预测生成交易信号 (1: 做多, -1: 做空, 0: 空仓)。"""
        df = self.predictions.copy()
        df['signal'] = np.where(df['y_pred'] > self.signal_threshold, 1, 0)
        df['signal'] = np.where(df['y_pred'] < -self.signal_threshold, -1, df['signal'])
        self.predictions['signal'] = df['signal']

    def _simulate_portfolio(self):
        """模拟投资组合表现，计算收益和资金曲线。"""
        # 关键: 使用昨天的信号(shift(1))来决定今天的收益，这是为了避免前视偏差！
        # y_true 是未来 N 天的收益，所以 signal.shift(1) * y_true 代表了
        # 根据 T-1 的信号，在 T 时刻开仓，持有 N 天后的收益。
        self.predictions['strategy_return'] = self.predictions['signal'].shift(1) * self.predictions['y_true']

        # 计算换手率和交易成本
        self.predictions['turnover'] = self.predictions['signal'].diff().abs()
        self.predictions['costs'] = self.predictions['turnover'] * self.transaction_cost
        
        # 计算净收益
        self.predictions['net_return'] = self.predictions['strategy_return'] - self.predictions['costs']
        
        # 计算资金曲线
        self.predictions['equity_curve'] = (1 + self.predictions['net_return']).cumprod()
        self.predictions.fillna(0, inplace=True)

    def _calculate_performance_metrics(self):
        """计算关键的策略表现指标 (KPIs)。"""
        equity_curve = self.predictions['equity_curve']
        net_returns = self.predictions['net_return']
        
        if equity_curve.empty or net_returns.empty:
            return

        # 1. 累计收益
        total_return = equity_curve.iloc[-1] - 1

        # 2. 年化收益
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0

        # 3. 年化波动率
        annualized_volatility = net_returns.std() * np.sqrt(252) # 假设每日调仓

        # 4. 夏普比率
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

        # 5. 最大回撤
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 6. 胜率
        win_rate = (net_returns > 0).sum() / (net_returns != 0).sum() if (net_returns != 0).sum() > 0 else 0
        
        self.results = {
            'Ticker': self.ticker,
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Annualized Volatility': f"{annualized_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
        }

    def run(self):
        """完整执行回测流程。"""
        self._generate_signals()
        self._simulate_portfolio()
        self._calculate_performance_metrics()

    def get_results(self) -> dict:
        """返回计算出的性能指标字典。"""
        return self.results

    def plot_results(self):
        """绘制资金曲线和回撤图。"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # 资金曲线
        self.predictions['equity_curve'].plot(ax=ax1, label='Strategy Equity Curve')
        ax1.set_title(f'Strategy Performance for {self.ticker}', fontsize=16)
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True)
        ax1.legend()
        
        # 回撤曲线
        drawdown = (self.predictions['equity_curve'] - self.predictions['equity_curve'].cummax()) / self.predictions['equity_curve'].cummax()
        drawdown.plot(ax=ax2, kind='area', color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()