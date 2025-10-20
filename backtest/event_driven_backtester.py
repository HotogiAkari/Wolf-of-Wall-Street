# backtest/event_driven_backtester.py

import numpy as np
import pandas as pd
import backtrader as bt

class SignalData(bt.feeds.PandasData):
    """
    自定义数据源，使 backtrader 能够读取我们的 OOF 预测信号。
    """
    lines = ('signal',)  # 声明我们要添加 'signal' 这一列
    params = (
        ('signal', -1), # -1 表示自动检测列的位置
    )

class ModelSignalStrategy(bt.Strategy):
    """
    一个基于外部信号进行交易的 backtrader 策略。
    """
    params = (
        ('holding_period', 45), # 默认持仓周期，应与 labeling_horizon 一致
        ('sizing_pct', 0.95),   # 每次交易使用的现金比例
    )

    def __init__(self):
        self.signal = self.datas[0].signal
        self.order = None
        self.entry_bar = 0

    def log(self, txt, dt=None):
        """日志记录函数"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        # 如果有挂单，不进行任何操作
        if self.order:
            return

        # 检查是否在持仓
        if not self.position:
            # 如果没有持仓，且信号为做多
            if self.signal[0] > 0:
                self.log(f'BUY CREATE, Signal: {self.signal[0]:.2f}')
                size_to_buy = self.broker.get_cash() * self.p.sizing_pct / self.datas[0].close[0]
                self.order = self.buy(size=size_to_buy)
                self.entry_bar = len(self)
        else:
            # 如果在持仓，检查是否达到持仓周期
            if len(self) >= (self.entry_bar + self.p.holding_period):
                self.log(f'CLOSE POSITION (Holding Period Reached)')
                self.order = self.close()

def run_backtrader_backtest(
    daily_ohlcv: pd.DataFrame, 
    oof_predictions: pd.DataFrame, 
    config: dict,
    plot=True
):
    """
    运行一次完整的 backtrader 事件驱动回测。
    
    :param daily_ohlcv: 包含 'open', 'high', 'low', 'close', 'volume' 的日线行情数据。
    :param oof_predictions: 包含 'y_pred' (模型预测) 的 DataFrame。
    :param config: 项目配置字典。
    :param plot: 是否绘制结果图表。
    """
    backtest_cfg = config.get('backtest_settings', {})
    
    # 1. 准备数据
    # 将模型预测转换为交易信号 (1, -1, 0)
    signal_threshold = backtest_cfg.get('signal_threshold', 0.005)
    signals = np.where(oof_predictions['y_pred'] > signal_threshold, 1, 0)
    signals = np.where(oof_predictions['y_pred'] < -signal_threshold, -1, signals)
    
    # 将信号合并到日线行情数据中
    data_with_signals = daily_ohlcv.join(pd.Series(signals, name='signal', index=oof_predictions.index), how='left')
    data_with_signals['signal'].fillna(0, inplace=True) # 用 0 填充没有信号的日期

    # 2. 初始化 Cerebro 引擎
    cerebro = bt.Cerebro()

    # 3. 添加策略
    cerebro.addstrategy(
        ModelSignalStrategy,
        holding_period=config.get('strategy_config', {}).get('labeling_horizon', 45)
    )

    # 4. 添加数据
    data_feed = SignalData(dataname=data_with_signals)
    cerebro.adddata(data_feed)

    # 5. 设置初始资金和佣金
    cerebro.broker.setcash(backtest_cfg.get('initial_cash', 100000.0))
    cerebro.broker.setcommission(commission=backtest_cfg.get('commission_pct', 0.0005)) # 0.05%

    # 6. 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, compression=252)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days, compression=252)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    
    # 7. 运行回测
    print("--- 启动事件驱动回测... ---")
    results = cerebro.run()
    strat = results[0]
    print("--- 事件驱动回测完成。 ---")
    
    # 8. 提取并打印结果
    sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()

    metrics = {
        'Start Portfolio Value': f"{cerebro.broker.startingcash:,.2f}",
        'Final Portfolio Value': f"{cerebro.broker.getvalue():,.2f}",
        'Total Return': f"{(cerebro.broker.getvalue() / cerebro.broker.startingcash - 1):.2%}",
        'Annualized Return': f"{returns.get('rnorm100', 0):.2f}%",
        'Sharpe Ratio (Annualized)': f"{sharpe.get('sharperatio', 0):.2f}",
        'Max Drawdown': f"{drawdown.max.drawdown:.2f}%",
        'Total Trades': trade_analysis.total.total,
        'Winning Trades': trade_analysis.won.total,
        'Losing Trades': trade_analysis.lost.total,
        'Win Rate': f"{(trade_analysis.won.total / trade_analysis.total.total):.2%}" if trade_analysis.total.total > 0 else "N/A"
    }
    
    print("\n--- 事件驱动回测业绩报告 ---")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    # 9. 绘图
    if plot:
        print("\n--- 生成事件驱动回测图表 ---")
        cerebro.plot(style='candlestick')
        
    return metrics