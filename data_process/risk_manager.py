# 文件路径: data_process/risk_manager.py

import uuid
import sqlite3
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

class OrderStatus(Enum):
    PENDING = 'PENDING'
    CONFIRMED = 'CONFIRMED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'
    FILLED = 'FILLED'

class RiskManager:
    """
    一个集成了订单历史、幂等性检查和核心风控规则的中心化风险控制服务。
    """
    def __init__(self, db_path: str = "order_history.db", timeout: float = 5.0):
        """
        初始化 RiskManager。
        Args:
            db_path (str): SQLite 数据库文件的路径。
            timeout (float): 数据库连接的超时时间 (秒)。
        """
        self.db_path = Path(db_path)
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
            self._initialize_db()
            print(f"SUCCESS: RiskManager initialized and connected to '{self.db_path}'.")
        except sqlite3.Error as e:
            print(f"ERROR: Failed to connect to SQLite database: {e}")
            raise

    def _initialize_db(self):
        """创建 orders 表（如果它不存在）。"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            price REAL,
            status TEXT NOT NULL
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"ERROR: Failed to create 'orders' table: {e}")

    def _check_price_deviation(self, price: float, latest_market_data: pd.DataFrame, z_score_threshold: float = 3.0) -> bool:
        """检查信号价格相对于近期历史波动的偏差。"""
        if latest_market_data.empty:
            print("WARNNING: Market data is empty, cannot check for price deviation.")
            return True # 如果没有数据，暂时放行
            
        returns = np.log(latest_market_data['close'] / latest_market_data['close'].shift(1))
        volatility = returns.std()
        last_price = latest_market_data['close'].iloc[-1]
        
        price_change_ratio = abs(price / last_price - 1)
        
        if volatility > 0 and price_change_ratio > (volatility * z_score_threshold):
            print(f"WARNNING: Price deviation > {z_score_threshold}-sigma for {latest_market_data.iloc[-1].name}. "
                  f"Change: {price_change_ratio:.2%}, Volatility: {volatility:.2%}. Signal rejected.")
            return False
        return True

    def _check_duplicate_signal(self, ticker: str, direction: str, window_minutes: int = 5) -> bool:
        """检查在指定时间窗口内是否已存在同方向的 PENDING 或 CONFIRMED 订单。"""
        five_mins_ago = (datetime.now() - timedelta(minutes=window_minutes)).isoformat()
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT 1 FROM orders WHERE ticker=? AND direction=? AND timestamp > ? AND (status = ? OR status = ?)",
                (ticker, direction, five_mins_ago, OrderStatus.PENDING.value, OrderStatus.CONFIRMED.value)
            )
            if cursor.fetchone():
                print(f"INFO: Duplicate signal for {ticker} ({direction}) within {window_minutes} mins. Skipped.")
                return False
            return True
        except sqlite3.Error as e:
            print(f"ERROR: Failed to check for duplicate signals: {e}")
            return False

    def approve_trade(self, ticker: str, direction: str, price: float, latest_market_data: pd.DataFrame, config: dict = None) -> tuple[bool, str | None]:
        """对一个交易信号进行完整的风险审批。"""
        cfg = config.get('strategy_params', {}) if config else {}

        if not self._check_price_deviation(price, latest_market_data, cfg.get('price_deviation_zscore', 3.0)):
            return False, None

        if not self._check_duplicate_signal(ticker, direction, cfg.get('duplicate_signal_window_min', 5)):
            return False, None
            
        order_id = str(uuid.uuid4())
        try:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.execute(
                "INSERT INTO orders (order_id, timestamp, ticker, direction, price, status) VALUES (?, ?, ?, ?, ?, ?)",
                (order_id, datetime.now().isoformat(), ticker, direction, price, OrderStatus.PENDING.value)
            )
            self.conn.commit()
            return True, order_id 
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"WARNNING: Database is locked. Could not write order for {ticker}. Trade rejected.")
            else:
                print(f"ERROR: Failed to write pending order to DB: {e}")
            self.conn.rollback()
            return False, None
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"ERROR: Failed to write pending order to DB: {e}")
            return False, None

    def update_order_status(self, order_id: str, new_status: OrderStatus):
        """更新数据库中指定订单的状态。"""
        if not isinstance(new_status, OrderStatus):
            raise TypeError(f"new_status must be an OrderStatus enum member, not {type(new_status)}")
            
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE orders SET status = ? WHERE order_id = ?", (new_status.value, order_id))
            self.conn.commit()
            print(f"INFO: Order {order_id} status updated to {new_status.name}.")
        except sqlite3.Error as e:
            print(f"ERROR: Failed to update order status for {order_id}: {e}")

    def close_connection(self):
        """安全地关闭数据库连接。"""
        if self.conn:
            self.conn.close()
            print("INFO: RiskManager database connection closed.")