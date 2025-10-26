# 文件路径: model/fusion_strategy.py
'''
模型权重训练
'''

import pandas as pd

class DynamicWeightingStrategy:
    def __init__(self, ticker: str, full_ic_history_df: pd.DataFrame, config: dict):
        """
        从一个包含所有模型IC历史的DataFrame中计算动态融合权重。
        """
        self.ticker = ticker
        self.config = config
        self.w_lgbm = 0.5  # 默认权重
        self.w_lstm = 0.5
        
        if full_ic_history_df.empty:
            print(f"WARNNING: IC 历史数据框为空。正在为 {ticker} 使用等权重.")
            return

        try:
            ic_lgbm_series = full_ic_history_df[
                (full_ic_history_df['ticker'] == ticker) & 
                (full_ic_history_df['model_type'] == 'lgbm')
            ]['rank_ic']
            
            ic_lstm_series = full_ic_history_df[
                (full_ic_history_df['ticker'] == ticker) & 
                (full_ic_history_df['model_type'] == 'lstm')
            ]['rank_ic']

            if ic_lgbm_series.empty or ic_lstm_series.empty:
                print(f"WARNNING: {ticker} 的IC历史不完整。使用等权重.")
                return

            strategy_cfg = self.config.get('strategy_config', {})
            span = strategy_cfg.get("fusion_ic_span", 120)
            
            rolling_ic_lgbm = abs(ic_lgbm_series.ewm(span=span, adjust=False).mean().iloc[-1])
            rolling_ic_lstm = abs(ic_lstm_series.ewm(span=span, adjust=False).mean().iloc[-1])
            
            total_ic = rolling_ic_lgbm + rolling_ic_lstm
            if total_ic < 1e-8:
                print(f"WARNNING: {ticker} 的总滚动 IC 接近零。使用等权重.")
                return

            min_weight = strategy_cfg.get("fusion_min_weight", 0.2)
            
            raw_w_lgbm = rolling_ic_lgbm / total_ic
            
            self.w_lgbm = max(min_weight, raw_w_lgbm)
            self.w_lstm = 1 - self.w_lgbm
            
            if self.w_lstm < min_weight:
                self.w_lstm = min_weight
                self.w_lgbm = 1 - min_weight
            
            print(f"INFO: {ticker} 的融合权重 -> LGBM: {self.w_lgbm:.2%}, LSTM: {self.w_lstm:.2%}")

        except (KeyError, IndexError) as e:
            print(f"WARNNING: Could not calculate weights for {ticker} due to data issue: {e}. Using equal weights.")

    def fuse(self, pred_lgbm_dict: dict, pred_lstm_scalar: float, historical_preds: pd.DataFrame) -> dict:
        """
        在Z-score标准化后的空间中融合来自不同模型的预测。
        """
        if historical_preds.empty or 'lgbm_pred' not in historical_preds or 'lstm_pred' not in historical_preds:
            print("WARNNING: 历史预测为空或缺少列。无法执行融合.")
            # 返回一个基于LGBM中位数的默认值
            return {key: val for key, val in pred_lgbm_dict.items()}

        mean_lgbm = historical_preds['lgbm_pred'].mean()
        std_lgbm = historical_preds['lgbm_pred'].std()
        mean_lstm = historical_preds['lstm_pred'].mean()
        std_lstm = historical_preds['lstm_pred'].std()
        
        norm_pred_lstm = (pred_lstm_scalar - mean_lstm) / (std_lstm + 1e-8)
        
        fused_preds = {}
        for quantile_key, lgbm_value in pred_lgbm_dict.items():
            norm_lgbm_value = (lgbm_value - mean_lgbm) / (std_lgbm + 1e-8)
            fused_norm_pred = self.w_lgbm * norm_lgbm_value + self.w_lstm * norm_pred_lstm
            fused_preds[quantile_key] = fused_norm_pred

        return fused_preds