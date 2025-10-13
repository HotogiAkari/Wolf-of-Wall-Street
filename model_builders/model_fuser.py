# 文件路径: model_builder/model_fuser.py (最终生产级版)

import json
import joblib
import random
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Union

def spearman_corr_scorer(y_true, y_pred):
    """自定义的 Spearman 相关系数评分函数，增加了对常量输入的保护。"""
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    # 如果任一输入数组的值全部相同，方差为0，corr会返回NaN或报错
    if y_true.var() < 1e-8 or y_pred.var() < 1e-8:
        return 0.0 # 返回一个中性/差的分数
    
    correlation = y_true.corr(y_pred, method='spearman')
    return correlation if pd.notna(correlation) else 0.0

class ModelFuser:
    """
    (已重构) 生产级模型融合器，集成了版本化、随机数控制、输出平滑、
    在线监控和多种元模型选项。
    """
    def __init__(self, ticker: str, config: Dict):
        self.ticker = ticker
        self.config = config
        self.meta_model: Optional[Union[Ridge, MLPRegressor]] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.use_fallback = False # 标志位，用于判断是否应回退到简单融合策略

        # 增强功能: 用于在线监控和实盘防抖动的历史记录
        self.online_ic_history: List[float] = []
        self.recent_preds: List[float] = []

        self.model_dir = Path(config.get('global_settings', {}).get('model_dir', 'models')) / ticker
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.fuser_path = self.model_dir / "fuser_model.pkl"
        self.scaler_path = self.model_dir / "fuser_scaler.pkl"
        self.meta_path = self.model_dir / "fuser_meta.json"

        # --- 为增量训练准备 ---
        # 如果模型不存在，我们会初始化一个新的、可以 partial_fit 的模型
        fuser_config = self.config.get('default_model_params', {}).get('fuser_params', {})
        meta_model_type = fuser_config.get('type', 'ridge')
        
        # 确保 Ridge 支持增量训练
        if meta_model_type == 'ridge':
             # 我们使用 SGDRegressor 来实现增量 Ridge，它更适合在线学习
             from sklearn.linear_model import SGDRegressor
             # 'l2' penalty 就是 Ridge 回归
             self.meta_model = SGDRegressor(penalty='l2', alpha=fuser_config.get('alpha', 0.0001), max_iter=1000, tol=1e-3, random_state=config.get('global_settings',{}).get('seed',42))
        elif meta_model_type == 'mlp':
             # MLPRegressor 也支持 partial_fit
             self.meta_model = MLPRegressor(hidden_layer_sizes=(8,), random_state=config.get('global_settings',{}).get('seed',42), warm_start=True)
        else:
            raise TypeError(f"不支持的元模型类型: {meta_model_type}")

        self.scaler = StandardScaler()
        # 高优先级 (2): 增强随机数控制，确保可复现性
        seed = self.config.get('global_settings', {}).get('seed', 42)
        np.random.seed(seed)
        random.seed(seed)
        # 如果未来使用 PyTorch 作为元模型，也应在此处设置:
        # try:
        #     import torch
        #     torch.manual_seed(seed)
        # except ImportError:
        #     pass

    def _get_oof_predictions(self) -> Optional[pd.DataFrame]:
        """加载并合并所有基础模型的 OOF (Out-of-Fold) 真实预测数据。"""
        oof_dfs = []
        models_to_fuse = self.config.get('global_settings', {}).get('models_to_train', ['lgbm', 'lstm'])

        for model_type in models_to_fuse:
            oof_path = self.model_dir / f"{model_type}_oof_preds.csv"
            if not oof_path.exists():
                print(f"WARNNING: 未找到 {model_type} 的 OOF 预测文件: {oof_path}。")
                return None
            
            df = pd.read_csv(oof_path, parse_dates=['date'])
            oof_dfs.append(df.set_index('date')[['y_pred']].rename(columns={'y_pred': f'pred_{model_type}'}))
        
        if not oof_dfs: return None
        
        all_oof_preds = pd.concat(oof_dfs, axis=1).dropna()
        if all_oof_preds.empty:
            print("ERROR: 无法对齐所有模型的 OOF 数据（没有重叠的有效预测日期）。")
            return None
            
        return all_oof_preds

    def train(self):
        """执行完整的、健壮的融合模型训练流程。"""
        print(f"\n--- 正在为 {self.ticker} 训练融合元模型... ---")
        
        all_oof_preds = self._get_oof_predictions()
        oof_lgbm_path = self.model_dir / "lgbm_oof_preds.csv" # 假设 lgbm 必须存在以获取 y_true
        if not oof_lgbm_path.exists() or all_oof_preds is None:
            print("ERROR: 基础模型的 OOF 预测不完整，无法训练融合模型。"); return

        y_true_df = pd.read_csv(oof_lgbm_path, parse_dates=['date']).set_index('date')[['y_true']]
        aligned_data = all_oof_preds.join(y_true_df, how='inner').dropna()
        
        if len(aligned_data) < 50:
             print(f"WARNNING: 对齐后的样本量过少 ({len(aligned_data)} < 50)，跳过融合模型训练。"); return

        pred_cols = [c for c in aligned_data.columns if c.startswith('pred_')]
        X_meta = aligned_data[pred_cols]
        y_meta = aligned_data['y_true']

        if X_meta.var().min() < 1e-8:
            print("WARNING: 元模型输入特征方差过小（预测值恒定），跳过训练。"); return

        self.scaler = StandardScaler(); X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        n_splits = min(5, max(2, len(X_meta) // 100)); tscv = TimeSeriesSplit(n_splits=n_splits)
        print(f"INFO: 样本量为 {len(X_meta)}，自动选择 {n_splits} 折交叉验证。")

        fuser_config = self.config.get('default_model_params', {}).get('fuser_params', {})
        meta_model_type = fuser_config.get('type', 'ridge')
        
        if meta_model_type == 'mlp':
            print("INFO: 使用 MLPRegressor 作为元模型。")
            meta_model_cv = MLPRegressor(hidden_layer_sizes=(8,), max_iter=500, random_state=self.config.get('global_settings',{}).get('seed',42))
        else:
            print("INFO: 使用 Ridge 作为元模型。")
            meta_model_cv = Ridge(alpha=fuser_config.get('alpha', 1.0))

        scorer = make_scorer(spearman_corr_scorer)
        cv_scores = cross_val_score(meta_model_cv, X_meta_scaled, y_meta, cv=tscv, scoring=scorer)
        mean_cv_ic, std_cv_ic = np.mean(cv_scores), np.std(cv_scores)
        
        print(f"  - 交叉验证 IC 分数: {[f'{s:.4f}' for s in cv_scores]}, 平均值: {mean_cv_ic:.4f}")
        
        ic_stability_score = std_cv_ic / (abs(mean_cv_ic) + 1e-8)
        print(f"  - IC 稳定性分数 (变异系数): {ic_stability_score:.4f}")
        
        stability_threshold = fuser_config.get('stability_threshold', 0.5)
        if ic_stability_score > stability_threshold:
            print(f"WARNNING: 模型稳定性差 ({ic_stability_score:.4f} > {stability_threshold})。将回退到简单加权策略。")
            self.use_fallback = True; return

        self.meta_model = meta_model_cv.fit(X_meta_scaled, y_meta)
        y_pred_meta = self.meta_model.predict(X_meta_scaled)
        
        ic_fuser = spearman_corr_scorer(y_meta, y_pred_meta)
        base_ics = {col: spearman_corr_scorer(y_meta, X_meta[col]) for col in pred_cols}

        print("\n--- 训练后性能评估 ---")
        print(f"  - 融合模型 IC: {ic_fuser:.4f}")
        for model_name, ic_val in base_ics.items():
            print(f"  - {model_name.replace('pred_', '').upper()} 单独 IC: {ic_val:.4f}")

        if ic_fuser < max(base_ics.values()):
            print("WARNNING: 融合模型性能未超越最佳单一模型。将回退到简单加权策略。")
            self.use_fallback = True; return

        # 高优先级 (1): 保存模型元信息
        meta_info = {
            "ticker": self.ticker,
            "trained_at": datetime.datetime.now().isoformat(),
            "models_fused": [p.replace('pred_', '') for p in pred_cols],
            "meta_model_type": meta_model_type,
            "cv_mean_ic": float(mean_cv_ic),
            "ic_stability_score": float(ic_stability_score),
            "in_sample_ic_fuser": float(ic_fuser),
            **{f"in_sample_ic_{name.replace('pred_', '')}": float(val) for name, val in base_ics.items()}
        }
        try:
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)
            joblib.dump(self.meta_model, self.fuser_path)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"SUCCESS: 融合模型、Scaler 及元信息已保存。")
            self.is_trained = True
        except Exception as e:
            print(f"ERROR: 保存融合构件失败: {e}"); self.is_trained = False

    def load(self) -> bool:
        """加载已训练的融合模型、Scaler 和元信息。"""
        if self.fuser_path.exists() and self.scaler_path.exists():
            try:
                self.meta_model = joblib.load(self.fuser_path)
                self.scaler = joblib.load(self.scaler_path)
                if not (hasattr(self.meta_model, "coef_") or hasattr(self.meta_model, "intercept_")):
                    raise ValueError("加载的元模型无效（可能未训练）。")
                self.is_trained = True
                print(f"SUCCESS: 融合模型及 Scaler 已成功加载。")
                if self.meta_path.exists():
                    with open(self.meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    print(f"  - 元信息: 模型于 {meta.get('trained_at')} 训练, CV IC={meta.get('cv_mean_ic', 0):.4f}")
                return True
            except Exception as e:
                print(f"ERROR: 加载融合构件时发生错误: {e}")
        return False
        
    def load(self) -> bool:
        """(已更新) 加载模型。如果模型不存在，则保持初始化的空模型状态。"""
        if self.fuser_path.exists() and self.scaler_path.exists():
            try:
                self.meta_model = joblib.load(self.fuser_path)
                self.scaler = joblib.load(self.scaler_path)
                # ... (加载验证)
                self.is_trained = True
                if self.verbose: print(f"SUCCESS: 融合构件已加载。")
                return True
            except Exception as e: 
                if self.verbose: print(f"ERROR: 加载融合构件失败: {e}")
                # 加载失败时，重置为初始状态，以备后续训练
                self.__init__(self.ticker, self.config) 
                self.is_trained = False
        else:
            if self.verbose: print("INFO: 未找到已训练的融合模型文件，将使用新初始化的模型。")
            self.is_trained = False # 明确表示未经过批量训练
        return self.is_trained

    def predict(self, preds: Dict[str, float]) -> float:
        """对一组新的基础模型预测进行融合，包含平滑和边界保护。"""
        if not self.is_trained or self.use_fallback:
            print("WARNNING: 融合模型不可用或不稳定，回退到简单平均融合。")
            return np.mean(list(preds.values()))

        pred_cols = [f'pred_{m}' for m in self.config.get('global_settings', {}).get('models_to_train', [])]
        try:
            X_new = np.array([[preds[col] for col in pred_cols]])
        except KeyError as e:
            raise ValueError(f"预测时缺少输入: {e}")

        X_new_scaled = self.scaler.transform(X_new)
        fused_prediction = self.meta_model.predict(X_new_scaled)[0]
        
        fuser_config = self.config.get('default_model_params', {}).get('fuser_params', {})
        pred_clip = fuser_config.get('pred_clip', 0.05)
        clipped_prediction = np.tanh(fused_prediction / pred_clip) * pred_clip
        
        smooth_window = fuser_config.get('smooth_window', 1)
        if smooth_window > 1:
            self.recent_preds.append(clipped_prediction)
            if len(self.recent_preds) > smooth_window:
                self.recent_preds.pop(0)
            return np.mean(self.recent_preds)
        
        return clipped_prediction
    
    def predict_batch(self, df_preds: pd.DataFrame) -> pd.Series:
        """对一批预测进行融合。"""
        required_cols = [f'pred_{m}' for m in self.config.get('global_settings', {}).get('models_to_train', [])]
        for col in required_cols:
            if col not in df_preds.columns:
                raise ValueError(f"批量预测时缺少必需列: {col}")
        
        if not self.is_trained or self.use_fallback:
             raise RuntimeError("融合模型未训练或不稳定，无法进行批量预测。")
        
        X = df_preds[required_cols].values
        X_scaled = self.scaler.transform(X)
        predictions = self.meta_model.predict(X_scaled)

        fuser_config = self.config.get('default_model_params', {}).get('fuser_params', {})
        pred_clip = fuser_config.get('pred_clip', 0.05)
        clipped_predictions = np.tanh(predictions / pred_clip) * pred_clip
        
        return pd.Series(clipped_predictions, index=df_preds.index)
        
    def update_online_ic(self, y_true_batch: list, y_pred_batch: list, window: int = 30) -> float:
        """
        (增强版) 接收一批真实值和预测值，更新并返回滚动窗口内的在线 IC。
        """
        if len(y_true_batch) != len(y_pred_batch):
            raise ValueError("y_true 和 y_pred 的长度必须一致。")
        
        # 计算当前批次的 IC
        batch_ic = spearman_corr_scorer(y_true_batch, y_pred_batch)
        
        if pd.notna(batch_ic):
            self.online_ic_history.append(batch_ic)
        
        if len(self.online_ic_history) > window:
            self.online_ic_history.pop(0)
        
        # 返回最近 N 个批次的平均 IC
        return np.mean(self.online_ic_history) if self.online_ic_history else 0.0