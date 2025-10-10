# ---build_models.py---
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import joblib
from pathlib import Path
import sys
import datetime
import torch
import logging
import uuid
import copy
import os
import random
import inspect
from typing import Dict, List, Tuple, Any, Optional
from collections.abc import Mapping
import hashlib

# 导入必要的sklearn组件
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import sklearn

# --- 日志与配置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [PhoenixBuilder:%(funcName)s] - %(levelname)s - %(message)s',
                    stream=sys.stdout)
DEFAULT_CONFIG = {
    "random_seed": 42,
    "walk_forward_splits": 5,
    "train_ratio_per_split": 0.7,
    "validation_ratio_per_split": 0.15,
    "embargo_days": 5,
    "label_column": "label_return",
    "normalize_label": True,
    "ic_rolling_window_ratio": 0.25,
    "min_ic_threshold": 0.02,
    "max_ic_std_threshold": 0.5,
    "feature_selection_method": "lgbm_importance",
    "num_final_features": 100,
    "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
    "early_stopping_rounds": 50,
    "calibration_sample_max": 5000
}

def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"全局随机种子已设置为: {seed}")

def quantile_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))

def deep_update(d, u):
    d = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def log_best_only_callback(period: int):
    """
    一个自定义的 LightGBM 回调函数工厂。
    它创建的回调函数只会在验证集损失创下新低时才打印日志。
    """
    # 使用闭包来为每个模型实例存储各自的最佳分数
    best_score = float('inf')

    def callback(env: lgb.callback.CallbackEnv):
        # 声明 best_score 是来自外部作用域的变量
        nonlocal best_score
        
        # 如果周期不匹配，则不执行任何操作
        if period <= 0 or (env.iteration + 1) % period != 0:
            return

        # 从评估结果列表中找到我们关心的验证集损失 ('valid_0's q_loss)
        for data_name, eval_name, value, _ in env.evaluation_result_list:
            if data_name == 'valid_0' and eval_name == 'q_loss':
                current_score = value
                # 如果当前分数比记录的最佳分数要好（更低）
                if current_score < best_score:
                    # 更新最佳分数
                    best_score = current_score
                    # 打印日志，并附加 "(New best)" 提示
                    print(f'[{env.iteration + 1}]\t{data_name}'
                          f'\'s {eval_name}: {current_score:.6f} (New best)')
                # 找到后即可退出内部循环
                break
    
    # 返回最终创建的回调函数
    return callback

class QuantileInferencePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, models_dict, calibrators_dict, selected_features, quantiles_order, label_transformer=None):
        self.scaler = scaler
        self.models = models_dict
        self.calibrators = calibrators_dict
        self.selected_features_ = selected_features
        self.quantiles_order_ = quantiles_order
        self.label_transformer_ = label_transformer
        self.required_features_set_ = set(selected_features)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.required_features_set_.issubset(X.columns):
            missing = self.required_features_set_ - set(X.columns)
            raise ValueError(f"输入数据中缺少必要的特征: {missing}")
        X_reordered = X[self.selected_features_]
        X_scaled = self.scaler.transform(X_reordered)

        # 预测 (归一化/模型输出)
        norm_predictions = np.column_stack([self.models[f'q_{q}'].predict(X_scaled) for q in self.quantiles_order_])

        # [FIX v4.5] QuantileTransformer was fit on a single-column label.
        # We must inverse_transform column-by-column.
        if self.label_transformer_:
            # each column is shape (n_samples, 1) for inverse_transform
            orig_list = []
            for i in range(norm_predictions.shape[1]):
                col = norm_predictions[:, i].reshape(-1, 1)
                inv = self.label_transformer_.inverse_transform(col).ravel()
                orig_list.append(inv)
            orig_predictions = np.column_stack(orig_list)
        else:
            orig_predictions = norm_predictions

        # 校准（Isotonic 回归或其它）
        calibrated_predictions = np.column_stack([
            self.calibrators[f'q_{q}'].predict(orig_predictions[:, i]) for i, q in enumerate(self.quantiles_order_)
        ])

        # 强制单调（防止分位数交叉）
        for i in range(1, calibrated_predictions.shape[1]):
            calibrated_predictions[:, i] = np.maximum(calibrated_predictions[:, i], calibrated_predictions[:, i-1])

        return pd.DataFrame(calibrated_predictions, index=X.index, columns=[f'pred_{q}' for q in self.quantiles_order_])

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return self.predict(X)

class ModelBuilder:
    """封装整个模型构建、验证和保存流程的类。"""
    def __init__(self, ticker: str, config: Dict, input_dir: str, output_dir: str):
        self.ticker = ticker
        self.config = deep_update(DEFAULT_CONFIG, config)
        set_global_seed(self.config['random_seed'])
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_artifacts = {}
        self.walk_forward_metrics = []

        self.device = 'cpu'
        if self.config.get('device', 'auto').lower() in ['gpu', 'auto'] and torch.cuda.is_available():
            try:
                lgb.LGBMRegressor(device='gpu')
                self.device = 'gpu'
            except Exception:
                logging.warning("LightGBM GPU 支持似乎未正确配置，将回退到CPU模式。")

        logging.info(f"已为股票 {self.ticker} 初始化模型构建器，将使用设备: {self.device.upper()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_lgbm_device_arg(self) -> Dict:
        """根据当前环境和 LightGBM 版本，返回正确的设备参数字典。"""
        if self.device == 'gpu':
            sig = inspect.signature(lgb.LGBMRegressor.__init__)
            if 'device' in sig.parameters:
                return {'device': 'gpu'}
            elif 'device_type' in sig.parameters:
                return {'device_type': 'gpu'}
        return {}

    def build(self):
        logging.info(f"[M01.0] 开始为股票 {self.ticker} 构建模型...")
        df = self._load_latest_data()
        if df is None:
            return

        for i, (train_df, val_df) in enumerate(self._walk_forward_split(df)):
            logging.info("\n" + "="*80)
            logging.info(f"[WF Split {i+1}/{self.config['walk_forward_splits']}] 开始处理...")
            X_train, y_train = self._extract_xy(train_df)
            X_val, y_val = self._extract_xy(val_df)
            y_train_norm, y_val_norm, label_transformer = self._normalize_labels(y_train, y_val)
            X_train_processed, X_val_processed, y_train_processed, y_val_processed, selected_features, scaler = self._preprocess_and_select_features(
                X_train, X_val, y_train_norm, y_val_norm)
            if X_train_processed is None:
                continue
            quantile_models = self._train_quantile_models(X_train_processed, y_train_processed, X_val_processed, y_val_processed)
            self._evaluate_and_log_split_metrics(quantile_models, X_val_processed, y_val, label_transformer, split_num=i+1)

        self._finalize_and_save_model(df)
        logging.info(f"[M08.0] SUCCESS: 股票 {self.ticker} 的模型构建与Walk-Forward验证完成。")

    def _load_latest_data(self) -> Optional[pd.DataFrame]:
        """从确定性的数据路径中加载特征数据文件。"""
        logging.info("正在加载最新的已处理数据...")
        try:
            # 从配置中获取构建路径所需的所有信息
            start_date = self.config.get('start_date')
            end_date = self.config.get('end_date')
            if not start_date or not end_date:
                raise ValueError("配置文件中缺少 'start_date' 或 'end_date'。")

            # 1. 构建确定性的目录路径
            date_range_str = f"{start_date}_to_{end_date}"
            # self.input_dir 是 'data/processed'
            # self.ticker 是 '600519.SH'
            # 最终目录是 'data/processed/600519.SH/2021-01-01_to_2023-12-31/'
            data_dir = self.input_dir / self.ticker / date_range_str
            
            if not data_dir.exists():
                raise FileNotFoundError(f"未找到股票 {self.ticker} 在日期范围 {date_range_str} 内的数据目录: {data_dir}。请先运行 `save_data.py`。")

            # 2. 在目录中查找最新的 .pkl 文件
            # 允许多个哈希版本存在，总是使用最新的一个
            all_pkl_files = list(data_dir.glob('features_*.pkl'))
            if not all_pkl_files:
                raise FileNotFoundError(f"在目录 {data_dir} 中未找到任何特征数据文件 (*.pkl)。")
            
            # 按修改时间排序，获取最新的文件
            latest_file = max(all_pkl_files, key=os.path.getmtime)
            logging.info(f"找到最新的数据文件: {latest_file}")
            
            # 3. 加载数据
            df = pd.read_pickle(latest_file)
            if 'date' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            
            logging.info(f"数据加载成功。数据集维度: {df.shape}")
            return df
            
        except (FileNotFoundError, IndexError, ValueError) as e:
            logging.error(f"未能找到或加载 {self.ticker} 的已处理数据: {e}")
            return None
        except Exception as e:
            logging.error(f"加载数据时发生未知错误 ({self.ticker}): {e}")
            return None

    def _walk_forward_split(self, df: pd.DataFrame):
        df = df.sort_values('date').reset_index(drop=True)
        n_splits = self.config['walk_forward_splits']
        total_len = len(df)
        initial_train_size = int(total_len * self.config['train_ratio_per_split'])
        validation_size = int(total_len * self.config['validation_ratio_per_split'])
        step_size = validation_size
        if initial_train_size < 100 or validation_size < 20:
            logging.error(f"数据集太小 (n={total_len})，无法进行有效的Walk-Forward分割。")
            return
        for i in range(n_splits):
            train_end = initial_train_size + i * step_size
            val_start = train_end + self.config['embargo_days']
            val_end = val_start + validation_size
            if val_end > total_len:
                break
            yield df.iloc[0:train_end], df.iloc[val_start:val_end]

    def _extract_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        label_col = self.config['label_column']
        features = [col for col in df.columns if col not in [label_col, 'date']]
        # 保证返回的 label Series 带 name
        y = df[label_col] if label_col in df.columns else pd.Series(dtype=float, name=label_col)
        if y.name is None:
            y.name = label_col
        return df[features], y

    def _normalize_labels(self, y_train: pd.Series, y_val: pd.Series) -> Tuple[pd.Series, pd.Series, Optional[QuantileTransformer]]:
        logging.info(f"标签标准差: {y_train.std():.6f}")
        if y_train.std() < 1e-6:
            logging.warning("WARNNING: 标签几乎没有波动，模型可能无法学习有效分裂。")
        if not self.config['normalize_label']:
            # Ensure names are set
            if y_train.name is None:
                y_train = y_train.rename(self.config['label_column'])
            if y_val.name is None:
                y_val = y_val.rename(self.config['label_column'])
            return y_train, y_val, None

        logging.info("正在对标签进行Rank Gauss正态化...")
        # 确保 y_train 有足够样本
        n_quantiles = min(max(len(y_train) // 10, 30), 1000)
        # [安全] n_quantiles must be >= 2
        n_quantiles = max(2, n_quantiles)
        transformer = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles, random_state=self.config['random_seed'])
        # if y_train empty -> fit will fail. Guard:
        if y_train.empty:
            y_train_norm = pd.Series(dtype=float, name=y_train.name if y_train.name else self.config['label_column'])
        else:
            y_train_norm = pd.Series(transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index, name=y_train.name if y_train.name else self.config['label_column'])

        # prepare y_val_norm with same name
        if y_val is None or y_val.empty:
            y_val_norm = pd.Series(dtype=float, name=y_train_norm.name)
        else:
            y_val_norm = pd.Series(transformer.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index, name=y_val.name if y_val.name else y_train_norm.name)

        return y_train_norm, y_val_norm, transformer

    def _preprocess_and_select_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> Tuple:
        logging.info("正在进行数据预处理和特征筛选...")
        for df_ in [X_train, X_val, y_train, y_val]:
            df_.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_.fillna(0, inplace=True)
        # ensure label names
        label_name = y_train.name if (y_train is not None and y_train.name) else self.config['label_column']
        if y_val is None:
            y_val = pd.Series(dtype=float, name=label_name)
        elif y_val.name is None:
            y_val = y_val.rename(label_name)
        if y_train is None:
            y_train = pd.Series(dtype=float, name=label_name)
        elif y_train.name is None:
            y_train = y_train.rename(label_name)

        window = min(max(30, int(max(1, len(y_train)) * self.config['ic_rolling_window_ratio'])), 252)
        ic_values = {}
        for col in X_train.columns:
            aligned = pd.concat([X_train[col], y_train], axis=1).dropna()
            if len(aligned) < window:
                continue
            rolling_ic = aligned.iloc[:, 0].rolling(window=window, min_periods=max(1, window//2)).corr(aligned.iloc[:, 1])
            ic_values[col] = (rolling_ic.mean(), rolling_ic.std())

        ic_rolling_mean = pd.Series({k: v[0] for k, v in ic_values.items()}).fillna(0)
        ic_rolling_std = pd.Series({k: v[1] for k, v in ic_values.items()}).fillna(0)
        min_ic, max_ic_std = self.config['min_ic_threshold'], self.config['max_ic_std_threshold']
        stable_features = ic_rolling_mean[(ic_rolling_mean.abs() > min_ic) & (ic_rolling_std < max_ic_std)].index.tolist()

        if not stable_features:
            logging.error("经过稳定性筛选后没有剩余特征。")
            return None, None, None, None, None, None

        # 临时训练轻量模型以得到特征重要性
        temp_model = lgb.LGBMRegressor(random_state=self.config['random_seed'], **self._get_lgbm_device_arg())
        # guard: X_train[stable_features] may have NaN rows; LightGBM can handle but we drop rows with NaN in both X/y
        tmp_comb = pd.concat([X_train[stable_features], y_train], axis=1).dropna()
        if tmp_comb.empty:
            logging.error("用于特征重要性计算的训练数据为空。")
            return None, None, None, None, None, None
        temp_model.fit(tmp_comb[stable_features], tmp_comb[y_train.name])

        # feature importance: ensure index alignment
        try:
            gains = temp_model.booster_.feature_importance(importance_type='gain')
            feat_names = temp_model.booster_.feature_name()
            feature_importance = pd.Series(gains, index=[str(f) for f in feat_names])
            # intersect with stable_features in case ordering differs
            feature_importance = feature_importance.reindex([str(f) for f in stable_features]).fillna(0)
        except Exception:
            # fallback
            feature_importance = pd.Series(0, index=stable_features)

        selected_features = feature_importance.sort_values(ascending=False).head(self.config['num_final_features']).index.tolist()

        # Prepare cleaned train
        combined_train = pd.concat([X_train[selected_features], y_train], axis=1).dropna()
        if combined_train.empty:
            logging.error("经过去NaN清理后，训练集为空。")
            return None, None, None, None, None, None
        X_train_clean = combined_train[selected_features]
        y_train_clean = combined_train[y_train.name]

        # Prepare cleaned val (y_val may be empty)
        if y_val is None or y_val.empty:
            X_val_clean = pd.DataFrame(columns=selected_features)
            y_val_clean = pd.Series(dtype=float, name=y_train.name)
        else:
            combined_val = pd.concat([X_val[selected_features], y_val], axis=1).dropna()
            if combined_val.empty:
                X_val_clean = pd.DataFrame(columns=selected_features)
                y_val_clean = pd.Series(dtype=float, name=y_train.name)
            else:
                X_val_clean = combined_val[selected_features]
                y_val_clean = combined_val[y_val.name]

        if X_train_clean.empty or (not y_val.empty and X_val_clean.empty):
            logging.error("清理NaN后，训练集或验证集为空。")
            return None, None, None, None, None, None

        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_clean), index=X_train_clean.index, columns=selected_features)
        if not X_val_clean.empty:
            X_val_scaled = pd.DataFrame(scaler.transform(X_val_clean), index=X_val_clean.index, columns=selected_features)
        else:
            X_val_scaled = pd.DataFrame(columns=selected_features)

        return X_train_scaled, X_val_scaled, y_train_clean, y_val_clean, selected_features, scaler

    def _train_quantile_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        if X_train.shape[1] == 0:
            logging.error("ERROR: 没有可用特征，跳过模型训练。")
            return None
        logging.info("正在训练LightGBM分位数模型...")
        quantile_models = {}

        def make_lgb_feval(q):
            # LightGBM scikit-learn wrapper expects feval(y_true, y_pred)
            def feval(y_true, y_pred):
                return 'q_loss', float(quantile_loss(y_true, y_pred, q)), False
            return feval

        for q in self.config['quantiles']:
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "None",
                "n_estimators": 2000,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "num_leaves": 31,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
                "n_jobs": -1,
                "random_state": self.config['random_seed'],
                "deterministic": True,
                "force_col_wise": True,
                **self._get_lgbm_device_arg()
            }
            try:
                model = lgb.LGBMRegressor(**params)
            except TypeError:
                params.pop('deterministic', None)
                model = lgb.LGBMRegressor(**params)

            log_period = self.config.get('log_evaluation_period', 0)
            
            fit_params = {
                "X": X_train,
                "y": y_train,
                "callbacks": [log_best_only_callback(period=log_period)]
            }

            # 确保 early_stopping 逻辑不受影响
            if X_val is not None and not (isinstance(X_val, pd.DataFrame) and X_val.empty):
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["callbacks"].insert(0, lgb.early_stopping(self.config['early_stopping_rounds'], verbose=False))
                fit_params["eval_metric"] = make_lgb_feval(q)

            # [FIX v4.5] 仅在有验证集时使用 early_stopping 与 eval_metric
            if X_val is not None and not (isinstance(X_val, pd.DataFrame) and X_val.empty):
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["callbacks"].insert(0, lgb.early_stopping(self.config['early_stopping_rounds'], verbose=False))
                fit_params["eval_metric"] = make_lgb_feval(q)

            model.fit(**fit_params)
            quantile_models[f'q_{q}'] = model

        logging.info("所有分位数模型训练完成。")
        return quantile_models

    def _evaluate_and_log_split_metrics(self, models: Dict, X_val: pd.DataFrame, y_val: pd.Series, label_transformer: Optional[QuantileTransformer], split_num: int):
        logging.info(f"正在评估模型 (Split {split_num})...")
        if X_val is None or (isinstance(X_val, pd.DataFrame) and X_val.empty):
            logging.warning(f"验证集 (Split {split_num}) 为空，跳过评估。")
            return

        preds_norm = pd.DataFrame({name: model.predict(X_val) for name, model in models.items()}, index=X_val.index)

        if label_transformer:
            preds_orig = pd.DataFrame(index=X_val.index)
            for col in preds_norm.columns:
                arr2d = preds_norm[[col]].values.reshape(-1, 1)
                preds_orig[col] = label_transformer.inverse_transform(arr2d).ravel()
        else:
            preds_orig = preds_norm

        quantiles_sorted = self.config['quantiles']
        quantile_col_names = [f'q_{q}' for q in quantiles_sorted]
        preds_orig = preds_orig.reindex(columns=quantile_col_names)

        split_metrics = {'split': split_num, 'metrics': {}}
        for q in quantiles_sorted:
            col_name = f'q_{q}'
            if col_name not in preds_orig.columns or preds_orig[col_name].isna().all():
                actual_coverage, loss, rank_ic = np.nan, np.nan, np.nan
            else:
                actual_coverage = (y_val < preds_orig[col_name]).mean()
                loss = quantile_loss(y_val.values, preds_orig[col_name].values, q)
                rank_ic = preds_orig[col_name].rank().corr(y_val.rank(), method='spearman')
            
            split_metrics['metrics'][f'q_{q}'] = {'coverage': actual_coverage, 'loss': loss, 'rank_ic': rank_ic}

        # --- 新增逻辑：计算方向准确率 ---
        median_pred = preds_orig.get('q_0.5')
        if median_pred is not None and not y_val.empty:
            # 如果预测和真实的符号相同（都>0或都<0），则方向正确
            correct_direction = (np.sign(median_pred) * np.sign(y_val)) >= 0
            accuracy = correct_direction.mean()
            # 将准确率添加到中位数模型的指标中
            if 'q_0.5' in split_metrics['metrics']:
                split_metrics['metrics']['q_0.5']['accuracy'] = accuracy
                logging.info(f"    - 方向准确率 (Split {split_num}): {accuracy:.2%}")

        self.walk_forward_metrics.append(split_metrics)
        logging.info(f"评估完成 (Split {split_num}).")

        preds_norm = pd.DataFrame({name: model.predict(X_val) for name, model in models.items()}, index=X_val.index)

        if label_transformer:
            # [FIX v4.5] 对每列单独 inverse_transform，避免维度/特征数量不匹配
            preds_orig = pd.DataFrame(index=X_val.index)
            for col in preds_norm.columns:
                # pass a 2D array with shape (n_samples, 1)
                arr2d = preds_norm[[col]].values.reshape(-1, 1)
                preds_orig[col] = label_transformer.inverse_transform(arr2d).ravel()
        else:
            preds_orig = preds_norm

        quantiles_sorted = self.config['quantiles']
        quantile_col_names = [f'q_{q}' for q in quantiles_sorted]
        # ensure columns exist (models dict keys may be in certain order)
        preds_orig = preds_orig.reindex(columns=quantile_col_names)

        split_metrics = {'split': split_num, 'metrics': {}}
        for q in quantiles_sorted:
            col_name = f'q_{q}'
            # handle possible NaNs
            if col_name not in preds_orig.columns or preds_orig[col_name].isna().all():
                actual_coverage = np.nan
                loss = np.nan
                rank_ic = np.nan
            else:
                actual_coverage = (y_val < preds_orig[col_name]).mean()
                loss = quantile_loss(y_val.values, preds_orig[col_name].values, q)
                rank_ic = preds_orig[col_name].rank().corr(y_val.rank(), method='spearman')
            split_metrics['metrics'][f'q_{q}'] = {'coverage': actual_coverage, 'loss': loss, 'rank_ic': rank_ic}

        self.walk_forward_metrics.append(split_metrics)
        logging.info(f"评估完成 (Split {split_num}).")

    def _calibrate_single_model(self, preds_col: pd.Series, y_val: pd.Series, sample_indices: np.ndarray) -> IsotonicRegression:
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        # [FIX v4.5] 使用 numpy 索引，避免 pandas 索引错配（datetime index 等）
        x_sample = preds_col.values[sample_indices]
        y_sample = y_val.values[sample_indices]
        iso_reg.fit(x_sample, y_sample)
        # set bounds for later use (not strictly necessary but consistent with earlier pattern)
        iso_reg.X_min_ = x_sample.min() if len(x_sample) > 0 else None
        iso_reg.X_max_ = x_sample.max() if len(x_sample) > 0 else None
        return iso_reg

    def _finalize_and_save_model(self, full_df: pd.DataFrame):
        logging.info("\n" + "="*80)
        logging.info("正在使用全部数据最终化模型...")

        X_full, y_full = self._extract_xy(full_df)
        y_full_norm, _, label_transformer = self._normalize_labels(y_full, pd.Series(dtype=float, name=self.config['label_column']))

        X_full_processed, _, y_full_processed, _, selected_features, scaler = self._preprocess_and_select_features(
            X_full, pd.DataFrame(columns=X_full.columns), y_full_norm, pd.Series(dtype=float, name=self.config['label_column'])
        )
        if X_full_processed is None:
            logging.error("最终化模型失败：预处理后无数据。")
            return

        logging.info("正在重新训练最终的分位数模型...")
        final_models = self._train_quantile_models(X_full_processed, y_full_processed, pd.DataFrame(), pd.Series(dtype=float, name=self.config['label_column']))

        calib_size = max(1, int(len(X_full_processed) * 0.1))
        X_calib = X_full_processed.iloc[-calib_size:]
        # y_full is original label (not normalized) - pick by index
        y_calib_orig = y_full.loc[X_calib.index] if not y_full.empty else pd.Series(dtype=float, index=X_calib.index, name=self.config['label_column'])

        logging.info("正在校准最终模型...")
        preds_norm_calib = pd.DataFrame({name: model.predict(X_calib) for name, model in final_models.items()}, index=X_calib.index)

        if label_transformer:
            # [FIX v4.5] 逐列 inverse_transform，确保动作与 QuantileTransformer 的 fit 时列数一致
            preds_orig_calib = pd.DataFrame(index=X_calib.index)
            for col in preds_norm_calib.columns:
                arr2d = preds_norm_calib[[col]].values.reshape(-1, 1)
                preds_orig_calib[col] = label_transformer.inverse_transform(arr2d).ravel()
        else:
            preds_orig_calib = preds_norm_calib

        # 校准采样保护
        calib_max_size = self.config['calibration_sample_max']
        num_samples_to_take = min(len(y_calib_orig), calib_max_size)
        if num_samples_to_take <= 0:
            logging.warning("校准样本数量为0，跳过校准步骤。")
            final_calibrators = {col: IsotonicRegression(out_of_bounds="clip") for col in preds_orig_calib.columns}
        else:
            uniform_sample_indices = np.linspace(0, len(y_calib_orig) - 1, num_samples_to_take, dtype=int)
            # [FIX v4.5] 并行校准器构建时确保使用 numpy 索引
            calibrator_tuples = Parallel(n_jobs=-1)(
                delayed(self._calibrate_single_model)(preds_orig_calib[col], y_calib_orig, uniform_sample_indices)
                for col in preds_orig_calib.columns
            )
            final_calibrators = {col: iso for col, iso in zip(preds_orig_calib.columns, calibrator_tuples)}

        self._build_and_save_final_artifacts(scaler, final_models, final_calibrators, selected_features, label_transformer)

    def _build_and_save_final_artifacts(self, scaler, models, calibrators, features, label_transformer):
        logging.info("正在构建并保存最终的推理Pipeline和所有构件...")
        quantiles_sorted = self.config['quantiles']
        pipeline = QuantileInferencePipeline(scaler=scaler, models_dict=models, calibrators_dict=calibrators, selected_features=features, quantiles_order=quantiles_sorted, label_transformer=label_transformer)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:6]
        horizon = self.config.get('labeling_horizon', 'N/A')
        model_version_dir = self.output_dir / self.ticker / f"model_h{horizon}_v_{timestamp}_{unique_id}"
        model_version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_version_dir / "inference_pipeline.joblib")

        median_model = models.get(f"q_{0.5}")
        if median_model:
            try:
                feat_names = [str(f) for f in median_model.booster_.feature_name()]
                gains = median_model.booster_.feature_importance(importance_type='gain')
                importance_df = pd.DataFrame({'feature': feat_names, 'gain': gains}).sort_values('gain', ascending=False)
                importance_df.to_csv(model_version_dir / "feature_importance.csv", index=False)
            except Exception:
                logging.warning("保存特征重要性时发生异常，已跳过。")

        avg_wf_metrics = {}
        if self.walk_forward_metrics:
            # --- 修改的逻辑：处理包含准确率的指标 ---
            metrics_list = []
            for m in self.walk_forward_metrics:
                for q, v in m['metrics'].items():
                    metrics_list.append({
                        'split': m['split'], 'quantile': q, 'coverage': v['coverage'],
                        'loss': v['loss'], 'rank_ic': v['rank_ic'], 'accuracy': v.get('accuracy')
                    })
            df_metrics = pd.DataFrame(metrics_list)
            avg_metrics = df_metrics.groupby('quantile')[['coverage', 'loss', 'rank_ic', 'accuracy']].mean().to_dict('index')
            
            avg_wf_metrics = {}
            for q, v in avg_metrics.items():
                avg_wf_metrics[q] = {
                    'avg_coverage': v['coverage'], 'avg_loss': v['loss'],
                    'avg_rank_ic': v['rank_ic']
                }
                if pd.notna(v.get('accuracy')):
                    avg_wf_metrics[q]['avg_accuracy'] = v['accuracy']
            # --- 修改结束 ---

        metadata = {
            "build_info": {"ticker": self.ticker, "build_id": f"{timestamp}_{unique_id}", "model_type": "LightGBM_Quantile_v4.5"},
            "config_snapshot": self.config,
            "walk_forward_validation_summary": avg_wf_metrics or "No validation splits executed.",
            "walk_forward_validation_details": self.walk_forward_metrics,
            "final_model_artifacts": {"num_final_features": len(features)},
            "dependencies": {"lightgbm": lgb.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
        }

        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()
        metadata['sha256_signature'] = metadata_hash

        with open(model_version_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logging.info(f"Pipeline和元数据已保存至: {model_version_dir}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Phoenix Project - Model Builder (v4.5)")
    parser.add_argument("keyword", type=str, help="要构建模型的股票名称 (来自 config.json 中的 'keyword' 字段)")
    parser.add_argument("--num_final_features", type=int, help="最终筛选出的特征数量")
    parser.add_argument("--early_stopping_rounds", type=int, help="Early stopping 的轮数")
    parser.add_argument("--walk_forward_splits", type=int, help="Walk-forward 交叉验证的折数")
    parser.add_argument("--normalize_label", type=lambda x: (str(x).lower() == 'true'), help="是否对标签进行正态化 (True/False)")
    args = parser.parse_args()

    KEYWORD_TO_BUILD = args.keyword
    CONFIG_FILE_PATH = 'config.json'
    logging.info(f"开始为 '{KEYWORD_TO_BUILD}' 构建模型...")

    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.critical(f"致命错误，配置文件未找到: '{CONFIG_FILE_PATH}'。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.critical(f"致命错误，配置文件格式无效: {e}")
        sys.exit(1)

    global_settings = config.get('global_settings', {})

    stock_config = next((item for item in config.get('stocks_to_process', []) if item.get("keyword") == KEYWORD_TO_BUILD), None)

    if not stock_config:
        logging.critical(f"致命错误，关键字 '{KEYWORD_TO_BUILD}' 未在配置文件中找到。")
        sys.exit(1)

    TICKER_TO_BUILD = stock_config.get("ticker")
    if not TICKER_TO_BUILD:
        logging.critical(f"致命错误，在 '{KEYWORD_TO_BUILD}' 的配置中未找到 'ticker' 字段。")
        sys.exit(1)

    final_config = deep_update(DEFAULT_CONFIG, global_settings)
    final_config = deep_update(final_config, stock_config)

    cli_args = {key: val for key, val in vars(args).items() if val is not None and key != 'keyword'}
    final_config = deep_update(final_config, cli_args)
    logging.info(f"找到配置，将为股票 {TICKER_TO_BUILD} 构建模型。")
    logging.info(f"最终生效的配置 (部分): num_features={final_config.get('num_final_features')}, early_stopping={final_config.get('early_stopping_rounds')}")

    INPUT_DATA_DIR = final_config.get('output_dir', 'data/processed')
    OUTPUT_MODEL_DIR = final_config.get('model_dir', 'models')

    builder = ModelBuilder(ticker=TICKER_TO_BUILD, config=final_config, input_dir=INPUT_DATA_DIR, output_dir=OUTPUT_MODEL_DIR)
    builder.build()
