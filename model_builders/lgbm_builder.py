# 文件路径: model_builders/lgbm_builder.py
'''
LGBM模型构建 (已修正数据泄露问题)
'''

import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple

class LGBMBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        global_cfg = config.get('global_settings', {})
        lgbm_cfg = config.get('lgbm_params', {})
        
        default_lgbm_params = {
            "random_state": global_cfg.get('seed', 42),
            "feature_fraction_seed": global_cfg.get('seed', 42) + 1,
            "bagging_seed": global_cfg.get('seed', 42) + 2,
            "objective": "quantile", "metric": "quantile", "n_estimators": 2000,
            "learning_rate": 0.01, "num_leaves": 10, "min_child_samples": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "reg_alpha": 0.1, "reg_lambda": 0.1, "n_jobs": -1, "verbose": -1,
        }
        
        self.lgbm_params = {**default_lgbm_params, **lgbm_cfg}
        self.quantiles = global_cfg.get('quantiles', [0.05, 0.5, 0.95])
        self.label_col = global_cfg.get('label_column', 'label_return')
        
        # 核心修正：移除 self.scaler。Scaler 将在需要它的方法内部被创建和管理。

    def _extract_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_with_date = df.reset_index()
        features = [col for col in df.columns if col not in [self.label_col]]
        X = df_with_date[['date'] + features]
        y = df[self.label_col]
        return X, y

    def train_and_evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        X_train, y_train = self._extract_xy(train_df)
        X_val, y_val = self._extract_xy(val_df)
        
        features_for_model = [col for col in X_train.columns if col != 'date']
        X_train_model = X_train[features_for_model]
        X_val_model = X_val[features_for_model]
        
        fold_scaler = StandardScaler()
        
        X_train_scaled_np = fold_scaler.fit_transform(X_train_model)
        X_train_scaled = pd.DataFrame(X_train_scaled_np, index=X_train_model.index, columns=features_for_model)
        
        X_val_scaled_np = fold_scaler.transform(X_val_model)
        X_val_scaled = pd.DataFrame(X_val_scaled_np, index=X_val_model.index, columns=features_for_model)

        quantile_models = {}
        for q in self.quantiles:
            params = self.lgbm_params.copy()
            params['alpha'] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[early_stopping(
                    stopping_rounds=self.config.get('global_settings', {}).get('early_stopping_rounds', 100),
                    verbose=False
                )]
            )
            quantile_models[f'q_{q}'] = model
        
        median_model = quantile_models.get(f'q_{0.5}')
        daily_ic_df = pd.DataFrame() # <-- 名字保留，但现在是 Fold IC

        if median_model and not y_val.empty:
            preds = median_model.predict(X_val_scaled)
            
            eval_df = pd.DataFrame({
                "pred": preds, "y": y_val.values, "date": pd.to_datetime(X_val['date'].values)
            })
            
            # --- 核心修正：在整个验证集上计算 IC，移除 groupby ---
            if len(eval_df) > 1: # 至少需要2个数据点来计算相关性
                fold_ic = eval_df['pred'].rank().corr(eval_df['y'].rank(), method='spearman')
                
                if pd.notna(fold_ic):
                    # 创建一个单行 DataFrame 来记录这个 fold 的结果
                    # 使用验证集的最后一个日期作为该 IC 的记录日期
                    last_date = eval_df['date'].max()
                    daily_ic_df = pd.DataFrame([{'date': last_date, 'rank_ic': fold_ic}])
            # --- 修正结束 ---
                
        return {'models': quantile_models}, daily_ic_df

    def train_final_model(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        X_full, y_full = self._extract_xy(full_df)
        features_for_model = [col for col in X_full.columns if col != 'date']
        X_full_model = X_full[features_for_model]
        
        final_scaler = StandardScaler()
        
        # --- 核心修正：同样地，重新包装成 DataFrame ---
        X_full_scaled_np = final_scaler.fit_transform(X_full_model)
        X_full_scaled = pd.DataFrame(X_full_scaled_np, index=X_full_model.index, columns=features_for_model)
        
        final_models = {}
        for q in self.quantiles:
            params = self.lgbm_params.copy()
            params['alpha'] = q
            # 移除早停，在全部数据上训练到指定轮次
            params.pop('n_estimators', None) # 使用默认或已优化的 n_estimators
            model = lgb.LGBMRegressor(**params)
            model.fit(X_full_scaled, y_full)
            final_models[f'q_{q}'] = model
            
        # --- 核心修正：返回包含正确 scaler 的 artifacts ---
        return {'models': final_models, 'scaler': final_scaler}