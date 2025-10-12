# 文件路径: model_builders/lgbm_builder.py (最终简化版)

import warnings
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Tuple
from lightgbm.callback import early_stopping
from sklearn.preprocessing import StandardScaler

class LGBMBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 1. 从 global_settings 获取基础配置
        global_cfg = config.get('global_settings', {})
        
        # 2. 从 default_model_params 获取 lgbm 的默认参数
        lgbm_defaults = config.get('default_model_params', {}).get('lgbm_params', {}).copy()
        
        # 3. 从 config 的顶层获取可能由 HPO 注入的 trial 参数
        trial_params = config.get('lgbm_params', {}).copy()
        
        # 4. 按优先级合并：trial > yaml_default > global
        # 我们不再设置代码层面的默认值，所有配置都应来自 config 文件
        final_params = {**lgbm_defaults, **trial_params}
        final_params['random_state'] = global_cfg.get('seed', 42) # 确保种子被设置

        self.lgbm_params = final_params
        # --- 修正结束 ---
        
        self.quantiles = global_cfg.get('quantiles', [0.05, 0.5, 0.95])
        self.label_col = global_cfg.get('label_column', 'label_return')

    def _extract_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_with_date = df.reset_index()
        features = [col for col in df.columns if col != self.label_col]
        X = df_with_date[['date'] + features]
        y = df[self.label_col]
        return X, y

    def train_and_evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        X_train, y_train = self._extract_xy(train_df)
        X_val, y_val = self._extract_xy(val_df)
        features_for_model = [col for col in X_train.columns if col != 'date']
        X_train_model, X_val_model = X_train[features_for_model], X_val[features_for_model]
        
        fold_scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(fold_scaler.fit_transform(X_train_model), index=X_train_model.index, columns=features_for_model)
        X_val_scaled = pd.DataFrame(fold_scaler.transform(X_val_model), index=X_val_model.index, columns=features_for_model)

        quantile_models = {}
        for q in self.quantiles:
            params = self.lgbm_params.copy()
            params['alpha'] = q
            
            # 只在第一次循环（第一个分位数）时打印一次参数
            if q == self.quantiles[0]:
                print(f"    - DEBUG: Params passed to LGBMRegressor: {params}")
            
            model = lgb.LGBMRegressor(**params)
            
            early_stopping_rounds = self.lgbm_params.get('early_stopping_rounds', 100)
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='quantile',
                callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )
            quantile_models[f'q_{q}'] = model
        
        median_model = quantile_models.get(f'q_{0.5}')
        daily_ic_df = pd.DataFrame()
        if median_model and not y_val.empty and len(y_val) > 1:
            preds = median_model.predict(X_val_scaled)
            eval_df = pd.DataFrame({"pred": preds, "y": y_val.values, "date": pd.to_datetime(X_val['date'].values)})
            
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", 
                    category=UserWarning, 
                    message="An input array is constant; the correlation coefficient is not defined."
                )
                try:
                    fold_ic = eval_df['pred'].rank().corr(eval_df['y'].rank(), method='spearman')
                    if pd.notna(fold_ic):
                        daily_ic_df = pd.DataFrame([{'date': eval_df['date'].max(), 'rank_ic': fold_ic}])
                except Exception:pass
            
        return {'models': quantile_models}, daily_ic_df

    def train_final_model(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        X_full, y_full = self._extract_xy(full_df)
        features_for_model = [col for col in X_full.columns if col != 'date']
        X_full_model = X_full[features_for_model]
        
        final_scaler = StandardScaler()
        X_full_scaled = pd.DataFrame(final_scaler.fit_transform(X_full_model), index=X_full_model.index, columns=features_for_model)
        
        final_models = {}
        for q in self.quantiles:
            params = self.lgbm_params.copy()
            params['alpha'] = q
            if 'early_stopping_rounds' in params: params.pop('early_stopping_rounds')
            model = lgb.LGBMRegressor(**params)
            model.fit(X_full_scaled, y_full)
            final_models[f'q_{q}'] = model
            
        return {'models': final_models, 'scaler': final_scaler}