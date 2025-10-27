# 文件路径: model/builders/lgbm_builder.py

import warnings
import pandas as pd
import lightgbm as lgb
from typing import Any, Dict, Tuple
from lightgbm.callback import early_stopping
from sklearn.preprocessing import StandardScaler

class LGBMBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        global_cfg = config.get('global_settings', {})
        # 合并所有层级的 lgbm 参数
        default_params = config.get('default_model_params', {}).get('lgbm_params', {})
        hpo_fixed_params = config.get('hpo_config', {}).get('lgbm_hpo_config', {}).get('params', {})
        trial_params = config.get('lgbm_params', {})
        
        final_params = {**default_params, **hpo_fixed_params, **trial_params}
        final_params['random_state'] = global_cfg.get('seed', 42)
        # 强制关闭 C++ 内核日志
        final_params['verbose'] = -1
        
        self.lgbm_params = final_params
        
        self.verbose_period = self.lgbm_params.get('verbose_period', -1)
        self.verbose = self.verbose_period > 0
        
        self.quantiles = global_cfg.get('quantiles', [0.05, 0.5, 0.95])
        self.label_col = global_cfg.get('label_column', 'label_return')

    def _extract_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_with_date = df.reset_index()
        features = [col for col in df.columns if col != self.label_col]
        X = df_with_date[['date'] + features]; y = df[self.label_col]
        return X, y

    def train_and_evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cached_data: dict = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict]:
        if cached_data:
            X_train_scaled = cached_data['X_train_scaled']
            y_train = cached_data['y_train']
            X_val_scaled = cached_data['X_val_scaled']
            y_val = cached_data['y_val']
        else: # 主要由 HPO 使用
            X_train, y_train = self._extract_xy(train_df)
            X_val, y_val = self._extract_xy(val_df)
            features_for_model = [col for col in X_train.columns if col != 'date']
            X_train_model, X_val_model = X_train[features_for_model], X_val[features_for_model]
            
            fold_scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(fold_scaler.fit_transform(X_train_model), index=X_train_model.index, columns=features_for_model)
            X_val_scaled = pd.DataFrame(fold_scaler.transform(X_val_model), index=X_val_model.index, columns=features_for_model)

        quantile_models = {}
        fold_stats = {}

        for q in self.quantiles:
            params = self.lgbm_params.copy()
            params['alpha'] = q
            model = lgb.LGBMRegressor(**params)
            
            early_stopping_rounds = self.lgbm_params.get('early_stopping_rounds', 100)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # 忽略此 with 块内发出的所有警告
                
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    eval_metric='quantile',
                    callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                )

            best_iteration = model.best_iteration_
            if self.verbose and best_iteration:
                # 注意：tqdm.write 应该在 build_models.py 中使用，这里我们暂时保留 print
                # 在 Builder 内部，我们不知道外部是否有 tqdm
                pass # 暂时关闭 fold 内部的打印，让外部的 tqdm 后缀来显示

            # 将最佳轮次信息存入状态字典，无论 verbose 如何
            if best_iteration:
                fold_stats[f'q_{q}_best_iter'] = best_iteration

            quantile_models[f'q_{q}'] = model
        
        median_model = quantile_models.get(f'q_{0.5}')
        ic_df = pd.DataFrame()
        oof_df = pd.DataFrame()

        if median_model and not y_val.empty and len(y_val) > 1:
            preds = median_model.predict(X_val_scaled)
            eval_df = pd.DataFrame({"y_pred": preds, "y_true": y_val.values, "date": y_val.index})
            oof_df = eval_df[['date', 'y_true', 'y_pred']]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fold_ic = eval_df['y_pred'].rank().corr(eval_df['y_true'].rank(), method='spearman')
                    if pd.notna(fold_ic):
                        ic_df = pd.DataFrame([{'date': eval_df['date'].max(), 'rank_ic': fold_ic}])
                except Exception: 
                    pass
                
        return {'models': quantile_models}, ic_df, oof_df, fold_stats

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

        metadata = {
            'feature_cols': features_for_model
        }
        return {'models': final_models, 'scaler': final_scaler, 'metadata': metadata}