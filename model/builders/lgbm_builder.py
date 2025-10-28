# 文件路径: model/builders/lgbm_builder.py

import warnings
import pandas as pd
import lightgbm as lgb
from typing import Any, Dict, Tuple
from lightgbm.callback import early_stopping
from sklearn.preprocessing import StandardScaler
from model.builders.base_builder import BaseBuilder, builder_registry

@builder_registry.register('lgbm')
class LGBMBuilder(BaseBuilder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        global_cfg = config.get('global_settings', {})

        default_params = config.get('model', {}).get('lgbm_params', {})
        hpo_fixed_params = config.get('hpo', {}).get('lgbm_hpo_config', {}).get('params', {})

        # 这里的 trial_params 是为了 HPO 流程准备的，它会覆盖其他同名参数
        trial_params = config.get('model', {}).get('lgbm_params', {})
        
        final_params = {**default_params, **hpo_fixed_params, **trial_params}
        final_params['random_state'] = global_cfg.get('seed', 42)
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

    def train_and_evaluate_fold(self, cached_data: dict = None, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None, **kwargs) -> Dict[str, Any]:
        if not cached_data:
            raise ValueError("LGBMBuilder requires 'cached_data'.")

        X_train_scaled = cached_data['X_train_scaled']
        y_train = cached_data['y_train']
        X_val_scaled = cached_data['X_val_scaled']
        y_val = cached_data['y_val']

        quantile_models = {}
        fold_stats = {}

        for q in self.quantiles:
            params = self.lgbm_params.copy()
            params['alpha'] = q
            model = lgb.LGBMRegressor(**params)
            
            early_stopping_rounds = self.lgbm_params.get('early_stopping_rounds', 100)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    eval_metric='quantile',
                    callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                )

            best_iteration = model.best_iteration_
            if best_iteration:
                fold_stats[f'q_{q}_best_iter'] = best_iteration

            quantile_models[f'q_{q}'] = model
        
        median_model = quantile_models.get(f'q_{0.5}')
        ic_df = pd.DataFrame()
        oof_df = pd.DataFrame()

        if median_model and not y_val.empty and len(y_val) > 1:
            preds = median_model.predict(X_val_scaled)
            # 使用 y_val 的索引（日期）来创建 eval_df
            eval_df = pd.DataFrame({"y_pred": preds, "y_true": y_val.values}, index=y_val.index)
            # oof_df 需要包含 date 列，所以从索引重置
            oof_df = eval_df.reset_index()[['date', 'y_true', 'y_pred']]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fold_ic = eval_df['y_pred'].rank().corr(eval_df['y_true'].rank(), method='spearman')
                    if pd.notna(fold_ic):
                        ic_df = pd.DataFrame([{'date': eval_df.index.max(), 'rank_ic': fold_ic}])
                except Exception: 
                    pass
                
        return {
            'artifacts': {'models': quantile_models},
            'ic_series': ic_df,
            'oof_preds': oof_df,
            'fold_stats': fold_stats
        }

    def train_final_model(self, full_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
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
        return {
            'model': final_models,
            'scaler': final_scaler,
            'metadata': metadata
        }