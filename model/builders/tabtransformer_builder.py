# 文件路径: model/builder/tabtransformer_builder.py

import gc
import copy
import torch
import warnings
import pandas as pd
import torch.nn as nn
from tqdm.autonotebook import tqdm
from typing import Any, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from model.builders.base_builder import BaseBuilder
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.encoding_utils import encode_categorical_features

class TabTransformerModel(nn.Module):
    """
    一个简化的、自包含的 TabTransformer 模型结构。
    """
    def __init__(self, num_continuous, cat_dims, dim=32, depth=4, heads=4, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        
        if not cat_dims:
            raise ValueError("TabTransformerModel 必须至少有一个类别特征。")

        # --- Categorical Embeddings ---
        self.cat_embeds = nn.ModuleList([nn.Embedding(c_dim, dim) for c_dim in cat_dims])
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dropout=attn_dropout, 
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- MLP Head ---
        # 计算 MLP 的输入维度
        mlp_input_dim = (len(cat_dims) * dim) + num_continuous
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 2, 1)
        )

    def forward(self, x_cont, x_cat):
        # 1. Process categorical features
        cat_embs = [embed_layer(x_cat[:, i]) for i, embed_layer in enumerate(self.cat_embeds)]
        x_cat_processed = torch.stack(cat_embs, 1)
        
        # 2. Pass categorical embeddings through Transformer
        x_transformer_out = self.transformer(x_cat_processed)
        x_transformer_out = x_transformer_out.flatten(1)

        # 3. Concatenate and pass to MLP head
        x_combined = torch.cat([x_transformer_out, x_cont], dim=1)
        return self.mlp(x_combined)

class TabTransformerBuilder(BaseBuilder):
    """
    TabTransformer 模型的完整构建器。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        global_cfg = config.get('global_settings', {})
        
        default_params = config.get('default_model_params', {}).get('tabtransformer_params', {})
        hpo_params = config.get('hpo_config', {}).get('tabtransformer_hpo_config', {}).get('params', {})
        self.model_params = {**default_params, **hpo_params}
        
        self.label_col = global_cfg.get('label_column', 'label_return')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose_period = self.model_params.get('verbose_period', 5)
        self.verbose = self.verbose_period > 0
        
        print(f"INFO: PyTorch TabTransformerBuilder initialized with device: {self.device.upper()}")

    def _prepare_data_tensors(self, X_df: pd.DataFrame, y_series: pd.Series = None):
        """
        将输入的 DataFrame 和 Series 转换为 TabTransformer 所需的 Tensors。
        """
        cat_features = self.model_params.get('categorical_features', ['day_of_week', 'month'])
        
        # 确保所有类别特征都存在
        missing_cats = [c for c in cat_features if c not in X_df.columns]
        if missing_cats:
            raise ValueError(f"DataFrame 中缺少必要的类别特征: {missing_cats}")
            
        cont_features = [c for c in X_df.columns if c not in cat_features]

        X_cont_tensor = torch.from_numpy(X_df[cont_features].values.copy()).float()
        X_cat_tensor = torch.from_numpy(X_df[cat_features].values.copy()).long()
        
        if y_series is not None:
            y_tensor = torch.from_numpy(y_series.values.copy()).float().unsqueeze(1)
            return X_cont_tensor, X_cat_tensor, y_tensor
        return X_cont_tensor, X_cat_tensor

    def train_and_evaluate_fold(self, cached_data: dict, **kwargs) -> Dict[str, Any]:
        if not cached_data:
            raise ValueError("'cached_data' is required for this method.")

        # --- 1. 直接从缓存加载预处理好的 Tensors ---
        X_train_cont, X_train_cat, y_train_tensor = cached_data['X_train_cont'], cached_data['X_train_cat'], cached_data['y_train_tensor']
        X_val_cont, X_val_cat, y_val_tensor = cached_data['X_val_cont'], cached_data['X_val_cat'], cached_data['y_val_tensor']
        y_val = cached_data['y_val']
        cat_dims = cached_data['cat_dims']
        
        if X_train_cont.shape[0] == 0:
            return {'model_state_dict': None, 'metadata': {}}, pd.DataFrame(), pd.DataFrame(), {}
        
        # --- 2. 初始化模型、损失函数、优化器 ---
        model = TabTransformerModel(
            num_continuous=X_train_cont.shape[1],
            cat_dims=cat_dims,
            dim=self.model_params.get('dim', 32),
            depth=self.model_params.get('depth', 4),
            heads=self.model_params.get('heads', 4),
            attn_dropout=self.model_params.get('dropout', 0.1),
            ff_dropout=self.model_params.get('dropout', 0.1)
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_params.get('learning_rate', 0.001))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        train_dataset = TensorDataset(X_train_cont, X_train_cat, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.model_params.get('batch_size', 256), shuffle=True)
        
        # --- 3. 执行训练循环 ---
        best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
        patience = self.model_params.get('early_stopping_rounds', 20)
        epochs = self.model_params.get('epochs', 100)
        
        epoch_iterator = tqdm(range(epochs), desc="    - Epochs (TabTransformer)", leave=False, disable=not self.verbose)

        for epoch in epoch_iterator:
            model.train()
            for X_batch_cont, X_batch_cat, y_batch in train_loader:
                X_batch_cont, X_batch_cat, y_batch = X_batch_cont.to(self.device), X_batch_cat.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch_cont, X_batch_cat)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_cont.to(self.device), X_val_cat.to(self.device))
                val_loss = criterion(val_outputs, y_val_tensor.to(self.device))
            
            scheduler.step(val_loss.item())
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % self.verbose_period == 0:
                epoch_iterator.set_postfix(best_val_loss=f"{best_val_loss:.6f}")
            if patience_counter >= patience:
                if self.verbose: tqdm.write(f"    - INFO: 早停机制已在第 {epoch + 1} 轮触发。")
                break

        p = self.model_params
        model_artifacts = {
            'model_state_dict': best_model_state,
            'metadata': {
                'cat_dims': cat_dims,
                'feature_cols': cached_data.get('feature_cols'),
                'model_structure': {
                    'num_continuous': X_train_cont.shape[1],
                    'dim': p.get('dim', 32),
                    'depth': p.get('depth', 4),
                    'heads': p.get('heads', 4),
                    'attn_dropout': p.get('dropout', 0.1),
                    'ff_dropout': p.get('dropout', 0.1)
                }
            }
        }

        # --- 4. 处理并返回结果 ---
        ic_df, oof_df, fold_stats = pd.DataFrame(), pd.DataFrame(), {}
        if best_model_state:
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                preds = model(X_val_cont.to(self.device), X_val_cat.to(self.device)).cpu().numpy().flatten()
            
            eval_df = pd.DataFrame({"y_pred": preds, "y_true": y_val.values}, index=y_val.index)
            oof_df = eval_df.reset_index()[['date', 'y_true', 'y_pred']]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_ic = eval_df['y_pred'].rank().corr(eval_df['y_true'].rank(), method='spearman')
                if pd.notna(fold_ic):
                    ic_df = pd.DataFrame([{'date': eval_df.index.max(), 'rank_ic': fold_ic}])
            
            fold_stats['best_loss'] = f"{best_val_loss:.6f}"

        metadata = {'cat_dims': cat_dims} # 保存 cat_dims 以备后用
        
        del model, train_loader, train_dataset
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()
            
        return {
            'artifacts': model_artifacts,
            'ic_series': ic_df,
            'oof_preds': oof_df,
            'fold_stats': fold_stats
        }

    def train_final_model(self, full_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        在全部数据上训练最终的生产模型。
        """
        print(f"    - INFO: Starting final TabTransformer model training...")

        # --- 1. 特征和标签分离 ---
        features = [col for col in full_df.columns if col != self.label_col and not col.startswith('future_')]
        X_full, y_full = full_df[features], full_df[self.label_col]
        
        cat_features = self.model_params.get('categorical_features', ['day_of_week', 'month'])
        cont_features = [c for c in X_full.columns if c not in cat_features]

        # --- 2. 类别特征编码 ---
        # 我们需要一个虚拟的 df_val 来满足 _encode_categorical_features 的接口
        X_full_encoded, _, encoders = encode_categorical_features(X_full.copy(), X_full.head(1).copy(), cat_features)
        
        # --- 3. 只对连续特征进行标准化 ---
        final_scaler = StandardScaler()
        # fit_transform 只在连续特征上进行
        X_full_encoded[cont_features] = final_scaler.fit_transform(X_full_encoded[cont_features])

        # --- 4. 准备 Tensors ---
        X_full_cont, X_full_cat, y_full_tensor = self._prepare_data_tensors(X_full_encoded, y_full)
        
        # --- 5. 正确计算 cat_dims ---
        # 从编码后的数据中获取正确的类别数量
        cat_dims = [len(encoders[col].classes_) for col in cat_features]

        # --- 6. 模型训练 ---
        p = self.model_params
        model = TabTransformerModel(
            num_continuous=len(cont_features),
            cat_dims=cat_dims, # <-- 使用正确的 cat_dims
            dim=self.model_params.get('dim', 32),
            depth=self.model_params.get('depth', 4),
            heads=self.model_params.get('heads', 4),
            attn_dropout=self.model_params.get('dropout', 0.1),
            ff_dropout=self.model_params.get('dropout', 0.1)
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_params.get('learning_rate', 0.001))
        
        train_dataset = TensorDataset(X_full_cont, X_full_cat, y_full_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.model_params.get('batch_size', 256), shuffle=True)
        
        final_epochs = self.model_params.get('final_model_epochs', 50)
        
        model.train()
        for epoch in range(final_epochs):
            for X_batch_cont, X_batch_cat, y_batch in train_loader:
                X_batch_cont, X_batch_cat, y_batch = X_batch_cont.to(self.device), X_batch_cat.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch_cont, X_batch_cat)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        print("    - SUCCESS: Final TabTransformer model training complete.")

        p = self.model_params
        metadata = {
            'feature_cols': features,
            'cat_features': cat_features,
            'cont_features': cont_features,
            'cat_dims': cat_dims, 
            'model_structure': {
                'num_continuous': len(cont_features),
                'dim': p.get('dim', 32),
                'depth': p.get('depth', 4),
                'heads': p.get('heads', 4),
                'attn_dropout': p.get('dropout', 0.1),
                'ff_dropout': p.get('dropout', 0.1)
            }
        }

        # (核心修改) 将返回值包装成标准化的字典
        return {
            'model': model,
            'scaler': final_scaler,
            'metadata': metadata,
            'encoders': encoders
        }