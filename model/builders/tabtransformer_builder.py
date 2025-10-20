import gc
import copy
import torch
import warnings
import pandas as pd
import torch.nn as nn
from tqdm.autonotebook import tqdm
from typing import Any, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

class TabTransformerBuilder:
    """
    TabTransformer 模型的完整构建器。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
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

        X_cont_tensor = torch.from_numpy(X_df[cont_features].values).float()
        X_cat_tensor = torch.from_numpy(X_df[cat_features].values).long()
        
        if y_series is not None:
            y_tensor = torch.from_numpy(y_series.values).float().unsqueeze(1)
            return X_cont_tensor, X_cat_tensor, y_tensor
        return X_cont_tensor, X_cat_tensor

    def train_and_evaluate_fold(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None, cached_data: dict = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict]:
        if not cached_data:
            raise ValueError("'cached_data' is required for this method.")

        X_train, y_train = cached_data['X_train_scaled'], cached_data['y_train']
        X_val, y_val = cached_data['X_val_scaled'], cached_data['y_val']

        if X_train.empty or X_val.empty:
            return {'model_state_dict': None, 'metadata': {}}, pd.DataFrame(), pd.DataFrame(), {}

        X_train_cont, X_train_cat, y_train_tensor = self._prepare_data_tensors(X_train, y_train)
        X_val_cont, X_val_cat, y_val_tensor = self._prepare_data_tensors(X_val, y_val)
        
        cat_features = self.model_params.get('categorical_features', ['day_of_week', 'month'])
        cat_dims = [int(X_train[col].max()) + 1 for col in cat_features]
        cont_features = [c for c in X_train.columns if c not in cat_features]

        model = TabTransformerModel(
            num_continuous=len(cont_features), cat_dims=cat_dims,
            dim=self.model_params.get('dim', 32), depth=self.model_params.get('depth', 4),
            heads=self.model_params.get('heads', 4), attn_dropout=self.model_params.get('dropout', 0.1),
            ff_dropout=self.model_params.get('dropout', 0.1)
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_params.get('learning_rate', 0.001))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        train_dataset = TensorDataset(X_train_cont, X_train_cat, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.model_params.get('batch_size', 128), shuffle=True)
        
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
            
            scheduler.step(val_loss)
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % self.verbose_period == 0:
                epoch_iterator.set_postfix(best_val_loss=f"{best_val_loss:.6f}")
            if patience_counter >= patience:
                break
        
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

        metadata = {'input_size_cont': len(cont_features), 'input_size_cat': len(cat_features), 'cat_dims': cat_dims, 'feature_cols': X_train.columns.tolist()}
        
        del model, train_loader, train_dataset
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()
            
        return {'model_state_dict': best_model_state, 'metadata': metadata}, ic_df, oof_df, fold_stats

    def train_final_model(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        label_col = self.label_col
        features = [col for col in full_df.columns if col != label_col and not col.startswith('future_')]
        
        X_full, y_full = full_df[features], full_df[label_col]
        
        final_scaler = StandardScaler()
        X_full_scaled = X_full.copy()
        X_full_scaled[:] = final_scaler.fit_transform(X_full)

        X_full_cont, X_full_cat, y_full_tensor = self._prepare_data_tensors(X_full_scaled, y_full)
        
        cat_features = self.model_params.get('categorical_features', ['day_of_week', 'month'])
        cat_dims = [int(X_full[col].max()) + 1 for col in cat_features]
        cont_features = [c for c in X_full.columns if c not in cat_features]

        model = TabTransformerModel(
            num_continuous=len(cont_features), cat_dims=cat_dims,
            dim=self.model_params.get('dim', 32), depth=self.model_params.get('depth', 4),
            heads=self.model_params.get('heads', 4), attn_dropout=self.model_params.get('dropout', 0.1),
            ff_dropout=self.model_params.get('dropout', 0.1)
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_params.get('learning_rate', 0.001))
        
        train_dataset = TensorDataset(X_full_cont, X_full_cat, y_full_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.model_params.get('batch_size', 128), shuffle=True)
        
        final_epochs = self.model_params.get('final_model_epochs', self.model_params.get('epochs', 100) // 2)
        
        model.train()
        for epoch in range(final_epochs):
            for X_batch_cont, X_batch_cat, y_batch in train_loader:
                X_batch_cont, X_batch_cat, y_batch = X_batch_cont.to(self.device), X_batch_cat.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch_cont, X_batch_cat)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        metadata = {'input_size_cont': len(cont_features), 'input_size_cat': len(cat_features), 'cat_dims': cat_dims, 'feature_cols': features}

        return {'model': model, 'scaler': final_scaler, 'metadata': metadata}