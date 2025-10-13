# 文件路径: model_builders/lstm_builder.py
'''
LSTM模型构建 (已修正数据泄露问题)
'''

import gc
import copy
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.autonotebook import tqdm
from typing import Any, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LSTMModel(nn.Module):
    """定义 PyTorch LSTM 模型结构。"""
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x); out = self.dropout1(out)
        out, (h_n, _) = self.lstm2(out)
        out = h_n.squeeze(0); out = self.dropout2(out)
        out = self.fc(out)
        return out

class LSTMBuilder:
    """LSTM 模型的构建器，现在适配全局预处理流程。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        global_cfg = config.get('global_settings', {})
        
        default_params = config.get('default_model_params', {}).get('lstm_params', {})
        hpo_params = config.get('hpo_config', {}).get('lstm_hpo_params', {}) # 为未来 HPO 预留
        
        self.lstm_params = {**default_params, **hpo_params}
        
        self.sequence_length = self.lstm_params.get('sequence_length', 60)
        self.label_col = global_cfg.get('label_column', 'label_return')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"INFO: PyTorch LSTMBuilder will use device: {self.device.upper()}")

    def _create_sequences(self, df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(公共辅助函数) 使用滑动窗口，将DataFrame转换为序列。现在由 Notebook 调用。"""
        xs, ys, dates = [], [], []
        # 确保索引是标准的 RangeIndex 以便 loc 操作
        df_reset = df.reset_index()
        
        if 'date' not in df_reset.columns:
            raise ValueError("'date' column not found after reset_index. Make sure the index has a name.")

        for i in range(len(df_reset) - self.sequence_length):
            x = df_reset.loc[i : i + self.sequence_length - 1, feature_cols].values
            label_index = i + self.sequence_length
            y = df_reset.loc[label_index, self.label_col]
            date = df_reset.loc[label_index, 'date']
            xs.append(x); ys.append(y); dates.append(date)
            
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(dates)

    def train_and_evaluate_fold(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None, cached_data: dict = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """
        (已重构) 接收预处理好的 Tensors，执行模型训练和评估，并返回 3 个值。
        """
        if not cached_data: raise ValueError("train_and_evaluate_fold now requires 'cached_data'.")

        # --- 从缓存中获取 Tensors ---
        X_train_tensor = cached_data['X_train_tensor'].to(self.device)
        y_train_tensor = cached_data['y_train_tensor'].to(self.device)
        X_val_tensor = cached_data['X_val_tensor'].to(self.device)
        y_val_tensor = cached_data['y_val_tensor'].to(self.device)
        y_val_seq = cached_data['y_val_seq']
        dates_val_seq = cached_data['dates_val_seq']

        if X_train_tensor.shape[0] == 0 or X_val_tensor.shape[0] == 0:
            # 确保即使提前退出，也返回 3 个值
            return {'model_state_dict': None}, pd.DataFrame(), pd.DataFrame()

        p = self.lstm_params
        batch_size = p.get('batch_size', 32)    # 默认 batch 大小为32
        num_workers = p.get('num_workers', 0)   # 默认为 0，保证在未配置时也能安全运行
        if num_workers > 0:
            print(f"INFO: DataLoader will use {num_workers} parallel workers.")

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,  # <--- 启用并行数据加载
            pin_memory=True,          # <--- 启用内存锁定以加速传输
            drop_last=True            # (可选) 如果最后一个 batch 不完整，则丢弃，可以稳定训练
        )
        
        model = LSTMModel(
            input_size=X_train_tensor.shape[2],
            hidden_size_1=p.get('units_1', 64),
            hidden_size_2=p.get('units_2', 32),
            dropout=p.get('dropout', 0.2)
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        scaler_amp = torch.amp.GradScaler(enabled=(self.device == 'cuda'))

        best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
        patience = p.get('early_stopping_rounds_lstm', 50)
        epochs = p.get('epochs', 100)
        verbose_period = p.get('verbose_period', 5)
        
        epoch_iterator = tqdm(range(epochs), desc="    - Epochs", leave=False)
        
        for epoch in epoch_iterator:
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
            
            scheduler.step(val_loss)
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item(); patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if verbose_period > 0 and (epoch + 1) % verbose_period == 0:
                # 动态更新前面的描述
                epoch_iterator.set_description(f"    - Epochs {epoch + 1}")
                # 更新后面的附加信息
                epoch_iterator.set_postfix(
                    total_epochs=epochs, 
                    best_val_loss=f"{best_val_loss:.6f}"
                )

            if patience_counter >= patience: break
        
        # print(f"    - Fold finished. Best validation loss: {best_val_loss:.6f} at epoch {epoch - patience_counter + 1}")

        ic_df = pd.DataFrame()
        oof_df = pd.DataFrame()

        if X_val_tensor.shape[0] > 0 and best_model_state:
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                preds = model(X_val_tensor).cpu().numpy().flatten()
            
            eval_df = pd.DataFrame({'y_pred': preds, 'y_true': y_val_seq, 'date': pd.to_datetime(dates_val_seq)})
            
            oof_df = eval_df[['date', 'y_true', 'y_pred']]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(eval_df) > 1:
                    try:
                        fold_ic = eval_df['y_pred'].rank().corr(eval_df['y_true'].rank(), method='spearman')
                        if pd.notna(fold_ic):
                            ic_df = pd.DataFrame([{'date': eval_df['date'].max(), 'rank_ic': fold_ic}])
                    except Exception: 
                        pass

        del model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, train_loader
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()
            
        # --- 核心修正 2：确保总是返回 3 个元素的元组 ---
        return {'model_state_dict': best_model_state}, ic_df, oof_df

    def train_final_model(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        """在全部数据上训练最终模型 (此函数保持独立，不使用缓存)。"""
        label_col = self.label_col
        features = [col for col in full_df.columns if col != label_col]
        
        final_scaler = StandardScaler()
        full_df_scaled = full_df.copy()
        full_df_scaled.index.name = 'date' # 确保索引有名字
        full_df_scaled[features] = final_scaler.fit_transform(full_df[features])

        X_full, y_full, _ = self._create_sequences(full_df_scaled, features)
        if len(X_full) == 0:
            raise ValueError("Cannot train final model with no data sequences.")

        p = self.lstm_params
        batch_size = p.get('batch_size', 32)
        train_dataset = TensorDataset(torch.from_numpy(X_full), torch.from_numpy(y_full).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = LSTMModel(
            input_size=X_full.shape[2],
            hidden_size_1=p.get('units_1', 64),
            hidden_size_2=p.get('units_2', 32),
            dropout=p.get('dropout', 0.2)
        ).to(self.device)
        
        criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
        
        final_epochs = max(1, int(p.get('epochs', 100) * 0.5))
        for epoch in tqdm(range(final_epochs), desc="    - Final Model Epochs", leave=False):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch); loss = criterion(outputs, y_batch)
                loss.backward(); optimizer.step()

        return {'model': model, 'scaler': final_scaler}