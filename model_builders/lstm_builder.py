# 文件路径: model_builders/lstm_builder.py
'''
LSTM模型构建 (已修正数据泄露问题)
'''

import gc
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple

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
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, (h_n, _) = self.lstm2(out)
        out = h_n.squeeze(0)
        out = self.dropout2(out)
        out = self.fc(out)
        return out

class LSTMBuilder:
    """LSTM 模型的构建器，封装了数据准备、训练、评估和最终模型生成的完整流程。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        global_cfg = config.get('global_settings', {})
        self.lstm_params = config.get('lstm_params', {})
        self.sequence_length = self.lstm_params.get('sequence_length', 60)
        self.label_col = global_cfg.get('label_column', 'label_return')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"INFO: PyTorch LSTMBuilder will use device: {self.device.upper()}")
        # 核心修正：移除 self.scaler

    def _create_sequences(self, df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用滑动窗口，将DataFrame转换为对齐的 (X, y, dates) 序列。"""
        xs, ys, dates = [], [], []
        df_reset = df.reset_index()

        for i in range(len(df_reset) - self.sequence_length):
            x = df_reset.loc[i:(i + self.sequence_length - 1), feature_cols].values
            label_index = i + self.sequence_length
            y = df_reset.loc[label_index, self.label_col]
            date = df_reset.loc[label_index, 'date']
            xs.append(x)
            ys.append(y)
            dates.append(date)
            
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(dates)

    def train_and_evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """在一个 Walk-Forward fold 上训练并评估 LSTM 模型 (含进度条和精简输出)。"""
        features = [col for col in train_df.columns if col != self.label_col]
        
        fold_scaler = StandardScaler()
        train_df_scaled = train_df.copy()
        train_df_scaled[features] = fold_scaler.fit_transform(train_df[features])
        
        val_df_scaled = val_df.copy()
        if not val_df.empty:
            val_df_scaled[features] = fold_scaler.transform(val_df[features])

        X_train, y_train, _ = self._create_sequences(train_df_scaled, features)
        X_val, y_val_seq, dates_val_seq = self._create_sequences(val_df_scaled, features)

        if len(X_train) == 0 or len(X_val) == 0:
            return {'model_state_dict': None}, pd.DataFrame()

        p = self.lstm_params
        batch_size = p.get('batch_size', 32)
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        X_val_tensor = torch.from_numpy(X_val).to(self.device)
        y_val_tensor = torch.from_numpy(y_val_seq).unsqueeze(1).to(self.device)

        model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size_1=p.get('units_1', 64),
            hidden_size_2=p.get('units_2', 32),
            dropout=p.get('dropout', 0.2)
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
        patience = self.config.get('global_settings', {}).get('early_stopping_rounds_lstm', 15)
        
        # --- 核心修正：添加 tqdm 进度条 ---
        epochs = p.get('epochs', 100)
        epoch_iterator = tqdm(range(epochs), desc="Epochs", leave=False)
        
        for epoch in epoch_iterator:
            model.train()
            # Mini-batch 循环可以不加进度条，以保持输出简洁
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
            
            scheduler.step(val_loss)
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            # --- 核心修正：更新进度条的描述信息 ---
            epoch_iterator.set_postfix(best_val_loss=f"{best_val_loss:.6f}", patience=f"{patience_counter}/{patience}")

            if patience_counter >= patience:
                break # 不再打印早停信息，因为进度条已经显示了
        
        # --- 核心修正：在 fold 结束后，打印一次最终的最佳损失 ---
        print(f"    - Fold finished. Best validation loss: {best_val_loss:.6f}")

        if best_model_state:
            model.load_state_dict(best_model_state)

        daily_ic_df = pd.DataFrame() # <-- 名字保留，但现在是 Fold IC
        
        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_val_tensor).cpu().numpy().flatten()
            
            eval_df = pd.DataFrame({'pred': preds, 'y': y_val_seq, 'date': pd.to_datetime(dates_val_seq)})
            
            # --- 核心修正：在整个验证集上计算 IC，移除 groupby ---
            if len(eval_df) > 1:
                fold_ic = eval_df['pred'].rank().corr(eval_df['y'].rank(), method='spearman')
                
                if pd.notna(fold_ic):
                    last_date = eval_df['date'].max()
                    daily_ic_df = pd.DataFrame([{'date': last_date, 'rank_ic': fold_ic}])
            # --- 修正结束 ---

        del model, X_val_tensor, y_val_tensor, train_loader, train_dataset, X_train, y_train, X_val
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()
            
        return {'model_state_dict': best_model_state}, daily_ic_df

    def train_final_model(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        """在全部数据上训练最终模型。"""
        features = [col for col in full_df.columns if col != self.label_col]

        # --- 核心修正：创建并训练最终的 scaler ---
        final_scaler = StandardScaler()
        full_df_scaled = full_df.copy()
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
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
        
        final_epochs = max(1, int(p.get('epochs', 100) * 0.5))
        for epoch in range(final_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Final training, epoch {epoch+1}/{final_epochs}, loss: {loss.item():.6f}")

        return {'model': model, 'scaler': final_scaler}