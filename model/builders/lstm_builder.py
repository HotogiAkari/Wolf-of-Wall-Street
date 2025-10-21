# 文件路径: model_builders/lstm_builder.py
'''
LSTM模型构建
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
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

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
    """LSTM 模型的构建器，适配全局预处理流程。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        global_cfg = config.get('global_settings', {})
        
        default_params = config.get('default_model_params', {}).get('lstm_params', {})
        hpo_params = config.get('hpo_config', {}).get('lstm_hpo_config', {}).get('params', {})
        self.lstm_params = {**default_params, **hpo_params}
        
        self.sequence_length = self.lstm_params.get('sequence_length', 60)
        self.label_col = global_cfg.get('label_column', 'label_return')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.verbose_period = self.lstm_params.get('verbose_period', 5)
        self.verbose = self.verbose_period > 0
        
        if self.verbose:
            print(f"INFO: PyTorch LSTMBuilder initialized with device: {self.device.upper()}")
            num_workers = self.lstm_params.get('num_workers', 0)
            if num_workers > 0: print(f"INFO: DataLoader will use {num_workers} parallel workers.")
            
            precision = self.lstm_params.get('precision', 32)
            use_amp = (precision == 16) and (self.device == 'cuda')
            amp_status = "ENABLED (float16)" if use_amp else "DISABLED (float32)"
            print(f"INFO: Automatic Mixed Precision (AMP) is {amp_status}.")

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

    def train_and_evaluate_fold(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None, cached_data: dict = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict]:
        if not cached_data:
            raise ValueError("'cached_data' is required for this method.")

        X_train_tensor, y_train_tensor = cached_data['X_train_tensor'], cached_data['y_train_tensor']
        X_val_tensor, y_val_tensor = cached_data['X_val_tensor'], cached_data['y_val_tensor']
        y_val_seq, dates_val_seq = cached_data['y_val_seq'], cached_data['dates_val_seq']

        if X_train_tensor.shape[0] == 0 or X_val_tensor.shape[0] == 0:
            return {'model_state_dict': None, 'metadata': {}}, pd.DataFrame(), pd.DataFrame(), {}
        
        if not torch.all(torch.isfinite(X_train_tensor)) or not torch.all(torch.isfinite(y_train_tensor)):
            print("FATAL: 训练数据或标签中包含 inf/nan！跳过此 fold。")
            return {'model_state_dict': None, 'metadata': {}}, pd.DataFrame(), pd.DataFrame(), {}

        p = self.lstm_params
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=p.get('batch_size', 128), shuffle=True,
            num_workers=p.get('num_workers', 0), pin_memory=(self.device=='cuda'), drop_last=True
        )
        
        model = LSTMModel(
            input_size=X_train_tensor.shape[2],
            hidden_size_1=p.get('units_1', 64),
            hidden_size_2=p.get('units_2', 32),
            dropout=p.get('dropout', 0.2)
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
        
        # --- (核心修改) 手动实现 ChainedScheduler 逻辑 ---
        warmup_steps = p.get('warmup_steps', 5)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', 
            patience=p.get('plateau_patience', 15),
            factor=p.get('plateau_factor', 0.5),
        )
        
        use_amp = (p.get('precision', 32) == 16) and (self.device == 'cuda')
        scaler_amp = torch.amp.GradScaler(enabled=use_amp)

        best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
        patience = p.get('early_stopping_rounds_lstm', 25)
        epochs = p.get('epochs', 150)
        
        epoch_iterator = tqdm(range(epochs), desc="    - Epochs (LSTM)", leave=False, disable=not self.verbose)
        
        for epoch in epoch_iterator:
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                if not torch.isfinite(loss):
                    print(f"FATAL: Epoch {epoch}, 训练损失变为 {loss.item()}！训练提前终止。")
                    patience_counter = patience
                    break

                if use_amp:
                    scaler_amp.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            if patience_counter >= patience:
                break

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
                    val_outputs = model(X_val_tensor.to(self.device))
                    val_loss = criterion(val_outputs, y_val_tensor.to(self.device))
                
                if not torch.isfinite(val_loss):
                    print(f"FATAL: Epoch {epoch}, 验证损失变为 {val_loss.item()}！")
            
            # --- (核心修改) 手动管理学习率调度 ---
            if epoch < warmup_steps:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(val_loss.item())
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % self.verbose_period == 0:
                current_lr = optimizer.param_groups[0]['lr']
                epoch_iterator.set_postfix(best_val_loss=f"{best_val_loss:.6f}", lr=f"{current_lr:.1e}")

            if patience_counter >= patience:
                if self.verbose: tqdm.write(f"    - INFO: 早停机制已在第 {epoch + 1} 轮触发。")
                break
        
        epoch_finished = epoch - patience_counter + 1 if best_val_loss != float('inf') else 0
        if self.verbose: tqdm.write(f"    - Fold finished. Best validation loss: {best_val_loss:.6f} at epoch {epoch_finished}")

        ic_df, oof_df, fold_stats = pd.DataFrame(), pd.DataFrame(), {}
        if best_val_loss != float('inf'):
            fold_stats['best_loss'] = f"{best_val_loss:.6f}"
        
        metadata = {'input_size': X_train_tensor.shape[2], 'feature_cols': cached_data.get('feature_cols')}
        
        if X_val_tensor.shape[0] > 0 and best_model_state:
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
                    preds = model(X_val_tensor.to(self.device)).cpu().numpy().flatten()
            
            eval_df = pd.DataFrame({'y_pred': preds, 'y_true': y_val_seq, 'date': pd.to_datetime(dates_val_seq)})
            oof_df = eval_df[['date', 'y_true', 'y_pred']]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_ic = eval_df['y_pred'].rank().corr(eval_df['y_true'].rank(), method='spearman')
                if pd.notna(fold_ic):
                    ic_df = pd.DataFrame([{'date': eval_df['date'].max(), 'rank_ic': fold_ic}])

        del model, train_loader, train_dataset
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()
            
        return {'model_state_dict': best_model_state, 'metadata': metadata}, ic_df, oof_df, fold_stats
    
    def train_final_model(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        """
        在全部数据上训练最终的生产模型。
        """
        print(f"    - INFO: Starting final LSTM model training on all data...")
        
        features = [col for col in full_df.columns if col != self.label_col and not col.startswith('future_')]
        
        # 使用与 L3 缓存一致的手动、稳健标准化
        final_mean = full_df[features].mean()
        final_std = full_df[features].std() + 1e-8
        
        final_scaler = StandardScaler()
        final_scaler.mean_ = final_mean.values
        final_scaler.scale_ = final_std.values
        
        full_df_scaled = full_df.copy()
        full_df_scaled[features] = (full_df[features] - final_mean) / final_std

        X_full, y_full, _ = self._create_sequences(full_df_scaled.reset_index(), features)
        if len(X_full) == 0:
            raise ValueError("无法为最终模型创建任何数据序列。")

        p = self.lstm_params
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_full).float(), 
            torch.from_numpy(y_full).float().unsqueeze(1)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p.get('batch_size', 128), shuffle=True)
        
        model = LSTMModel(
            input_size=X_full.shape[2], hidden_size_1=p.get('units_1', 64),
            hidden_size_2=p.get('units_2', 32), dropout=p.get('dropout', 0.2)
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
        
        final_epochs = p.get('final_model_epochs', max(1, p.get('epochs', 100) // 2))
        if self.verbose:
            print(f"    - INFO: Training final model for a fixed {final_epochs} epochs.")
            
        final_epoch_iterator = tqdm(range(final_epochs), desc="    - Final Epochs", leave=False, disable=not self.verbose)

        model.train()
        for epoch in final_epoch_iterator:
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metadata = {'input_size': X_full.shape[2], 'feature_cols': features}

        return {'model': model, 'scaler': final_scaler, 'metadata': metadata}