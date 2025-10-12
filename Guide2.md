### **【Phoenix项目】最终版工程实施白皮书 (模块化扩展版)**

**版本：Final (Modular Expansion Blueprint)**
**核心使命：** 在**不改变现有稳定数据流** (`get_data.py`, `save_data.py`) 和**核心协调器 (`train.ipynb`)** 的前提下，通过模块化扩展的方式，将深度学习模型（LSTM）无缝集成到现有 LightGBM 的训练与预测流程中。

---

### **第一部分：最终文件结构与开发路线图**

#### **1.1 最终系统文件结构**

```
quant_system_final/
├── configs/
│   └── system_config.yaml
├── data/
│   ├── get_data.py                # (保持不变) 核心数据获取与特征工程
│   └── save_data.py               # (保持不变) 数据管道协调器
├── model_builders/                # V-Final 新增目录：存放具体模型实现
│   ├── __init__.py
│   ├── base_builder.py            # (可选) 定义一个抽象基类
│   ├── lgbm_builder.py            # 模块 1: LGBM 模型实现
│   └── lstm_builder.py            # 模块 2: LSTM 模型实现
├── model/
│   └── build_models.py            # 模块 3: 扩充后的模型训练总协调器
├── trading/
│   └── (其他模块保持不变)
├── train.ipynb                    # 核心 Notebook: 只需微小改动
└── prophet.ipynb                  # 核心 Notebook: 只需微小改动
```

#### **1.2 建议开发顺序 (符合依赖关系)**

1.  **模块 1: `model_builders/lgbm_builder.py`**
    *   **任务**: 将现有 `build_models.py` 中的 **LGBM 专属训练逻辑**剥离出来，封装成一个独立的构建器。
2.  **模块 2: `model_builders/lstm_builder.py`**
    *   **任务**: 创建一个全新的 LSTM 构建器，使其遵循与 `lgbm_builder.py` 相同的接口和规范。
3.  **模块 3: `model/build_models.py`**
    *   **任务**: **重构**这个文件，使其从一个“LGBM专用训练脚本”升格为“**模型训练的总协调器**”。它将动态地调用不同的构建器来完成具体模型的训练。
4.  **最终集成**: 对 `train.ipynb` 进行微小调整，以新的方式调用 `build_models.py`。

---

### **第二部分：核心模块开发任务书**

#### **第1章：`model_builders/lgbm_builder.py` - LGBM 模型构建器**

*   **目标**: 封装所有与 LightGBM 模型**具体实现**相关的逻辑。
*   **核心类**: `LGBMBuilder`
*   **关键实现**:
    ```python
    # model_builders/lgbm_builder.py
    import lightgbm as lgb
    
    class LGBMBuilder:
        def __init__(self, config):
            self.config = config

        def prepare_data(self, df):
            """LGBM使用原始的DataFrame，无需特殊处理。"""
            return df

        def train_and_evaluate_fold(self, train_df, val_df):
            """
            在一个 Walk-Forward fold 上训练并评估 LGBM 模型。
            """
            # 提取 X, y
            X_train, y_train = self._extract_xy(train_df)
            X_val, y_val = self._extract_xy(val_df)
            
            # ... (此处是原 build_models.py 中关于特征筛选、标签归一化的逻辑) ...

            # ... (此处是原 build_models.py 中关于训练分位数模型的核心逻辑) ...
            # for q in self.config['quantiles']:
            #     model = lgb.LGBMRegressor(...)
            #     model.fit(...)

            # ... (此处是原 build_models.py 中关于评估IC和准确率的逻辑) ...
            
            # 返回训练好的模型（或模型字典）和该 fold 的评估结果
            return trained_models, daily_ic_series
    ```

#### **第2章：`model_builders/lstm_builder.py` - LSTM 模型构建器**

*   **目标**: 封装所有与 LSTM 模型**具体实现**相关的逻辑。
*   **核心类**: `LSTMBuilder`
*   **关键实现**:
    ```python
    # model_builders/lstm_builder.py
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers

    class LSTMBuilder:
        def __init__(self, config):
            self.config = config
            self.sequence_length = config.get('lstm_sequence_length', 60)

        def _create_sequences(self, df, feature_cols, label_col):
            # ... (实现滑动窗口，将DataFrame转换为序列格式) ...
            return np.array(xs), np.array(ys)

        def prepare_data(self, df):
            """LSTM需要将数据转换为序列格式。"""
            feature_cols = # ...
            label_col = 'label_return'
            return self._create_sequences(df, feature_cols, label_col)

        def _create_model(self, input_shape):
            # ... (定义Keras/PyTorch模型结构) ...
            return model

        def train_and_evaluate_fold(self, train_data, val_data):
            """

            在一个 Walk-Forward fold 上训练并评估 LSTM 模型。
            """
            X_train, y_train = train_data
            X_val, y_val = val_data

            model = self._create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            early_stopping = keras.callbacks.EarlyStopping(...)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping], ...)
            
            # ... (评估模型在 val_data 上的 IC 和准确率) ...
            
            # 返回训练好的模型和该 fold 的评估结果
            return model, daily_ic_series
    ```

#### **第3章：`model/build_models.py` - 模型训练总协调器 (重构)**

*   **目标**: 成为一个与具体模型实现无关的、**通用的 Walk-Forward 训练引擎**。
*   **入口函数**: `run_training_for_ticker(ticker: str, model_type: str, config: dict, force_retrain: bool = False)`
*   **内部实现**:
    ```python
    # model/build_models.py
    from data import get_data # 复用现有的数据加载
    from model_builders.lgbm_builder import LGBMBuilder
    from model_builders.lstm_builder import LSTMBuilder
    
    def run_training_for_ticker(ticker, model_type, config, force_retrain=False):
        print(f"--- Starting {model_type.upper()} training for {ticker} ---")
        
        # 1. 根据 model_type 选择并实例化对应的构建器
        if model_type.lower() == 'lgbm':
            builder = LGBMBuilder(config)
        elif model_type.lower() == 'lstm':
            builder = LSTMBuilder(config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 2. 加载数据 (这是所有模型共享的)
        df = get_data.get_full_feature_df_for_ticker(ticker, config) # 假设有这样一个函数
        
        # 3. 执行通用的 Walk-Forward 交叉验证循环
        all_fold_ics = []
        # _walk_forward_split 逻辑现在位于此文件中，是通用的
        for train_df, val_df in _walk_forward_split(df):
            
            # 4. 让构建器自己准备数据
            train_data = builder.prepare_data(train_df)
            val_data = builder.prepare_data(val_df)
            
            # 5. 调用构建器执行 fold 内的训练和评估
            model_fold, ic_series_fold = builder.train_and_evaluate_fold(train_data, val_data)
            
            all_fold_ics.append(ic_series_fold)

        # 6. 汇总IC历史并保存
        full_ic_history = pd.concat(all_fold_ics)
        # ... (将 full_ic_history 保存到 ic_history.csv 中) ...

        # 7. 在全部数据上训练最终模型并保存
        full_data = builder.prepare_data(df)
        final_model = builder.train_final_model(full_data) # 假设构建器有此方法
        # ... (保存 final_model) ...

        print(f"--- SUCCESS: {model_type.upper()} training for {ticker} completed. ---")
    ```

---

### **第三部分：核心 Notebook 工作流 (微调)**

#### **`train.ipynb` (模型训练协调器)**

您的 `train.ipynb` 现在变得极其**简洁和清晰**。

```python
# (在主训练单元格中)
import build_models

for stock_config in tqdm(config['stocks_to_process'], desc="Overall Training Progress"):
    ticker = stock_config['ticker']
    
    # --- 依次调用总协调器来训练不同的模型 ---
    
    # 训练 LGBM 模型
    build_models.run_training_for_ticker(
        ticker=ticker, 
        model_type='lgbm', 
        config=config, 
        force_retrain=FORCE_RETRAIN
    )
    
    # 训练 LSTM 模型
    build_models.run_training_for_ticker(
        ticker=ticker, 
        model_type='lstm', 
        config=config, 
        force_retrain=FORCE_RETRAIN
    )
```

**结论**: 这个方案完美地实现了您的目标。它通过引入一个**抽象层**（`model_builders/`），将**“做什么”（Walk-Forward验证）**和**“怎么做”（具体模型的实现）**完美地分离开来。这使得您的系统极易扩展——未来若想加入`TCN`或`Transformer`模型，您只需在 `model_builders/` 目录下新增一个文件，并在 `build_models.py` 中增加一个 `elif` 分支即可，而无需改动任何核心的验证逻辑和 Notebook 流程。