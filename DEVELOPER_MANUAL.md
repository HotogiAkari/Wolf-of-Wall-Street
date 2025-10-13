# 量化研究与预测框架 - 开发者技术手册

**版本**: 3.0 (Q4 2025)
**状态**: 稳定 (Stable)
**维护者**: [您的名字/团队名]

---

## 1. 架构总览与设计哲学

### 1.1 系统目标

本框架旨在提供一个从数据采集到模型预测的、端到端的、工业级解决方案，用于量化 Alpha 策略的研究与部署。其核心目标是实现**快速迭代**、**结果可复现**、**过程可追溯**以及**决策稳健性**。

### 1.2 核心设计哲学

*   **关注点分离 (Separation of Concerns, SoC)**: 系统的每一部分都应只负责一件事情，并通过明确的接口与其他部分交互。我们将系统垂直划分为**数据层**、**模型层**和**应用层**；水平划分为**配置**与**逻辑**。
*   **配置即代码 (Configuration as Code)**: 所有的实验变量（参数、股票池、日期）都通过唯一的 YAML 配置文件 (`config.yaml`) 进行管理。这使得任何实验都可以被版本控制工具（如 Git）追踪，实现了完全的可复现性。
*   **不可变数据模型 (Immutable Data Model)**: 受函数式编程启发，数据处理流水线被设计为一系列转换。一旦特征数据被生成并缓存（L2/L3 Cache），它在本次实验生命周期中应被视为不可变的，后续的模型训练流程只读取而不修改这些数据。
*   **防御性编程 (Defensive Programming)**: 系统的每个关键节点，从数据校验到模型融合，都内置了对异常输入的检查和安全回退机制，以最大限度地保证在真实、嘈杂的金融数据环境下的稳定运行。

### 1.3 宏观架构图

```
+--------------------------+      +-------------------------+      +------------------------+
|      执行层 (IPYNB)      |----->|      模型层 (.py)       |----->|      数据层 (.py)      |
|  (train / predict)       |      |   (build/hpo/fuse)      |      | (get/save/contracts)   |
+--------------------------+      +-------------------------+      +------------------------+
             ^                                  ^                            ^
             |                                  |                            |
             |      Reads                      |       Reads                |      Reads
             |                                  |                            |
+------------------------------------------------------------------------------------------+
|                                     配置层 (YAML)                                        |
|                                     config.yaml                                        |
+------------------------------------------------------------------------------------------+
```

### 1.4 数据流水线与缓存策略

本框架采用三级缓存策略，以在效率和灵活性之间取得最佳平衡：

*   **L1 Cache (原始数据缓存)**:
    *   **位置**: `data_cache/raw_ohlcv/`
    *   **格式**: Pickle (`.pkl`)
    *   **内容**: 从 Baostock API 直接下载的、未经任何处理的原始 OHLCV 数据。
    *   **作用**: 避免对外部 API 的重复请求，应对网络问题和速率限制。

*   **L2 Cache (特征数据缓存)**:
    *   **位置**: `data/processed/{ticker}/{date_range}/`
    *   **格式**: Pickle (`.pkl`)
    *   **内容**: 经过完整特征工程（技术指标、日历、相对强度等）的、干净的特征数据集。
    *   **作用**: 隔离数据准备与模型训练。一旦生成，模型实验可以反复、快速地从此加载数据，无需重新计算特征。文件名包含**配置哈希**，确保了特征集与其生成逻辑的唯一对应。

*   **L3 Cache (预处理数据缓存)**:
    *   **位置**: `data/processed/_preprocessed_cache.joblib`
    *   **格式**: Joblib (`.joblib`)
    *   **内容**: 一个包含了所有股票、所有 folds 的、已经过**最终预处理**（标准化、序列化、Tensor转换）的 Python 字典。
    *   **作用**: **性能优化的终极手段**。它将模型训练循环中的所有 CPU 密集型数据准备工作**一次性**完成，使得 HPO 和主训练循环可以几乎零 CPU 开销地运行，最大化 GPU 利用率。

### 1.5 项目文件结构
```
.
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── ...
├── data_process/
│   ├── data_contracts.py
│   ├── feature_calculators.py
│   ├── get_data.py
│   ├── risk_manager.py
│   └── save_data.py
├── model_builders/
│   ├── build_models.py
│   ├── model_fuser.py
│   └── hpo_utils.py
│   ├── lgbm_builder.py
│   └── lstm_builder.py
├── hpo_logs/
│   └── hpo_best_results.csv
├── Train.ipynb
├── Incremental_Update.ipynb
└── Prophet.ipynb
```

---

## 2. 核心模块 API 详解

本章节详细介绍了项目中每个核心 `.py` 文件的公共 API（函数和类）、它们的职责、接口定义以及内部关键实现。

### 2.1 数据层: `data_process/`

#### **`get_data.py`**
*   **文件职责**: 数据获取与特征工程流水线的核心引擎。

##### 公共函数

1.  **`initialize_apis(config: dict)`**
    *   **功能**: 初始化并登录所有外部数据 API（如 Baostock, Tushare）。
    *   **输入**: `config` (dict): 完整的项目配置字典。
    *   **输出**: 无。
    *   **用法**: 在任何需要进行数据下载的流程（如 `train.ipynb` 的阶段一）开始前**必须**调用一次。

2.  **`shutdown_apis()`**
    *   **功能**: 安全地登出所有 API 并重置内部登录状态。
    *   **输入**: 无。
    *   **输出**: 无。
    *   **用法**: 在数据下载流程结束后，通过 `finally` 块调用，以确保资源被正确释放。

3.  **`get_full_feature_df(ticker: str, config: dict, keyword: str = None, prediction_mode: bool = False) -> Optional[pd.DataFrame]`**
    *   **功能**: 为单只股票执行完整的端到端特征生成。这是数据处理的核心函数。
    *   **输入**:
        *   `ticker` (str): 股票代码 (e.g., `'600519.SH'`)。
        *   `config` (dict): 完整的项目配置字典。
        *   `keyword` (str, optional): 股票名称，用于日志输出。
        *   `prediction_mode` (bool): 模式开关。`False` (默认) 用于训练，会根据配置获取长期历史数据；`True` 用于预测，会自动获取短期最新数据。
    *   **输出**:
        *   `pd.DataFrame`: 包含所有特征和标签的、经过清洗和校验的 DataFrame。
        *   `None`: 如果在任何步骤中发生严重错误或无数据。
    *   **调用链**: 内部按顺序调用 `_get_ohlcv_data_bs`, `_get_macroeconomic_data_cn`, `feature_calculators.run_all_feature_calculators`, `_add_relative_performance_features` 等一系列内部函数。

4.  **`process_all_from_config(config_path: str, tickers_to_generate: list = None) -> Dict[str, pd.DataFrame]`**
    *   **功能**: 批量为指定的股票列表生成特征数据。
    *   **输入**:
        *   `config_path` (str): `config.yaml` 的文件路径。
        *   `tickers_to_generate` (list, optional): 一个股票代码列表。如果提供，则只处理这个列表中的股票；如果为 `None`，则处理配置文件中的所有股票。
    *   **输出**: `dict`: 一个字典，键为股票代码，值为对应的特征 DataFrame。

##### 内部函数简介
*   `_download_with_retry`: 实现了指数退避重试，增强了网络请求的健壮性。
*   `_get_ohlcv_data_bs`: 实现了 L1 缓存，先查本地，再下载。
*   `_make_features_stationary`, `_create_and_clean_labels`: 实现了具体的特征计算步骤，职责单一。

---

#### **`save_data.py`**
*   **文件职责**: 数据流水线的顶层协调器，并负责 L2（特征数据）缓存的写入与路径管理。

##### 公共函数

1.  **`run_data_pipeline(config_path: str)`**
    *   **功能**: `train.ipynb` **阶段一的唯一入口**。它智能地检查每只股票的 L2 缓存是否存在且最新，只为需要更新的股票触发 `process_all_from_config` 流程。
    *   **输入**: `config_path` (str): `config.yaml` 的文件路径。
    *   **输出**: 无 (副作用是生成或更新磁盘上的 L2 缓存文件)。

2.  **`get_processed_data_path(stock_info: dict, config: dict) -> Path`**
    *   **功能**: **数据身份的唯一权威**。根据静态配置计算哈希值，并返回唯一的 L2 缓存文件路径。
    *   **输入**:
        *   `stock_info` (dict): 单只股票的配置（来自 `stocks_to_process` 列表）。
        *   `config` (dict): 完整的项目配置字典。
    *   **输出**: `pathlib.Path` 对象，指向最终的 `.pkl` 文件。
    *   **用法**: 在项目的**任何地方**（阶段一保存、阶段二加载）需要定位 L2 数据文件时，都**必须**调用此函数。

---

#### `data_contracts.py`
*   **文件职责**: 定义数据契约 (Schema)，保障数据质量。
*   **核心类**: `DataValidator`
    *   `__init__(self, config: dict)`: 初始化校验器，并根据配置构建 `pandera` 的 Schema。
    *   `validate_schema(self, df: pd.DataFrame) -> bool`: 对输入的 DataFrame 执行完整的格式、类型、范围和连续性校验。
*   **内部函数简介**:
    *   `check_no_large_gaps`: 智能地忽略上市前的空白期，只检查交易历史中的数据断层。

---

### 2.2 模型层 (`model/`)

#### `build_models.py`
*   **文件职责**: 实现 Walk-Forward 滚动训练的顶层协调逻辑。

##### 公共函数

1.  **`run_training_for_ticker(preprocessed_folds: list, ticker: str, ... ) -> Optional[pd.DataFrame]`**
    *   **功能**: `train.ipynb` **阶段 2.3（主训练）的唯一入口**。
    *   **输入**:
        *   `preprocessed_folds` (list): 一个列表，包含了为某只股票预处理好的**所有** folds 的数据。每个元素是一个字典，如 `{'X_train_scaled': ..., 'y_train': ...}`。
        *   其他参数: `ticker`, `model_type`, `config` 等元信息。
    *   **输出**:
        *   `pd.DataFrame`: 包含该模型在该股票上所有 folds 的 IC 历史记录。
        *   `None`: 如果训练失败。
    *   **副作用**:
        *   训练并保存最新版本的模型 (`_model.pkl/.pt`) 和 Scaler (`_scaler.pkl`)。
        *   生成并保存 `_oof_preds.csv` 文件，用于后续的融合模型训练。
        *   生成并保存 `_ic_history.csv` 文件。

##### 内部函数简介
*   `_walk_forward_split`: 一个纯粹的数据切分工具，根据 `train_window` 和 `val_window` 将时间序列 DataFrame 切分为一个 folds 列表。

---

#### `model_fuser.py`
*   **文件职责**: 实现生产级的模型堆叠（Stacking）融合器。

##### 核心类: `ModelFuser`

*   **`__init__(self, ticker: str, config: dict)`**
    *   **功能**: 初始化融合器实例，设置随机种子，并根据配置选择元模型（如 Ridge）。
*   **公共方法**:
    1.  **`train()`**
        *   **功能**: **训练入口**。执行完整的、健壮的元模型训练流程，包括加载 OOF 数据、标准化、时间序列交叉验证、稳定性与性能评估，并最终保存所有构件（模型、Scaler、元信息 JSON）。
        *   **输入**: 无 (它会自己从磁盘加载所需文件)。
        *   **输出**: 无。
    2.  **`load() -> bool`**
        *   **功能**: **加载入口**。安全地从磁盘加载已训练的融合模型、Scaler 和元信息。
        *   **输出**: `True` 如果加载成功，否则 `False`。
    3.  **`predict(preds: dict) -> float`**
        *   **功能**: **预测入口**。对一组新的基础模型预测进行融合。
        *   **输入**: `preds` (dict): 一个字典，包含基础模型的预测，如 `{'pred_lgbm': 0.01, 'pred_lstm': -0.005}`。
        *   **输出**: `float`: 经过融合、限幅和平滑处理后的最终信号。
    4.  **`partial_train(new_data: pd.DataFrame)`**
        *   **功能**: **增量训练入口**。使用新的 `(X, y)` 数据点对现有融合模型进行在线更新。
        *   **输入**: `new_data` (pd.DataFrame): 包含 `pred_lgbm`, `pred_lstm`, `y_true` 等列的新数据。
        *   **输出**: 无。

---

### 2.3 模型构建器 (`model_builders/`)

#### `lgbm_builder.py` & `lstm_builder.py`
*   **文件职责**: 封装具体模型的实现细节。
*   **核心类**: `LGBMBuilder` / `LSTMBuilder`
    *   `__init__(self, config: dict)`: 根据配置初始化模型参数。
    *   **`train_and_evaluate_fold(self, train_df, val_df, cached_data) -> Tuple[dict, pd.DataFrame, pd.DataFrame]`**:
        *   **功能**: **唯一的训练接口**。它被设计为既可以处理原始数据（由 HPO 调用），也可以处理预处理好的缓存数据（由主训练流程调用）。
        *   **输入**: `cached_data` (dict): 包含了预处理好的 `X_train_scaled`, `y_train` 等。
        *   **输出**: 一个元组，包含三个元素：
            1.  `dict`: 训练构件（当前未使用）。
            2.  `pd.DataFrame`: 当前 fold 的 IC 记录。
            3.  `pd.DataFrame`: 当前 fold 的 OOF 预测 (`date`, `y_true`, `y_pred`)。
    *   `train_final_model(self, full_df)`: 在全部数据上训练最终模型并返回构件。

---

## 6. 附录：关键概念

*   **OOF (Out-of-Fold)**: 在交叉验证（此处为 Walk-Forward）过程中，模型在每个验证集上生成的预测。将所有 folds 的 OOF 预测拼接起来，可以得到一个与原始数据集等长、且每个预测点都未见过其训练标签的“干净”预测序列。这是构建元模型训练集的黄金标准，可以有效避免数据泄露。
*   **IC (Information Coefficient)**: 信息系数，通常指 Spearman 秩相关系数。衡量模型预测值与未来真实收益率之间的排序一致性。是评估 Alpha 策略选股能力的核心指标。
*   **ICIR (Information Ratio)**: 信息比率，即 IC 的均值除以 IC 的标准差 (`IC_Mean / IC_Std`)。它综合了策略的**收益能力**（IC 均值）和**稳定性**（IC 标准差），是比单纯的 IC 均值更科学的评估指标。

---

## 5. 快速上手与未来展望

*   **快速上手**: 请参照 `train.ipynb` 中的 Markdown 文本，按顺序执行所有单元格。确保 `configs/config.yaml` 已正确配置。
*   **未来展望**:
    *   **扩展基础模型**: 在 `model_builders/` 中添加新的 `transformer_builder.py`，并在 `build_models.py` 和 `config.yaml` 中进行注册。`ModelFuser` 已原生支持 N 模型融合。
    *   **接入实盘数据**: 改造 `Incremental_Update.ipynb`，将其中的“模拟”部分替换为对您的真实数据源（数据库/API）的调用。
    *   **开发回测引擎**: 创建一个新的模块，利用已保存的 OOF 预测文件或 `ModelFuser`，进行向量化的事件驱动回测，生成资金曲线、夏普比率等最终的策略评估指标。

**本文档旨在作为项目知识传递的核心。请在对代码进行任何重大修改后，及时更新本文档。**