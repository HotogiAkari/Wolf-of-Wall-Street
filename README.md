# Wolf of Wall Street

![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch_|_LightGBM_|_Scikit--learn-orange.svg)

**WOWS** 是是一款安东星第三人称海战... 面向A股市场的端到端的机器学习量化研究与预测框架, 集成了从数据获取, 特征工程, 多模型训练, 超参数优化到高级模型融合的全流程.

---

## 部署与使用指南

### 1. 环境安装

1.  **克隆项目**
    ```bash
    git clone https://github.com/HotogiAkari/Wolf-of-Wall-Street.git
    cd Wolf-of-Wall-Street
    ```

2.  **创建虚拟环境** (推荐)
   
    该项目是在 Python `3.13.0` 版本下编写的.
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **安装依赖**
    *   **前提**:
        *   **PyTorch**: 请务必访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，根据你的 CUDA 版本，获取并执行官方的安装命令. 
        *   **LightGBM (GPU版)**:
            *   确保已安装 C++ 编译器 (如 Visual Studio). 
            *   然后运行 `pip install lightgbm --install-option=--gpu` 或使用 Conda 安装. 

        > *你说你没有GPU? 还是去租一个云计算器吧, CPU太慢了*
    *   **安装其他库**:
        ```bash
        pip install -r requirements.txt
        ```

### 2. 项目配置

**编辑配置文件**: 可以参考项目的 `config.yaml` 进行配置,每个参数都有详细的注释(累). 
*  在 `global_settings` 中，填入你的 `tushare_api_token` (不填也行, 我没积分所以用不了...). 
*  在 `stocks_to_process` 中，定义感兴趣的股票池. 
*  根据你的 CPU 核心数(默认为8)，调整 `lgbm_hpo_params.n_jobs` 和 `default_model_params.lgbm_params.n_jobs` .默认为8. 
*  根据你的 GPU 显存，调整 `default_model_params.lstm_params.batch_size` (默认为128). 

### 3. 运行程序：

在 Jupyter Lab 或 Jupyter Notebook 中打开 `train.ipynb`，然后按顺序执行以下单元格. 

#### **1：数据准备**
运行“**阶段一：数据准备与特征工程**”单元格. 

> 程序会检查并生成所有需要的 L1, L2 缓存文件. 首次运行会比较耗时. 

#### **2：模型训练**
依次运行“**阶段二**”之后的所有单元格. 

1.  **2.1 (预处理)**: 首次运行时，会生成 L3 缓存，将所有数据预处理到内存中. 后续运行会直接加载 L3 缓存. 
2.  **2.2 (HPO)**: 如果 `RUN_HPO = True`，会利用 L3 缓存进行自动调参(很耗时,慎重使用). 
3.  **2.3 (模型训练)**: 利用 L3 缓存，为所有股票和模型执行完整的滚动训练，并生成 OOF 文件. 
4.  **2.3.5 (融合训练)**: 利用 OOF 文件，训练融合模型. 
5.  **2.4 (评估)**: 生成包含所有模型（LGBM, LSTM, FUSION）性能的最终评估报告和图表. 

#### **3 ：进行预测**
打开并运行 `Prophet.ipynb`. 
*   **作用**: 加载最新训练好的基础模型和融合模型，获取最新的市场数据，并输出一份专业的, 风险可控的投资建议报告. 

#### **4 : 更新模型**
> 觉得模型过于老旧?

使用 `Incremental_Update.ipynb` 下载最新数据,对模型进行增量更新
 
---

## 特点

*   **多模型支持**: 内置了对 `LightGBM` (梯度提升树) 和 `LSTM` (长短期记忆网络) 的支持，能够同时从“横截面”和“时间序列”两个维度捕捉市场信息. 
*   **高级模型融合 (Stacking)**: 独创的 `ModelFuser` 模块，通过二次训练元模型（如 ElasticNet 或 MLP）来智能地, 非线性地结合基础模型的预测，而非简单的加权平均. 
*   **自动化超参数优化 (HPO)**: 集成 `Optuna`，能够为 `LGBM` 和 `LSTM` 进行高效的自动化参数搜索，并支持“冠军排行榜”模式，持续追踪并应用历史最佳参数. 


## 文件结构


```
.
├── configs/
│   └── config.yaml              # 中央配置文件
├── data/
│   └── processed/               # L2 和 L3 缓存的输出目录
├── data_cache/                  # L1 原始数据缓存目录
├── data_process/
│   ├── data_contracts.py        # 数据质量校验契约
│   ├── feature_calculators.py   # 自定义特征计算器
│   ├── get_data.py              # 核心数据获取与特征工程引擎
│   ├── factor_data.py           # 因子数据获取模块
│   └── save_data.py             # 数据流水线协调器与 L2 缓存管理
├── model_builders/
│   ├── lgbm_builder.py          # LightGBM 模型构建与训练实现
│   ├── lstm_builder.py          # LSTM 模型构建与训练实现
│   ├── build_models.py          # Walk-Forward 滚动训练协调器
│   ├── model_fuser.py           # 模型融合器
│   └── hpo_utils.py             # Optuna 超参数优化工具
├── hpo_logs/
│   └── hpo_best_results_*.csv   # HPO最优参数日志
├── Train.ipynb                  # 主工作流 Notebook (数据处理 + 模型训练)
├── Prophet.ipynb                # 预测应用 Notebook
└──Incremental_Update.ipynb      # 模型增量更新 Notebook
```

---

## 卫星

-   [ ] **构建向量化回测引擎**: 实现基于 OOF 预测的事件驱动或向量化回测，生成资金曲线, 夏普比率等策略最终指标. 
-   [ ] **引入更多基础模型**: 在 `model_builders` 中加入更多模型. 
-   [ ] **完善因子库**: 引入更多 Alpha 因子(想到啥就加). 
-   [ ] **制作交互界面**: 做成一个可执行程序或者加一个网页端的交互界面?
-   [ ] **新闻情绪因子**: 爬虫来读新闻
-   [ ] **其它还没想好的功能**

## 贡献

欢迎通过 Pull Requests 提交改进或贡献你的代码！对于重大的改动，请先开启一个 Issue 进行讨论. 

## 许可证

本项目采用 [MIT](https://choosealicense.com/licenses/mit/) 许可证. 