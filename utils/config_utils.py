# 文件路径: utils/config_utils.py

import json
import yaml
import torch
import joblib
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import StandardScaler

def load_and_merge_configs_for_notebook(main_config_path: str = 'configs/config.yaml') -> dict:
    """
    专门为 Jupyter Notebook 设计的配置加载器。
    模拟 Hydra 的行为，加载并合并所有模块化的子配置文件。
    """
    print("--- 正在为 Notebook 加载和合并所有配置文件 ---")
    main_config_path = Path(main_config_path)
    config_dir = main_config_path.parent
    with open(main_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"  - 主配置 '{main_config_path.name}' 已加载。")

    defaults = config.get('defaults', [])
    if not defaults: return config

    merged_config = {}
    for item in defaults:
        if isinstance(item, dict):
            group, name = list(item.items())[0]
            sub_config_path = config_dir / group / f"{name}.yaml"
            if sub_config_path.exists():
                with open(sub_config_path, 'r', encoding='utf-8') as f:
                    sub_config = yaml.safe_load(f)
                    # 将子配置内容合并到以组名为 key 的字典中
                    merged_config[group] = sub_config
                print(f"  - 子配置 '{sub_config_path.relative_to(config_dir)}' 已加载到 '{group}' 组。")
    
    # 深度合并，主配置可以覆盖子配置
    def deep_merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict) and key in destination:
                destination[key] = deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

    final_config = deep_merge(config, merged_config)
    final_config.pop('defaults', None); final_config.pop('_self_', None)
    print("--- 所有配置文件合并成功 ---")
    return final_config