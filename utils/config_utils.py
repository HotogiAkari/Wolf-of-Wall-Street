# 文件路径: utils/config_utils.py

import yaml
from pathlib import Path

def load_and_merge_configs_for_notebook(main_config_path: str = 'configs/config.yaml') -> dict:
    """
    专门为 Jupyter Notebook 设计的配置加载器。
    它可以模拟 Hydra 的行为，读取主配置文件并根据 'defaults' 列表
    加载和合并所有模块化的子配置文件。
    """
    print("--- 正在为 Notebook 加载和合并所有配置文件 ---")
    
    main_config_path = Path(main_config_path)
    config_dir = main_config_path.parent

    # 1. 加载主配置文件
    with open(main_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"  - 主配置 '{main_config_path.name}' 已加载。")

    # 2. 解析 defaults 列表并合并
    defaults = config.get('defaults', [])
    if not defaults:
        return config # 如果没有 defaults，直接返回主配置

    merged_config = {}
    for item in defaults:
        if isinstance(item, dict): # 处理字典形式的 default, e.g., { 'model': 'lgbm' }
            group, name = list(item.items())[0]
            sub_config_path = config_dir / group / f"{name}.yaml"
            
            if sub_config_path.exists():
                with open(sub_config_path, 'r', encoding='utf-8') as f:
                    sub_config = yaml.safe_load(f)
                    # 将子配置内容合并到以组名为 key 的字典中
                    if group not in merged_config:
                        merged_config[group] = {}
                    merged_config[group].update(sub_config)
                print(f"  - 子配置 '{sub_config_path.relative_to(config_dir)}' 已加载并合并到 '{group}' 组。")
            else:
                print(f"  - WARNNING: 找不到子配置文件: {sub_config_path}")

    # 3. 将主配置的内容合并进来（它拥有最高优先级，可以覆盖子配置）
    # 我们需要深度合并，而不是简单的 update
    def deep_merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    final_config = deep_merge(config, merged_config)
    
    # 删除 Hydra 特有的 defaults 键
    final_config.pop('defaults', None)
    final_config.pop('_self_', None)

    print("--- 所有配置文件合并成功 ---")
    return final_config