import yaml
from pathlib import Path
import json # 导入 json 以便美观地打印字典

def load_and_merge_configs_for_notebook(main_config_path: str = 'configs/config.yaml') -> dict:
    """
    为 Jupyter Notebook 设计的配置加载器。
    模拟 Hydra 的行为。
    """
    print("--- 开始为 Notebook 加载和合并所有配置文件 ---")
    
    main_config_path = Path(main_config_path)
    config_dir = main_config_path.parent

    # 1. 加载主配置文件作为基础
    try:
        with open(main_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        print(f"  - [DEBUG] 主配置 '{main_config_path.name}' 已加载。")
    except Exception as e:
        print(f"  - [FATAL_DEBUG] 加载主配置文件失败: {e}")
        return {}

    # 2. 创建一个空的最终配置
    final_config = {}

    # 3. 定义深度更新函数
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # 4. 遍历 defaults 列表，依次加载并深度合并
    defaults = base_config.get('defaults', [])
    
    for item in defaults:
        config_to_merge = None
        item_name = "unknown"

        # a. 处理子配置
        if isinstance(item, dict):
            group, name = list(item.items())[0]
            item_name = f"{group}/{name}"
            sub_config_path = config_dir / group / f"{name}.yaml"
            if sub_config_path.exists():
                with open(sub_config_path, 'r', encoding='utf-8') as f:
                    sub_config = yaml.safe_load(f)
                    # 将其内容包裹在组名下
                    config_to_merge = {group: sub_config}
            else:
                print(f"  - [WARN_DEBUG] 找不到子配置文件: {sub_config_path}")

        # b. 处理主配置 (_self_)
        elif isinstance(item, str) and item == '_self_':
            item_name = "_self_"
            config_to_merge = base_config

        # c. 执行合并并打印状态
        if config_to_merge:
            print(f"\n  - [DEBUG] 正在合并 '{item_name}'...")
            deep_update(final_config, config_to_merge)
            
            # 检查 stocks_to_process 在这一步之后的状态
            stocks_after_merge = final_config.get('data', {}).get('stocks_to_process')

    # 5. 清理
    final_config.pop('defaults', None)
    
    print("\n--- 所有配置文件合并完成 ---")
    return final_config