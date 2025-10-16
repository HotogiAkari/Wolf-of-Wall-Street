# 文件路径: cleanup_artifacts.py

import yaml
import shutil
import argparse
from pathlib import Path

def cleanup_artifacts(config_path: str, skip_confirmation: bool = False):
    """
    根据配置文件中的股票池和目录设置，清理不再需要的股票相关的构件。
    
    :param config_path: config.yaml 文件的路径。
    :param skip_confirmation: 如果为 True，则跳过交互式确认，直接删除。
    """
    print("--- 开始清理过时的股票构件 ---")

    # --- 1. 读取配置和股票池 ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: 加载或解析配置文件 '{config_path}' 失败: {e}。"); return

    stocks_to_process = config.get('stocks_to_process', [])
    if not stocks_to_process:
        print("ERROR: 配置文件中的股票池 'stocks_to_process' 为空。"); return
        
    active_tickers = {s['ticker'] for s in stocks_to_process if 'ticker' in s}
    print(f"INFO: 从配置文件中加载了 {len(active_tickers)} 只“现役”股票。")

    # --- 2. 动态读取要扫描的目录 ---
    global_settings = config.get('global_settings', {})
    paths_to_scan = [
        Path(global_settings.get('data_cache_dir', 'data_cache')),
        Path(global_settings.get('output_dir', 'data/processed')),
        Path(global_settings.get('model_dir', 'models')),
    ]
    
    print("\nINFO: 将扫描以下核心目录:")
    for path in paths_to_scan: print(f"  - {path.resolve()}")

    # --- 3. 扫描并识别待删除目录 ---
    directories_to_delete = []
    print("\n--- 正在扫描和盘点所有构件目录...")
    for base_path in paths_to_scan:
        if not base_path.exists() or not base_path.is_dir():
            print(f"  - 目录不存在，跳过: {base_path}"); continue
            
        print(f"  - 正在扫描: {base_path}")
        for stock_dir in base_path.iterdir():
            if stock_dir.is_dir() and stock_dir.name not in active_tickers:
                directories_to_delete.append(stock_dir)

    # --- 4. 汇总并等待确认 ---
    if not directories_to_delete:
        print("\n--- 盘点完成 ---"); print("\n所有构件都与当前股票池匹配，无需清理。"); return

    print("\n--- 盘点完成，以下目录被标记为“已淘汰”，将被删除 ---")
    for dir_path in sorted(list(set(directories_to_delete))): print(f"  - {dir_path}")

    confirm = 'no' # 默认不删除
    if skip_confirmation:
        confirm = 'yes'
    else:
        # --- 核心修正：使用 'yes' 作为确认，更安全 ---
        confirm = input("\n警告: 以上目录及其所有内容将被永久删除！是否继续? (请输入 'y' 或 'yes' 进行确认): ").lower()
    
    if confirm in ['y', 'yes']:
        print("\n--- 正在执行删除操作...")
        for dir_path in sorted(list(set(directories_to_delete))):
            try:
                shutil.rmtree(dir_path)
                print(f"  - 已删除: {dir_path}")
            except Exception as e:
                print(f"  - 删除失败: {dir_path} - 原因: {e}")
        print("\n--- 清理完成 ---")
    else:
        print("\n--- 操作已取消 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="清理与当前配置文件中股票池不匹配的旧构件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml', 
        help='配置文件的路径 (默认: configs/config.yaml)'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true', # 当出现 --yes 时，args.yes 的值为 True
        help='自动确认删除，不再进行交互式提问 (用于自动化脚本)。'
    )
    
    args = parser.parse_args()
    
    # 直接调用函数，并将 --yes 参数的结果传递给 skip_confirmation
    cleanup_artifacts(config_path=args.config, skip_confirmation=args.yes)