# 文件路径: clean_data/cleanup.py

import yaml
import shutil
import argparse
from pathlib import Path
import re

def cleanup_artifacts(config_path: str, skip_confirmation: bool = False):
    """
    根据配置文件中定义的、可扩展的策略，清理不再需要的股票相关的构件。
    能够智能区分个股、指数和全局缓存文件。
    
    :param config_path: config.yaml 文件的路径。
    :param skip_confirmation: 如果为 True，则跳过交互式确认，直接删除。
    """
    print("--- 开始清理过时的股票构件 ---")

    # --- 1. 读取配置和所有“白名单” ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: 加载或解析配置文件 '{config_path}' 失败: {e}。"); return

    # a. 白名单1: 现役个股
    stocks_to_process = config.get('stocks_to_process', [])
    if not stocks_to_process:
        print("ERROR: 配置文件中的股票池 'stocks_to_process' 为空。"); return
    active_tickers = {s['ticker'] for s in stocks_to_process if 'ticker' in s}
    active_safe_tickers = {t.replace('.', '_') for t in active_tickers} # 文件名安全版本
    print(f"INFO: 从配置文件中加载了 {len(active_tickers)} 只“现役”个股。")

    # b. 白名单2: 系统保留的指数/外部市场代码
    system_tickers = set()
    strategy_config = config.get('strategy_config', {})
    if strategy_config.get('benchmark_ticker'):
        system_tickers.add(strategy_config['benchmark_ticker'])
    for stock_info in stocks_to_process:
        if stock_info.get('industry_etf'):
            system_tickers.add(stock_info['industry_etf'])
    system_tickers.update(strategy_config.get('external_market_tickers', []))
    system_safe_tickers = {t.replace('.', '_') for t in system_tickers} # 文件名安全版本
    print(f"INFO: 加载了 {len(system_tickers)} 个系统代码 (指数等) 作为白名单。")

    # c. 白名单3: 全局文件名的正则表达式模式
    global_file_patterns = [
        re.compile(r'^market_breadth_.*'),
        re.compile(r'^market_sentiment_.*'),
        re.compile(r'^_global_data_cache.*'),
    ]

    # --- 2. 从 cleanup_settings 读取清理策略 ---
    cleanup_strategies = config.get('cleanup_settings', {}).get('paths', [])
    if not cleanup_strategies:
        print("WARNNING: 未在 'cleanup_settings' 中定义任何清理策略。"); return

    print("\nINFO: 将根据以下策略进行扫描:")
    for strategy in cleanup_strategies:
        print(f"  - 路径: {strategy.get('path')}, 策略: {strategy.get('strategy')}")

    # --- 3. 根据策略扫描并识别待删除项 ---
    items_to_delete = []
    print("\n--- 正在扫描和盘点所有构件...")
    
    for strategy_config in cleanup_strategies:
        base_path = Path(strategy_config.get('path'))
        strategy = strategy_config.get('strategy')

        if not base_path.exists() or not base_path.is_dir():
            print(f"\n  - 目录不存在，跳过: {base_path}")
            continue
        
        print(f"\n  - 正在扫描 '{base_path}' (策略: {strategy})")

        # --- 策略 A: 按顶级子目录名清理 ---
        if strategy == 'by_toplevel_dir_name':
            for stock_dir in base_path.iterdir():
                # 目录名必须完全匹配一个股票代码
                if stock_dir.is_dir() and stock_dir.name not in active_tickers:
                    items_to_delete.append(stock_dir)
                    print(f"    - [标记删除/目录] {stock_dir.name} (原因: 不在现役股票池中)")

        # --- 策略 B: 按文件名模式递归清理 ---
        elif strategy == 'by_filename_pattern':
            pattern_template = strategy_config.get('pattern', '_{safe_ticker}_')
            
            for file_path in base_path.rglob('*'):
                if not file_path.is_file(): continue

                is_active_file = False
                file_name = file_path.name

                # 规则 1: 检查是否为必须保留的全局文件
                for pat in global_file_patterns:
                    if pat.match(file_name):
                        is_active_file = True
                        break
                if is_active_file: continue

                # 规则 2: 检查是否属于任何一个现役个股
                for safe_ticker in active_safe_tickers:
                    pattern = pattern_template.format(safe_ticker=safe_ticker)
                    if pattern in file_name:
                        is_active_file = True
                        break
                if is_active_file: continue

                # 规则 3: 检查是否属于任何一个系统保留的指数
                for safe_ticker in system_safe_tickers:
                    pattern = pattern_template.format(safe_ticker=safe_ticker)
                    if pattern in file_name:
                        is_active_file = True
                        break
                if is_active_file: continue

                # 如果所有白名单检查都失败，则标记为待删除
                if not is_active_file:
                    # 安全检查：只删除符合已知命名规范的文件，防止误删
                    if file_path.stem.startswith(('raw_', 'market_breadth_')):
                        items_to_delete.append(file_path)
                        print(f"    - [标记删除/文件] {file_name} (原因: 不属于任何现役个股、指数或全局文件)")
        else:
            print(f"    - WARNNING: 未知的清理策略 '{strategy}'，跳过。")

    # --- 4. 汇总并等待确认 ---
    if not items_to_delete:
        print("\n--- 盘点完成 ---"); print("\n所有构件都与当前股票池匹配，无需清理。"); return

    print("\n--- 盘点完成，以下目录或文件被标记为“已淘汰”，将被删除 ---")
    # 使用 set 去重，然后排序，使输出更整洁
    for item_path in sorted(list(set(items_to_delete))): 
        print(f"  - {item_path}")
    
    confirm = 'no'
    if skip_confirmation:
        confirm = 'yes'
    else:
        confirm = input("\n警告: 以上项目及其所有内容将被永久删除！是否继续? (请输入 'y' 或 'yes' 进行确认): ").lower()
    
    # --- 5. 执行删除 ---
    if confirm in ['y', 'yes']:
        print("\n--- 正在执行删除操作...")
        for item_path in sorted(list(set(items_to_delete))):
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"  - 已删除目录: {item_path}")
                elif item_path.is_file():
                    item_path.unlink()
                    print(f"  - 已删除文件: {item_path}")
            except Exception as e:
                print(f"  - 删除失败: {item_path} - 原因: {e}")
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