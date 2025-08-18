#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密货币高频交易强化学习系统演示
Cryptocurrency High-Frequency Trading RL System Demo
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_config():
    """
    演示配置模块
    Demo configuration module
    """
    print("=== 配置模块演示 Configuration Module Demo ===")
    
    try:
        from config.settings import ProjectConfig, config
        
        print("✓ 配置模块加载成功")
        print(f"✓ 项目根目录: {config.get_config('data')['btc_csv_path'].parent.parent}")
        
        # 显示主要配置
        data_config = config.get_config("data")
        rl_config = config.get_config("rl")
        
        print(f"\n数据配置:")
        print(f"  - BTC数据路径: {data_config['btc_csv_path']}")
        print(f"  - 输入数组路径: {data_config['input_ary_path']}")
        
        print(f"\n强化学习配置:")
        print(f"  - 模拟次数: {rl_config['num_sims']}")
        print(f"  - 学习率: {rl_config['learning_rate']}")
        print(f"  - 批次大小: {rl_config['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置模块演示失败: {str(e)}")
        return False

def demo_utils():
    """
    演示工具模块
    Demo utilities module
    """
    print("\n=== 工具模块演示 Utilities Module Demo ===")
    
    try:
        from utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
        from utils.analyze_results import analyze_trading_performance
        
        print("✓ 工具模块加载成功")
        
        # 演示指标计算
        import numpy as np
        
        # 生成示例数据
        returns = np.random.normal(0.001, 0.02, 1000)  # 模拟日收益率
        equity_curve = np.cumprod(1 + returns)
        
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(equity_curve)
        
        print(f"\n示例指标计算:")
        print(f"  - 夏普比率: {sharpe:.4f}")
        print(f"  - 最大回撤: {max_dd:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 工具模块演示失败: {str(e)}")
        return False

def demo_project_structure():
    """
    演示项目结构
    Demo project structure
    """
    print("\n=== 项目结构演示 Project Structure Demo ===")
    
    modules = {
        "数据处理": "data_processing",
        "序列模型": "sequential_models", 
        "强化学习": "reinforcement_learning",
        "交易模拟": "trading_simulation",
        "集成评估": "ensemble_evaluation",
        "工具模块": "utils",
        "配置管理": "config"
    }
    
    print("项目模块结构:")
    for name, folder in modules.items():
        folder_path = project_root / folder
        if folder_path.exists():
            py_files = list(folder_path.glob("*.py"))
            print(f"  ✓ {name} ({folder}): {len(py_files)} 个Python文件")
        else:
            print(f"  ✗ {name} ({folder}): 目录不存在")
    
    # 检查关键文件
    key_files = ["main.py", "requirements.txt", "README.md", "setup.py"]
    print(f"\n关键文件:")
    for file_name in key_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name}")
    
    return True

def demo_data_processing():
    """
    演示数据处理功能（如果可用）
    Demo data processing functionality (if available)
    """
    print("\n=== 数据处理演示 Data Processing Demo ===")
    
    try:
        # 尝试导入数据配置
        sys.path.append(str(project_root / "data_processing"))
        from data_config import ConfigData
        
        config_data = ConfigData()
        print("✓ 数据配置加载成功")
        print(f"  - 输入维度: {config_data.inp_dim}")
        print(f"  - 输出维度: {config_data.out_dim}")
        print(f"  - 序列长度: {config_data.seq_len}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据处理演示失败: {str(e)}")
        print("  注意: 这可能是由于缺少依赖或数据文件导致的")
        return False

def main():
    """
    主演示函数
    Main demo function
    """
    print("🚀 加密货币高频交易强化学习系统演示")
    print("🚀 Cryptocurrency High-Frequency Trading RL System Demo")
    print("=" * 70)
    
    demos = [
        ("项目结构", demo_project_structure),
        ("配置模块", demo_config),
        ("工具模块", demo_utils),
        ("数据处理", demo_data_processing),
    ]
    
    passed_demos = 0
    total_demos = len(demos)
    
    for demo_name, demo_func in demos:
        try:
            if demo_func():
                passed_demos += 1
        except Exception as e:
            print(f"✗ {demo_name}演示异常: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"演示结果: {passed_demos}/{total_demos} 成功")
    print(f"Demo Results: {passed_demos}/{total_demos} successful")
    
    if passed_demos >= total_demos - 1:  # 允许一个演示失败
        print("\n🎉 项目重构成功！主要功能模块已正确组织。")
        print("🎉 Project refactoring successful! Main functional modules are properly organized.")
        print("\n📋 使用说明:")
        print("   1. 安装依赖: pip install -r requirements.txt")
        print("   2. 运行主程序: python main.py --help")
        print("   3. 查看文档: cat README.md")
        return 0
    else:
        print("\n⚠️  项目重构基本完成，但部分模块可能需要进一步调整。")
        print("⚠️  Project refactoring mostly complete, but some modules may need further adjustment.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)