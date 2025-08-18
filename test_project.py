#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目功能测试脚本
Project functionality test script
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """
    测试所有模块导入
    Test all module imports
    """
    print("=== 测试模块导入 Testing Module Imports ===")
    
    modules_to_test = [
        # 数据处理模块
        "data_processing.seq_data",
        "data_processing.data_config",
        
        # 序列模型模块
        "sequential_models.seq_net",
        "sequential_models.seq_run",
        "sequential_models.seq_record",
        
        # 强化学习模块
        "reinforcement_learning.erl_agent",
        "reinforcement_learning.erl_config",
        "reinforcement_learning.erl_run",
        "reinforcement_learning.erl_evaluator",
        "reinforcement_learning.erl_replay_buffer",
        
        # 交易模拟模块
        "trading_simulation.trade_simulator",
        "trading_simulation.validate_model",
        
        # 集成评估模块
        "ensemble_evaluation.task2_eval",
        "ensemble_evaluation.task2_ensemble",
        "ensemble_evaluation.benchmark_models",
        
        # 工具模块
        "utils.metrics",
        "utils.generate_article_figures",
        "utils.plot_trade_log",
        "utils.analyze_results",
        
        # 配置模块
        "config.settings",
    ]
    
    success_count = 0
    failed_modules = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
            success_count += 1
        except Exception as e:
            print(f"✗ {module_name}: {str(e)}")
            failed_modules.append((module_name, str(e)))
    
    print(f"\n导入测试结果: {success_count}/{len(modules_to_test)} 成功")
    print(f"Import test results: {success_count}/{len(modules_to_test)} successful")
    
    if failed_modules:
        print("\n失败的模块 Failed modules:")
        for module, error in failed_modules:
            print(f"  - {module}: {error}")
    
    return len(failed_modules) == 0

def test_config():
    """
    测试配置模块
    Test configuration module
    """
    print("\n=== 测试配置模块 Testing Configuration Module ===")
    
    try:
        from config.settings import ProjectConfig, config
        
        # 测试配置获取
        data_config = config.get_config("data")
        model_config = config.get_config("model")
        rl_config = config.get_config("rl")
        
        print(f"✓ 数据配置项数量: {len(data_config)}")
        print(f"✓ 模型配置项数量: {len(model_config)}")
        print(f"✓ 强化学习配置项数量: {len(rl_config)}")
        
        # 测试路径配置
        from config.settings import PROJECT_ROOT, DATA_ROOT
        print(f"✓ 项目根目录: {PROJECT_ROOT}")
        print(f"✓ 数据目录: {DATA_ROOT}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置模块测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_directory_structure():
    """
    测试目录结构
    Test directory structure
    """
    print("\n=== 测试目录结构 Testing Directory Structure ===")
    
    required_dirs = [
        "data_processing",
        "sequential_models",
        "reinforcement_learning",
        "trading_simulation",
        "ensemble_evaluation",
        "utils",
        "config"
    ]
    
    required_files = [
        "main.py",
        "requirements.txt",
        "README.md",
        "setup.py"
    ]
    
    project_root = Path(".")
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ 目录存在: {dir_name}")
        else:
            print(f"✗ 目录缺失: {dir_name}")
            return False
    
    # 检查文件
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            print(f"✓ 文件存在: {file_name}")
        else:
            print(f"✗ 文件缺失: {file_name}")
            return False
    
    return True

def main():
    """
    主测试函数
    Main test function
    """
    print("加密货币高频交易强化学习系统 - 项目测试")
    print("Cryptocurrency High-Frequency Trading RL System - Project Test")
    print("=" * 60)
    
    # 添加当前目录到Python路径
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    tests = [
        ("目录结构测试", test_directory_structure),
        ("配置模块测试", test_config),
        ("模块导入测试", test_imports),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n开始 {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} 通过")
                passed_tests += 1
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！项目设置正确。")
        print("🎉 All tests passed! Project setup is correct.")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查项目配置。")
        print("❌ Some tests failed, please check project configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)