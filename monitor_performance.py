#!/usr/bin/env python3
"""
性能监控脚本 - 跟踪优化效果
Performance Monitoring Script - Track optimization results
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

def analyze_trading_results():
    """分析交易结果"""
    print("="*60)
    print("交易结果分析 / Trading Results Analysis")
    print("="*60)
    
    # 查找最新的结果文件
    result_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'net_assets' in file and file.endswith('.npy'):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("未找到结果文件 / No result files found")
        return
    
    # 分析最新结果
    latest_file = max(result_files, key=os.path.getctime)
    print(f"分析文件: {latest_file}")
    
    try:
        net_assets = np.load(latest_file)
        print(f"数据形状: {net_assets.shape}")
        
        if len(net_assets) > 0:
            initial_value = net_assets[0] if net_assets[0] != 0 else 1e6
            final_value = net_assets[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            print(f"初始资产: ${initial_value:,.2f}")
            print(f"最终资产: ${final_value:,.2f}")
            print(f"总收益率: {total_return:.2f}%")
            
            # 计算年化收益率（假设数据覆盖时间）
            time_periods = len(net_assets)
            if time_periods > 252:  # 假设每日数据
                years = time_periods / 252
                annual_return = ((final_value / initial_value) ** (1/years) - 1) * 100
                print(f"年化收益率: {annual_return:.2f}%")
            
            # 计算最大回撤
            peak = np.maximum.accumulate(net_assets)
            drawdown = (net_assets - peak) / peak * 100
            max_drawdown = np.min(drawdown)
            print(f"最大回撤: {max_drawdown:.2f}%")
            
            # 计算夏普比率（简化版本）
            returns = np.diff(net_assets) / net_assets[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                print(f"夏普比率: {sharpe_ratio:.3f}")
        
    except Exception as e:
        print(f"分析出错: {e}")

if __name__ == "__main__":
    analyze_trading_results()
