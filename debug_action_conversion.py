#!/usr/bin/env python3

import numpy as np

# 从调试输出中看到的典型智能体动作
sample_actions = [
    [1, 2, 2],  # 从调试输出中看到的
    [0, 1, 1],  # 另一个例子
    [0, 0, 0],  # 全部为0的情况
    [2, 2, 2],  # 全部为2的情况
    [0, 1, 2],  # 混合情况
]

print("Action conversion analysis:")
print("Original action -> Converted action")
print("0 -> -1 (sell)")
print("1 -> 0 (hold)")
print("2 -> 1 (buy)")
print()

for i, actions in enumerate(sample_actions):
    converted = [a - 1 for a in actions]
    print(f"Sample {i+1}: {actions} -> {converted}")
    
    # 分析转换后的动作分布
    sell_count = converted.count(-1)
    hold_count = converted.count(0)
    buy_count = converted.count(1)
    
    print(f"  Sell: {sell_count}, Hold: {hold_count}, Buy: {buy_count}")
    
    # 模拟ensemble决策（简单多数投票）
    if sell_count > hold_count and sell_count > buy_count:
        decision = "SELL"
    elif buy_count > hold_count and buy_count > sell_count:
        decision = "BUY"
    else:
        decision = "HOLD"
    
    print(f"  Majority decision: {decision}")
    print()

print("\nAnalysis:")
print("- Agent actions 0, 1, 2 are converted to -1, 0, 1 (sell, hold, buy)")
print("- If agents mostly output 1 and 2, converted actions will be 0 and 1 (hold and buy)")
print("- To get sell signals (-1), agents need to output action 0")
print("- This suggests the trained agents rarely choose action 0, hence no sell signals")