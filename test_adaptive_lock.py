import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from improved_trading_strategy import AdaptiveSignalLock

# 创建AdaptiveSignalLock实例
adaptive_lock = AdaptiveSignalLock(window_size=20, volatility_threshold=0.02)

print("Testing AdaptiveSignalLock behavior with sell signals...")

# 模拟连续的卖出信号
sell_signals = [-1] * 25  # 25个连续的卖出信号
market_volatility = 0.015  # 低波动性
trend_strength = -0.02  # 负趋势

results = []
for i, signal in enumerate(sell_signals):
    is_locked, final_action, lock_steps = adaptive_lock.should_lock_signal(
        signal, market_volatility, trend_strength
    )
    
    results.append({
        'step': i,
        'input_signal': signal,
        'is_locked': is_locked,
        'final_action': final_action,
        'lock_steps': lock_steps
    })
    
    if i < 10 or is_locked:  # 打印前10步和所有锁定情况
        print(f"Step {i}: Input={signal}, Locked={is_locked}, Final={final_action}, Steps={lock_steps}")

# 统计结果
final_actions = [r['final_action'] for r in results]
sell_count = sum(1 for action in final_actions if action == -1)
hold_count = sum(1 for action in final_actions if action == 0)
buy_count = sum(1 for action in final_actions if action == 1)

print(f"\nResults Summary:")
print(f"Total steps: {len(results)}")
print(f"Sell actions (-1): {sell_count} ({sell_count/len(results)*100:.1f}%)")
print(f"Hold actions (0): {hold_count} ({hold_count/len(results)*100:.1f}%)")
print(f"Buy actions (1): {buy_count} ({buy_count/len(results)*100:.1f}%)")

# 测试混合信号
print("\n" + "="*50)
print("Testing with mixed signals (realistic scenario)...")

# 重新创建实例
adaptive_lock2 = AdaptiveSignalLock(window_size=20, volatility_threshold=0.02)

# 模拟更真实的信号序列：75%卖出，2.5%持有，22.5%买入
np.random.seed(42)
mixed_signals = []
for _ in range(100):
    rand = np.random.random()
    if rand < 0.75:
        signal = -1  # 卖出
    elif rand < 0.775:
        signal = 0   # 持有
    else:
        signal = 1   # 买入
    mixed_signals.append(signal)

print(f"Generated {len(mixed_signals)} mixed signals")
print(f"Sell: {mixed_signals.count(-1)}, Hold: {mixed_signals.count(0)}, Buy: {mixed_signals.count(1)}")

mixed_results = []
for i, signal in enumerate(mixed_signals):
    is_locked, final_action, lock_steps = adaptive_lock2.should_lock_signal(
        signal, market_volatility, trend_strength
    )
    
    mixed_results.append({
        'step': i,
        'input_signal': signal,
        'is_locked': is_locked,
        'final_action': final_action,
        'lock_steps': lock_steps
    })
    
    if i < 10 or (is_locked and i < 50):  # 打印前10步和前50步中的锁定情况
        print(f"Step {i}: Input={signal}, Locked={is_locked}, Final={final_action}, Steps={lock_steps}")

# 统计混合信号结果
final_mixed_actions = [r['final_action'] for r in mixed_results]
sell_mixed_count = sum(1 for action in final_mixed_actions if action == -1)
hold_mixed_count = sum(1 for action in final_mixed_actions if action == 0)
buy_mixed_count = sum(1 for action in final_mixed_actions if action == 1)

print(f"\nMixed Signals Results Summary:")
print(f"Total steps: {len(mixed_results)}")
print(f"Sell actions (-1): {sell_mixed_count} ({sell_mixed_count/len(mixed_results)*100:.1f}%)")
print(f"Hold actions (0): {hold_mixed_count} ({hold_mixed_count/len(mixed_results)*100:.1f}%)")
print(f"Buy actions (1): {buy_mixed_count} ({buy_mixed_count/len(mixed_results)*100:.1f}%)")

# 检查锁定频率
locked_steps = [r for r in mixed_results if r['is_locked']]
print(f"\nLocking behavior:")
print(f"Locked steps: {len(locked_steps)} ({len(locked_steps)/len(mixed_results)*100:.1f}%)")

if locked_steps:
    locked_actions = [r['final_action'] for r in locked_steps]
    print(f"Locked actions - Sell: {locked_actions.count(-1)}, Hold: {locked_actions.count(0)}, Buy: {locked_actions.count(1)}")