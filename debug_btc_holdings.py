import numpy as np

# 加载BTC持仓数据
btc_positions = np.load('trained_agents_btc_positions.npy')
print(f"BTC positions shape: {btc_positions.shape}")
print(f"Initial BTC holding: {btc_positions[0]:.6f}")
print(f"Final BTC holding: {btc_positions[-1]:.6f}")
print(f"Min BTC holding: {np.min(btc_positions):.6f}")
print(f"Max BTC holding: {np.max(btc_positions):.6f}")

# 检查BTC持仓变化
btc_changes = np.diff(btc_positions)
increases = np.sum(btc_changes > 0)
decreases = np.sum(btc_changes < 0)
no_change = np.sum(btc_changes == 0)

print(f"\nBTC holding changes:")
print(f"Increases: {increases}")
print(f"Decreases: {decreases}")
print(f"No change: {no_change}")

# 检查是否有足够的BTC进行卖出
print(f"\nBTC availability for selling:")
print(f"Steps with BTC > 0.001: {np.sum(btc_positions > 0.001)}")
print(f"Steps with BTC > 0.01: {np.sum(btc_positions > 0.01)}")
print(f"Steps with BTC > 0.1: {np.sum(btc_positions > 0.1)}")

# 分析BTC持仓的分布
print(f"\nBTC holding distribution:")
print(f"Zero holdings: {np.sum(btc_positions == 0)} ({np.sum(btc_positions == 0)/len(btc_positions)*100:.2f}%)")
print(f"Very small holdings (< 0.001): {np.sum((btc_positions > 0) & (btc_positions < 0.001))}")
print(f"Small holdings (0.001-0.01): {np.sum((btc_positions >= 0.001) & (btc_positions < 0.01))}")
print(f"Medium holdings (0.01-0.1): {np.sum((btc_positions >= 0.01) & (btc_positions < 0.1))}")
print(f"Large holdings (>= 0.1): {np.sum(btc_positions >= 0.1)}")

# 检查现金持仓
try:
    net_assets = np.load('trained_agents_net_assets.npy')
    print(f"\nNet assets shape: {net_assets.shape}")
    print(f"Initial net assets: {net_assets[0]:.2f}")
    print(f"Final net assets: {net_assets[-1]:.2f}")
    
    # 估算现金（假设初始现金为100万）
    initial_cash = 1000000
    estimated_cash = []
    for i in range(len(net_assets)):
        btc_value = btc_positions[i] * 50000  # 假设BTC价格约50000
        cash = net_assets[i] - btc_value
        estimated_cash.append(cash)
    
    estimated_cash = np.array(estimated_cash)
    print(f"\nEstimated cash holdings:")
    print(f"Initial cash: {estimated_cash[0]:.2f}")
    print(f"Final cash: {estimated_cash[-1]:.2f}")
    print(f"Min cash: {np.min(estimated_cash):.2f}")
    print(f"Max cash: {np.max(estimated_cash):.2f}")
    
except Exception as e:
    print(f"Could not load net assets: {e}")

# 检查具体的卖出机会
print(f"\n检查前20步的BTC持仓情况:")
for i in range(min(20, len(btc_positions))):
    print(f"Step {i}: BTC={btc_positions[i]:.6f}, Can sell (>0.001)? {btc_positions[i] > 0.001}")

# 检查BTC持仓为0的连续步数
zero_sequences = []
current_sequence = 0
for btc in btc_positions:
    if btc == 0:
        current_sequence += 1
    else:
        if current_sequence > 0:
            zero_sequences.append(current_sequence)
            current_sequence = 0
if current_sequence > 0:
    zero_sequences.append(current_sequence)

if zero_sequences:
    print(f"\nZero BTC holding sequences:")
    print(f"Number of sequences: {len(zero_sequences)}")
    print(f"Longest sequence: {max(zero_sequences)} steps")
    print(f"Average sequence length: {np.mean(zero_sequences):.2f} steps")
else:
    print(f"\nNo zero BTC holding sequences found")