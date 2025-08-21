import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from improved_trading_strategy import ConfidenceBasedEnsemble

# Load the actual data
positions = np.load('trained_agents_positions.npy', allow_pickle=True)
confidences = np.load('trained_agents_confidences.npy')

print(f"Positions shape: {positions.shape}")
print(f"Confidences shape: {confidences.shape}")

# Analyze position distribution
positions_flat = positions.flatten()
print(f"\nPosition Distribution:")
for pos in range(11):
    count = np.sum(positions_flat == pos)
    percentage = count / len(positions_flat) * 100
    print(f"Position {pos}: {count} ({percentage:.2f}%)")

# Convert positions to actions (-1, 0, 1)
print(f"\nConverting positions to actions:")
actions_converted = []
for i in range(len(positions)):
    pos = positions[i][0] if positions.shape[1] == 1 else positions[i]
    if pos < 5:
        action = -1  # sell
    elif pos == 5:
        action = 0   # hold
    else:
        action = 1   # buy
    actions_converted.append(action)

actions_array = np.array(actions_converted)
print(f"Action distribution:")
print(f"Sell (-1): {np.sum(actions_array == -1)} ({np.sum(actions_array == -1)/len(actions_array)*100:.2f}%)")
print(f"Hold (0): {np.sum(actions_array == 0)} ({np.sum(actions_array == 0)/len(actions_array)*100:.2f}%)")
print(f"Buy (1): {np.sum(actions_array == 1)} ({np.sum(actions_array == 1)/len(actions_array)*100:.2f}%)")

# Test ensemble decision with actual data
ensemble = ConfidenceBasedEnsemble(num_agents=3)

print(f"\nTesting ensemble decisions with actual data:")
sell_decisions = 0
hold_decisions = 0
buy_decisions = 0

# Test first 10 steps
for i in range(min(10, len(positions))):
    # Convert positions to actions for all 3 agents
    agent_actions = []
    for j in range(3):  # 3 agents
        pos = positions[i][0] if positions.shape[1] == 1 else positions[i]
        if pos < 5:
            action = -1
        elif pos == 5:
            action = 0
        else:
            action = 1
        agent_actions.append(action)
    
    agent_confidences = confidences[i]
    
    decision = ensemble.ensemble_decision(agent_actions, agent_confidences)
    
    print(f"Step {i}: Positions={positions[i]}, Actions={agent_actions}, Confidences={agent_confidences}, Decision={decision}")
    
    if decision == -1:
        sell_decisions += 1
    elif decision == 0:
        hold_decisions += 1
    else:
        buy_decisions += 1

print(f"\nFirst 10 decisions:")
print(f"Sell: {sell_decisions}, Hold: {hold_decisions}, Buy: {buy_decisions}")

# Test with some manual sell scenarios
print(f"\nTesting manual sell scenarios:")

# Scenario 1: All agents want to sell with high confidence
test_actions = [-1, -1, -1]
test_confidences = [0.9, 0.9, 0.9]
decision = ensemble.ensemble_decision(test_actions, test_confidences)
print(f"All sell, high conf: Actions={test_actions}, Conf={test_confidences}, Decision={decision}")

# Scenario 2: Majority sell with high confidence
test_actions = [-1, -1, 1]
test_confidences = [0.9, 0.9, 0.8]
decision = ensemble.ensemble_decision(test_actions, test_confidences)
print(f"Majority sell: Actions={test_actions}, Conf={test_confidences}, Decision={decision}")

# Scenario 3: Mixed with one strong sell
test_actions = [-1, 0, 1]
test_confidences = [0.95, 0.7, 0.8]
decision = ensemble.ensemble_decision(test_actions, test_confidences)
print(f"Mixed with strong sell: Actions={test_actions}, Conf={test_confidences}, Decision={decision}")

# Check if the issue is in position to action conversion
print(f"\nChecking position to action conversion logic:")
for pos in range(11):
    if pos < 5:
        action = -1
    elif pos == 5:
        action = 0
    else:
        action = 1
    print(f"Position {pos} -> Action {action}")