#!/usr/bin/env python3

import subprocess
import re
from collections import Counter

print("Running evaluation and analyzing agent actions...")
print("This may take a few minutes...")

# Run the evaluation and capture output
result = subprocess.run(
    ['python3', 'ensemble_evaluation/task2_eval.py', '--rl-only'],
    cwd='/home/jcma_legion/crypto_trading_system',
    capture_output=True,
    text=True
)

# Extract individual agent actions
action_pattern = r'Individual agent actions: \[(\d+), (\d+), (\d+)\]'
matches = re.findall(action_pattern, result.stdout)

if not matches:
    print("No agent actions found in output. Checking stderr...")
    matches = re.findall(action_pattern, result.stderr)

if matches:
    print(f"Found {len(matches)} action samples")
    
    # Analyze action distribution for each agent
    agent1_actions = [int(match[0]) for match in matches]
    agent2_actions = [int(match[1]) for match in matches]
    agent3_actions = [int(match[2]) for match in matches]
    
    print("\nAgent Action Distribution:")
    print("Action 0 -> Sell (-1 after conversion)")
    print("Action 1 -> Hold (0 after conversion)")
    print("Action 2 -> Buy (1 after conversion)")
    print()
    
    for i, actions in enumerate([agent1_actions, agent2_actions, agent3_actions], 1):
        counter = Counter(actions)
        total = len(actions)
        print(f"Agent {i}:")
        for action in [0, 1, 2]:
            count = counter.get(action, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  Action {action}: {count:4d} ({percentage:5.1f}%)")
        print()
    
    # Analyze converted actions
    print("Converted Action Distribution (after -1 conversion):")
    for i, actions in enumerate([agent1_actions, agent2_actions, agent3_actions], 1):
        converted = [a - 1 for a in actions]
        counter = Counter(converted)
        total = len(converted)
        print(f"Agent {i}:")
        for action in [-1, 0, 1]:
            count = counter.get(action, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            action_name = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}[action]
            print(f"  {action_name:4s} ({action:2d}): {count:4d} ({percentage:5.1f}%)")
        print()
    
    # Analyze ensemble decisions
    ensemble_decisions = []
    for match in matches:
        actions = [int(match[i]) - 1 for i in range(3)]  # Convert to -1, 0, 1
        # Simple majority vote
        action_counts = Counter(actions)
        most_common = action_counts.most_common(1)[0][0]
        ensemble_decisions.append(most_common)
    
    ensemble_counter = Counter(ensemble_decisions)
    total_decisions = len(ensemble_decisions)
    print("Ensemble Decision Distribution (majority vote):")
    for action in [-1, 0, 1]:
        count = ensemble_counter.get(action, 0)
        percentage = (count / total_decisions) * 100 if total_decisions > 0 else 0
        action_name = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}[action]
        print(f"  {action_name:4s} ({action:2d}): {count:4d} ({percentage:5.1f}%)")
    
else:
    print("No agent actions found in the output.")
    print("\nStdout sample:")
    print(result.stdout[:1000])
    print("\nStderr sample:")
    print(result.stderr[:1000])