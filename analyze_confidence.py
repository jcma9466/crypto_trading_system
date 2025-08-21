import numpy as np
import matplotlib.pyplot as plt

# Load confidence data
confidences = np.load('trained_agents_confidences.npy')
print(f"Confidence data shape: {confidences.shape}")
print(f"Number of time steps: {len(confidences)}")

# Analyze confidence statistics
confidences_flat = confidences.flatten()
print(f"\nConfidence Statistics:")
print(f"Mean confidence: {np.mean(confidences_flat):.4f}")
print(f"Std confidence: {np.std(confidences_flat):.4f}")
print(f"Min confidence: {np.min(confidences_flat):.4f}")
print(f"Max confidence: {np.max(confidences_flat):.4f}")
print(f"Median confidence: {np.median(confidences_flat):.4f}")

# Check confidence distribution
print(f"\nConfidence Distribution:")
print(f"Confidence < 0.15: {np.sum(confidences_flat < 0.15)} ({np.sum(confidences_flat < 0.15)/len(confidences_flat)*100:.2f}%)")
print(f"Confidence < 0.3: {np.sum(confidences_flat < 0.3)} ({np.sum(confidences_flat < 0.3)/len(confidences_flat)*100:.2f}%)")
print(f"Confidence >= 0.3: {np.sum(confidences_flat >= 0.3)} ({np.sum(confidences_flat >= 0.3)/len(confidences_flat)*100:.2f}%)")
print(f"Confidence >= 0.5: {np.sum(confidences_flat >= 0.5)} ({np.sum(confidences_flat >= 0.5)/len(confidences_flat)*100:.2f}%)")

# Analyze per-agent confidence
if len(confidences.shape) == 2:
    num_agents = confidences.shape[1]
    print(f"\nPer-Agent Confidence Analysis (Number of agents: {num_agents}):")
    for i in range(num_agents):
        agent_conf = confidences[:, i]
        print(f"Agent {i}: Mean={np.mean(agent_conf):.4f}, Std={np.std(agent_conf):.4f}, Min={np.min(agent_conf):.4f}, Max={np.max(agent_conf):.4f}")

# Load actions to analyze relationship between actions and confidences
try:
    positions = np.load('trained_agents_positions.npy', allow_pickle=True)
    print(f"\nPosition data shape: {positions.shape}")
    
    # Analyze confidence for different position values
    if len(confidences.shape) == 2 and len(positions) == len(confidences):
        print(f"\nConfidence by Position Analysis:")
        for pos in range(11):  # positions 0-10
            pos_mask = positions == pos
            if np.any(pos_mask):
                pos_confidences = confidences[pos_mask]
                print(f"Position {pos}: Count={np.sum(pos_mask)}, Mean Conf={np.mean(pos_confidences):.4f}, Std={np.std(pos_confidences):.4f}")
except Exception as e:
    print(f"Could not load position data: {e}")

# Create histogram
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(confidences_flat, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=0.15, color='red', linestyle='--', label='Threshold 0.15')
plt.axvline(x=0.3, color='orange', linestyle='--', label='Old Threshold 0.3')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot for per-agent confidence
if len(confidences.shape) == 2:
    plt.subplot(2, 2, 2)
    plt.boxplot([confidences[:, i] for i in range(confidences.shape[1])], 
                labels=[f'Agent {i}' for i in range(confidences.shape[1])])
    plt.ylabel('Confidence')
    plt.title('Per-Agent Confidence Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

# Time series of average confidence
plt.subplot(2, 2, 3)
if len(confidences.shape) == 2:
    avg_confidence = np.mean(confidences, axis=1)
else:
    avg_confidence = confidences
plt.plot(avg_confidence)
plt.axhline(y=0.15, color='red', linestyle='--', label='Threshold 0.15')
plt.axhline(y=0.3, color='orange', linestyle='--', label='Old Threshold 0.3')
plt.xlabel('Time Step')
plt.ylabel('Average Confidence')
plt.title('Average Confidence Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Confidence vs Position scatter plot
if len(confidences.shape) == 2:
    try:
        positions = np.load('trained_agents_positions.npy', allow_pickle=True)
        if len(positions) == len(confidences):
            plt.subplot(2, 2, 4)
            avg_conf_per_step = np.mean(confidences, axis=1)
            plt.scatter(positions, avg_conf_per_step, alpha=0.5, s=1)
            plt.axhline(y=0.15, color='red', linestyle='--', label='Threshold 0.15')
            plt.axhline(y=0.3, color='orange', linestyle='--', label='Old Threshold 0.3')
            plt.xlabel('Position Signal')
            plt.ylabel('Average Confidence')
            plt.title('Confidence vs Position Signal')
            plt.legend()
            plt.grid(True, alpha=0.3)
    except:
        pass

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis complete. Confidence analysis plot saved as 'confidence_analysis.png'")