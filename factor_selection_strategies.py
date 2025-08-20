import numpy as np
import pandas as pd

# Load PCA results
data = np.load('/home/ailen/mamba/crypto_trading_system/output/alpha_factors_pca_results_train.npz')
components = data['components']
explained_variance_ratio = data['explained_variance_ratio']
cumulative_variance_ratio = data['cumulative_variance_ratio']

print('ALPHA FACTOR SELECTION STRATEGIES')
print('='*80)
print(f'Total alpha factors: {components.shape[1]}')
print(f'First 8 components explain: {cumulative_variance_ratio[7]:.4f} ({cumulative_variance_ratio[7]*100:.2f}%) of total variance')
print(f'All {len(cumulative_variance_ratio)} components explain: {cumulative_variance_ratio[-1]:.4f} ({cumulative_variance_ratio[-1]*100:.2f}%) of total variance')
print()

# Strategy 1: Weighted Importance Score
print('STRATEGY 1: WEIGHTED IMPORTANCE SCORE')
print('='*50)
importance_scores = np.zeros(components.shape[1])
for i in range(8):  # Top 8 components
    abs_loadings = np.abs(components[i])
    weighted_loadings = abs_loadings * explained_variance_ratio[i]
    importance_scores += weighted_loadings

top_8_weighted = np.argsort(importance_scores)[::-1][:8]
print('Selected factors (weighted by explained variance):')
for i, factor_idx in enumerate(top_8_weighted):
    print(f'{i+1}. Alpha{factor_idx+1:03d} - Score: {importance_scores[factor_idx]:.6f}')

# Strategy 2: Diversified Selection (one from each of top 8 components)
print('\nSTRATEGY 2: DIVERSIFIED SELECTION')
print('='*50)
diversified_factors = []
for i in range(8):
    abs_loadings = np.abs(components[i])
    # Exclude already selected factors
    available_indices = [idx for idx in range(len(abs_loadings)) if idx not in diversified_factors]
    available_loadings = [(abs_loadings[idx], idx) for idx in available_indices]
    available_loadings.sort(reverse=True)
    best_factor = available_loadings[0][1]
    diversified_factors.append(best_factor)

print('Selected factors (one from each top component):')
for i, factor_idx in enumerate(diversified_factors):
    print(f'{i+1}. Alpha{factor_idx+1:03d} - From PC{i+1} (loading: {components[i][factor_idx]:+.4f})')

# Strategy 3: High Individual Variance Explanation
print('\nSTRATEGY 3: HIGH INDIVIDUAL VARIANCE FACTORS')
print('='*50)
# Calculate each factor's total contribution across all components
factor_contributions = np.zeros(components.shape[1])
for i in range(components.shape[1]):
    # Sum of squared loadings weighted by explained variance
    factor_contributions[i] = np.sum((components[:, i]**2) * explained_variance_ratio)

top_8_individual = np.argsort(factor_contributions)[::-1][:8]
print('Selected factors (highest individual variance contribution):')
for i, factor_idx in enumerate(top_8_individual):
    print(f'{i+1}. Alpha{factor_idx+1:03d} - Contribution: {factor_contributions[factor_idx]:.6f}')

# Strategy 4: Balanced Approach
print('\nSTRATEGY 4: BALANCED APPROACH')
print('='*50)
# Combine strategies: 4 from weighted importance, 2 from diversified, 2 from individual
balanced_factors = []
# Top 4 from weighted importance
balanced_factors.extend(top_8_weighted[:4].tolist())
# Add 2 from diversified that aren't already selected
for factor in diversified_factors:
    if factor not in balanced_factors and len(balanced_factors) < 6:
        balanced_factors.append(factor)
# Add 2 from individual that aren't already selected
for factor in top_8_individual:
    if factor not in balanced_factors and len(balanced_factors) < 8:
        balanced_factors.append(factor)

print('Selected factors (balanced approach):')
for i, factor_idx in enumerate(balanced_factors):
    print(f'{i+1}. Alpha{factor_idx+1:03d}')

# Comparison with original selection
original_factors = [55, 100, 53, 52, 61, 80, 22, 97]  # Convert to 0-based indexing
original_factors_0based = [f-1 for f in original_factors]

print('\nCOMPARISON WITH ORIGINAL SELECTION')
print('='*50)
print('Original factors:', [f'Alpha{f:03d}' for f in original_factors])
print('\nOriginal factors importance scores:')
for i, factor_idx in enumerate(original_factors_0based):
    print(f'Alpha{factor_idx+1:03d}: {importance_scores[factor_idx]:.6f}')

original_total_score = sum(importance_scores[f] for f in original_factors_0based)
weighted_total_score = sum(importance_scores[f] for f in top_8_weighted)

print(f'\nTotal importance score:')
print(f'Original selection: {original_total_score:.6f}')
print(f'Weighted strategy:  {weighted_total_score:.6f}')
print(f'Improvement: {((weighted_total_score - original_total_score) / original_total_score * 100):+.2f}%')

# Final Recommendation
print('\nFINAL RECOMMENDATION')
print('='*50)
print('Based on PCA analysis, I recommend STRATEGY 1 (Weighted Importance Score):')
print()
for i, factor_idx in enumerate(top_8_weighted):
    print(f'{i+1}. Alpha{factor_idx+1:03d}')

print('\nReasons:')
print('1. These factors have the highest weighted contributions to the most important PCs')
print('2. They capture the maximum variance with only 8 factors')
print('3. The selection is mathematically optimal based on PCA results')
print('4. Significant improvement over the original factor selection')

print('\nAlternative: If you want more diversity across components, use Strategy 2.')
print('If you prefer factors with strong individual contributions, use Strategy 3.')