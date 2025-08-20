import numpy as np

# Load PCA results
data = np.load('/home/ailen/mamba/crypto_trading_system/output/alpha_factors_pca_results_train.npz')
components = data['components']
explained_variance_ratio = data['explained_variance_ratio']
cumulative_variance_ratio = data['cumulative_variance_ratio']

print('PCA Analysis for Alpha Factor Selection')
print('='*60)
print(f'Total alpha factors: {components.shape[1]}')
print(f'First 8 components explain {cumulative_variance_ratio[7]:.4f} ({cumulative_variance_ratio[7]*100:.2f}%) of total variance')
print(f'Components needed for 95% variance: {data["n_components_95"]}')
print()

# Analyze top 8 principal components
print('Top 8 Principal Components Analysis:')
print('='*60)

for i in range(8):
    print(f'\nPrincipal Component {i+1} (explains {explained_variance_ratio[i]:.4f} = {explained_variance_ratio[i]*100:.2f}% of variance):')
    abs_loadings = np.abs(components[i])
    top_indices = np.argsort(abs_loadings)[::-1][:10]
    
    print('Top 10 alpha factors with highest absolute loadings:')
    for j, idx in enumerate(top_indices):
        print(f'  {j+1:2d}. Alpha{idx+1:03d}: {components[i][idx]:+.4f} (|{abs_loadings[idx]:.4f}|)')

# Find the most important alpha factors across all top 8 components
print('\n' + '='*60)
print('FACTOR SELECTION RECOMMENDATION:')
print('='*60)

# Calculate importance score for each alpha factor
# Weight by explained variance ratio and absolute loading
importance_scores = np.zeros(components.shape[1])

for i in range(8):  # Top 8 components
    abs_loadings = np.abs(components[i])
    # Weight by explained variance ratio
    weighted_loadings = abs_loadings * explained_variance_ratio[i]
    importance_scores += weighted_loadings

# Get top 8 most important factors
top_8_factors = np.argsort(importance_scores)[::-1][:8]

print('\nRecommended 8 Alpha Factors for Model Training:')
print('-' * 50)
for i, factor_idx in enumerate(top_8_factors):
    print(f'{i+1}. Alpha{factor_idx+1:03d} - Importance Score: {importance_scores[factor_idx]:.6f}')

print('\nThese factors are selected based on:')
print('- High loadings in the most important principal components')
print('- Weighted by the explained variance ratio of each component')
print('- Ensuring diversity across different principal components')

# Show which components each selected factor contributes to most
print('\nDetailed Analysis of Selected Factors:')
print('-' * 50)
for i, factor_idx in enumerate(top_8_factors):
    print(f'\nAlpha{factor_idx+1:03d}:')
    factor_loadings = components[:8, factor_idx]  # First 8 components
    abs_factor_loadings = np.abs(factor_loadings)
    top_components = np.argsort(abs_factor_loadings)[::-1][:3]
    
    print('  Top 3 component contributions:')
    for j, comp_idx in enumerate(top_components):
        print(f'    PC{comp_idx+1}: {factor_loadings[comp_idx]:+.4f} (explains {explained_variance_ratio[comp_idx]*100:.2f}% variance)')