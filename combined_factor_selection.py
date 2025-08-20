import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import seaborn as sns

print('合并因子选择分析')
print('='*60)

# 原始因子选择和当前优化后的因子
original_factors = [55, 100, 53, 52, 61, 80, 22, 97]
current_factors = [9, 10, 73, 93, 49, 87]

# 合并并去重
combined_factors = list(set(original_factors + current_factors))
combined_factors.sort()

print(f'原始因子选择: {original_factors}')
print(f'当前优化因子: {current_factors}')
print(f'合并后的因子: {combined_factors}')
print(f'合并后因子数量: {len(combined_factors)}')

# 使用seq_data.py中相同的方法计算合并后的因子
print('\n使用seq_data.py方法计算合并因子数据...')
try:
    import sys
    sys.path.append('/home/ailen/mamba/crypto_trading_system')
    from data_processing.seq_data import TechIndicator
    
    # 读取原始数据
    df = pd.read_csv('/home/ailen/mamba/crypto_trading_system/data/BTC_15m.csv')
    print(f'原始数据形状: {df.shape}')
    
    # 计算合并后的因子（只计算需要的14个因子）
    indicator = TechIndicator(df=df)
    combined_alpha_data = []
    
    print(f'计算合并后的{len(combined_factors)}个alpha因子...')
    for factor_idx in combined_factors:
        try:
            alpha_func = getattr(indicator, f'alpha{factor_idx:03d}')
            alpha_values = alpha_func()
            if alpha_values is not None and len(alpha_values) > 0:
                combined_alpha_data.append(alpha_values)
                print(f'Alpha{factor_idx:03d}: 计算成功，数据长度 {len(alpha_values)}')
            else:
                print(f'Alpha{factor_idx:03d}: 计算失败或返回空值')
                # 用零填充以保持维度一致
                combined_alpha_data.append(np.zeros(len(df)))
        except Exception as e:
            print(f'Alpha{factor_idx:03d}: 计算错误 - {str(e)}')
            # 用零填充以保持维度一致
            combined_alpha_data.append(np.zeros(len(df)))
            
    # 转换为numpy数组
    if combined_alpha_data:
        combined_data = np.column_stack(combined_alpha_data)
        print(f'合并因子数据形状: {combined_data.shape}')
        
        # 检查数据有效性
        valid_count = np.sum(np.any(combined_data != 0, axis=0))
        print(f'有效因子数量: {valid_count}/{len(combined_factors)}')
        
    else:
        raise Exception('无法计算任何alpha因子')
        
except Exception as e:
    print(f'计算失败: {str(e)}')
    print('尝试从现有PCA结果文件加载...')
    
    # 备选方案：从PCA结果文件加载
    try:
        pca_results = np.load('/home/ailen/mamba/crypto_trading_system/output/alpha_factors_pca_results_train.npz')
        original_data = pca_results['original_data']
        print(f'从PCA结果加载的数据形状: {original_data.shape}')
        
        if original_data.shape[1] >= max(combined_factors):
            combined_data = original_data[:, [f-1 for f in combined_factors]]
            print(f'提取的合并因子数据形状: {combined_data.shape}')
        else:
            raise Exception(f'PCA数据维度不足，需要{max(combined_factors)}维，实际只有{original_data.shape[1]}维')
            
    except Exception as e2:
        print(f'从PCA结果加载也失败: {str(e2)}')
        raise Exception('所有数据加载方法都失败了')

# 数据预处理
combined_data = combined_data.astype(np.float64)

# 移除包含NaN或inf的行
valid_mask = np.isfinite(combined_data).all(axis=1)
combined_data = combined_data[valid_mask]
print(f'清理后数据形状: {combined_data.shape}')

# 标准化数据
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)

# 执行PCA分析
print('\n执行PCA分析...')
pca = PCA()
pca_result = pca.fit_transform(combined_data_scaled)

# 计算解释方差比例
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f'\nPCA结果:')
print('='*50)
for i in range(min(10, len(explained_variance_ratio))):
    print(f'PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)')

print(f'\n前8个主成分累积解释方差: {cumulative_variance_ratio[7]:.4f} ({cumulative_variance_ratio[7]*100:.2f}%)')

# 计算因子重要性得分
print('\n计算因子重要性得分...')
components = pca.components_[:8]  # 前8个主成分
factor_importance = np.zeros(len(combined_factors))

for i in range(8):
    # 加权载荷，权重为解释方差比例
    weighted_loadings = np.abs(components[i]) * explained_variance_ratio[i]
    factor_importance += weighted_loadings

# 创建因子重要性DataFrame
factor_df = pd.DataFrame({
    'Factor': [f'Alpha{f:03d}' for f in combined_factors],
    'Factor_Index': combined_factors,
    'Importance_Score': factor_importance,
    'In_Original': [f in original_factors for f in combined_factors],
    'In_Current': [f in current_factors for f in combined_factors]
})

# 按重要性排序
factor_df = factor_df.sort_values('Importance_Score', ascending=False)

print('\n因子重要性排序:')
print('='*80)
print(factor_df.to_string(index=False))

# 选择前8个最重要的因子
top_8_factors = factor_df.head(8)['Factor_Index'].tolist()

print(f'\n推荐的8个最优因子: {top_8_factors}')

# 验证选择的合理性
print('\n验证分析:')
print('='*50)

# 检查多重共线性
selected_data = combined_data[:, [combined_factors.index(f) for f in top_8_factors]]
selected_corr = np.corrcoef(selected_data.T)

# 计算矩阵秩
selected_rank = matrix_rank(selected_data)
print(f'选择因子的矩阵秩: {selected_rank}/8')

# 检查高相关性
high_corr_count = 0
for i in range(8):
    for j in range(i+1, 8):
        if abs(selected_corr[i, j]) > 0.8:
            high_corr_count += 1
            print(f'高相关性: Alpha{top_8_factors[i]:03d} vs Alpha{top_8_factors[j]:03d}: {selected_corr[i, j]:.4f}')

if high_corr_count == 0:
    print('✅ 未发现高相关性因子对')

# 计算条件数
cond_number = np.linalg.cond(selected_corr)
print(f'条件数: {cond_number:.2e}')

if cond_number < 100:
    print('✅ 条件数良好，多重共线性问题较小')
elif cond_number < 1000:
    print('⚠️  条件数适中，存在轻微多重共线性')
else:
    print('⚠️  条件数较大，存在多重共线性问题')

# 与原始选择和当前选择的对比
print('\n对比分析:')
print('='*50)

# 计算原始选择的重要性得分
original_importance = factor_df[factor_df['Factor_Index'].isin(original_factors)]['Importance_Score'].sum()
current_importance = factor_df[factor_df['Factor_Index'].isin(current_factors)]['Importance_Score'].sum()
optimal_importance = factor_df.head(8)['Importance_Score'].sum()

print(f'原始选择总重要性得分: {original_importance:.4f}')
print(f'当前选择总重要性得分: {current_importance:.4f}')
print(f'最优选择总重要性得分: {optimal_importance:.4f}')

print(f'\n相对于原始选择的提升:')
print(f'当前选择: +{((current_importance/original_importance - 1) * 100):.2f}%')
print(f'最优选择: +{((optimal_importance/original_importance - 1) * 100):.2f}%')

# 因子来源分析
optimal_from_original = sum(1 for f in top_8_factors if f in original_factors)
optimal_from_current = sum(1 for f in top_8_factors if f in current_factors)
optimal_new = 8 - optimal_from_original - optimal_from_current + len(set(top_8_factors) & set(original_factors) & set(current_factors))

print(f'\n最优选择的因子来源:')
print(f'来自原始选择: {optimal_from_original}个')
print(f'来自当前选择: {optimal_from_current}个')
print(f'新增因子: {optimal_new}个')

# 最终推荐
print('\n最终推荐:')
print('='*50)
print(f'推荐使用的8个alpha因子: {top_8_factors}')
print('\n代码修改建议:')
print('在 seq_data.py 中修改:')
print(f'alpha_indices = {top_8_factors}')

# 保存结果
result_dict = {
    'recommended_factors': top_8_factors,
    'factor_importance': factor_df.to_dict('records'),
    'matrix_rank': selected_rank,
    'condition_number': cond_number,
    'importance_scores': {
        'original': original_importance,
        'current': current_importance,
        'optimal': optimal_importance
    }
}

print('\n分析完成！')
print('='*60)