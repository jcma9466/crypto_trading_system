import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# 检查文件是否存在
trade_log_path = 'trained_agents/trade_log.csv'
if not os.path.exists(trade_log_path):
    print(f"Warning: {trade_log_path} not found. Creating empty plot.")
    # 创建空图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    ax1.text(0.5, 0.5, 'No trade log data available', transform=ax1.transAxes, ha='center', va='center')
    ax2.text(0.5, 0.5, 'No trade log data available', transform=ax2.transAxes, ha='center', va='center')
    ax3.text(0.5, 0.5, 'No trade log data available', transform=ax3.transAxes, ha='center', va='center')
    ax1.set_title('BTC Price Over Time (with Trade Points)')
    ax2.set_title('Total Assets Over Time (with Trade Points)')
    ax3.set_title('Price vs Total Assets (Normalized)')
    plt.tight_layout()
    plt.savefig('trained_agents/trade_analysis.png')
    plt.close()
    print("Empty trade analysis chart saved.")
    exit()

# 读取CSV文件
try:
    df = pd.read_csv(trade_log_path)
except Exception as e:
    print(f"Error reading {trade_log_path}: {e}")
    exit()

# 检查数据是否为空
if len(df) == 0:
    print("Warning: Trade log is empty")
    # 创建空图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    ax1.text(0.5, 0.5, 'No trade data available', transform=ax1.transAxes, ha='center', va='center')
    ax2.text(0.5, 0.5, 'No trade data available', transform=ax2.transAxes, ha='center', va='center')
    ax3.text(0.5, 0.5, 'No trade data available', transform=ax3.transAxes, ha='center', va='center')
    ax1.set_title('BTC Price Over Time (with Trade Points)')
    ax2.set_title('Total Assets Over Time (with Trade Points)')
    ax3.set_title('Price vs Total Assets (Normalized)')
    plt.tight_layout()
    plt.savefig('trained_agents/trade_analysis.png')
    plt.close()
    print("Empty trade analysis chart saved.")
    exit()

# 创建图形和子图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# 绘制价格变化和交易点
ax1.plot(df['Step'], df['Price'], color='blue', label='Price')
# 标记买入点
buy_points = df[df['Action'] == 1]
if len(buy_points) > 0:
    ax1.scatter(buy_points['Step'], buy_points['Price'], color='red', marker='^', label='Buy', s=100)
# 标记卖出点
sell_points = df[df['Action'] == -1]
if len(sell_points) > 0:
    ax1.scatter(sell_points['Step'], sell_points['Price'], color='green', marker='v', label='Sell', s=100)
ax1.set_title('BTC Price Over Time (with Trade Points)')
ax1.set_xlabel('Step')
ax1.set_ylabel('Price')
ax1.grid(True)
ax1.legend()

# 绘制总资产变化和交易点
ax2.plot(df['Step'], df['Total'], color='green', label='Total Assets')
# 标记买入点
if len(buy_points) > 0:
    ax2.scatter(buy_points['Step'], buy_points['Total'], color='red', marker='^', label='Buy', s=100)
# 标记卖出点
if len(sell_points) > 0:
    ax2.scatter(sell_points['Step'], sell_points['Total'], color='black', marker='v', label='Sell', s=100)
ax2.set_title('Total Assets Over Time (with Trade Points)')
ax2.set_xlabel('Step')
ax2.set_ylabel('Total Assets')
ax2.grid(True)
ax2.legend()

# 计算相关性和拟合度
# 检查数据的有效性
price_std = df['Price'].std()
total_std = df['Total'].std()

if price_std == 0 or total_std == 0 or np.isnan(price_std) or np.isnan(total_std):
    print("Warning: Price or Total assets have zero or invalid standard deviation")
    correlation = 0.0
    r_squared = 0.0
    slope = 0.0
    intercept = 0.0
    p_value = 1.0
    # 创建空的归一化数据
    price_normalized = np.zeros(len(df))
    total_normalized = np.zeros(len(df))
else:
    # 标准化数据以便比较
    price_normalized = (df['Price'] - df['Price'].mean()) / price_std
    total_normalized = (df['Total'] - df['Total'].mean()) / total_std
    
    # 计算皮尔逊相关系数
    try:
        corr_matrix = np.corrcoef(price_normalized, total_normalized)
        correlation = corr_matrix[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    # 计算R方值（拟合优度）
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(price_normalized, total_normalized)
        slope = float(slope)
        intercept = float(intercept)
        r_value = float(r_value)
        r_squared = r_value * r_value
        if np.isnan(r_squared):
            r_squared = 0.0
    except:
        slope = 0.0
        intercept = 0.0
        r_squared = 0.0
        p_value = 1.0

# 绘制拟合度对比图
if len(price_normalized) > 0 and len(total_normalized) > 0 and not (price_std == 0 or total_std == 0):
    ax3.scatter(price_normalized, total_normalized, color='gray', alpha=0.5, label='Data Points')
    fitted_line = slope * np.array(price_normalized, dtype=float) + intercept
    ax3.plot(price_normalized, fitted_line, color='red', label='Fitted Line')
else:
    ax3.text(0.5, 0.5, 'No valid data to display', transform=ax3.transAxes, ha='center', va='center')

ax3.set_title(f'Price vs Total Assets (Normalized)\nCorrelation: {correlation:.4f}, R²: {r_squared:.4f}')
ax3.set_xlabel('Normalized Price')
ax3.set_ylabel('Normalized Total Assets')
ax3.grid(True)
ax3.legend()

# 调整子图间距
plt.tight_layout()

# 保存图片
plt.savefig('trained_agents/trade_analysis.png')
plt.close()

# 打印统计结果
print(f"交易分析完成，图表已保存到 trained_agents/trade_analysis.png")
print(f"统计分析结果：")
print(f"皮尔逊相关系数: {correlation:.4f}")
print(f"R方值（拟合优度）: {r_squared:.4f}")
print(f"斜率: {slope:.4f}")
print(f"截距: {intercept:.4f}")
print(f"P值: {p_value:.4f}")