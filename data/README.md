# BTC 15分钟交易数据说明 / BTC 15-Minute Trading Data Documentation

## 数据概述 / Data Overview

本目录包含用于加密货币高频交易强化学习系统的BTC 15分钟K线数据。

This directory contains BTC 15-minute candlestick data for the cryptocurrency high-frequency trading reinforcement learning system.

## 数据文件 / Data Files

### 1. BTC_15m_input.npy
- **文件大小 / File Size**: 4.27 MB
- **数据形状 / Data Shape**: (280,038, 8)
- **数据类型 / Data Type**: float16
- **描述 / Description**: 技术指标特征数据，包含8个alpha因子
- **时间范围 / Time Range**: 约2917天的15分钟数据点

**特征说明 / Feature Description**:
- 8个技术指标alpha因子 (Alpha indices: 55, 100, 53, 52, 61, 80, 22, 97)
- 数据已标准化处理，范围在[-1, 1]之间
- 使用分位数标准化方法处理异常值

### 2. BTC_15m_label.npy
- **文件大小 / File Size**: 8.54 MB
- **数据形状 / Data Shape**: (279,678, 8)
- **数据类型 / Data Type**: float32
- **描述 / Description**: 价格变化标签数据，用于监督学习

**标签说明 / Label Description**:
- 基于不同时间窗口的价格变化率
- 窗口大小: (4, 8, 12, 24, 32, 40, 80, 160) 个15分钟周期
- 用于预测未来价格走势

## 数据来源 / Data Source

数据来源于PostgreSQL数据库中的BTC现货交易数据，经过以下处理步骤：

Data sourced from BTC spot trading data in PostgreSQL database, processed through:

1. **原始数据提取 / Raw Data Extraction**:
   - 从数据库提取datetime, high, low, volume等字段
   - Extract datetime, high, low, volume fields from database

2. **特征工程 / Feature Engineering**:
   - 计算中间价格 (midpoint = (high + low) / 2)
   - 计算价差 (spread = high - low)
   - 生成买卖盘距离和成交量数据

3. **技术指标计算 / Technical Indicator Calculation**:
   - 计算8个alpha因子技术指标
   - 使用TechIndicator类进行批量计算
   - 处理NaN值和异常值

4. **数据标准化 / Data Normalization**:
   - 使用分位数标准化 (quantile normalization)
   - 将数据范围限制在[-1, 1]之间
   - 转换为float16格式以节省存储空间

## 使用方法 / Usage

```python
import numpy as np

# 加载输入特征数据
input_data = np.load('data/BTC_15m_input.npy')
print(f"输入数据形状: {input_data.shape}")

# 加载标签数据
label_data = np.load('data/BTC_15m_label.npy')
print(f"标签数据形状: {label_data.shape}")

# 数据已经预处理完成，可直接用于模型训练
```

## 数据质量 / Data Quality

- ✅ 无NaN值 / No NaN values
- ✅ 数据已标准化 / Data normalized
- ✅ 异常值已处理 / Outliers handled
- ✅ 时间序列连续 / Continuous time series

## 注意事项 / Notes

1. **存储优化 / Storage Optimization**: 使用float16格式减少存储空间
2. **内存使用 / Memory Usage**: 加载时会自动转换为适当的数据类型
3. **版本兼容 / Version Compatibility**: 兼容NumPy 1.19+
4. **更新频率 / Update Frequency**: 数据可通过重新运行数据预处理步骤更新

## 相关文件 / Related Files

- `../data_processing/seq_data.py`: 数据预处理脚本
- `../data_processing/data_config.py`: 数据配置文件
- `../main.py --step 1`: 数据预处理命令

## 技术支持 / Technical Support

如需重新生成数据文件，请运行：

To regenerate data files, run:

```bash
python main.py --step 1
```

数据生成时间约3-5分钟，取决于数据库连接速度。

Data generation takes approximately 3-5 minutes, depending on database connection speed.