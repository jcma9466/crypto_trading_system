# 加密货币高频交易强化学习系统
# Cryptocurrency High-Frequency Trading Reinforcement Learning System

## 项目概述 / Project Overview

本项目实现了一个完整的加密货币量化交易系统，结合了深度学习、强化学习和传统量化策略。系统包含五个主要模块：

This project implements a complete cryptocurrency quantitative trading system that combines deep learning, reinforcement learning, and traditional quantitative strategies. The system consists of five main modules:

1. **数据预处理和特征工程** / Data Preprocessing and Feature Engineering
2. **序列模型训练（RNN/LSTM + Mamba-TCN）** / Sequential Model Training (RNN/LSTM + Mamba-TCN)
3. **强化学习智能体训练（D3QN, DoubleDQN, TwinD3QN）** / Reinforcement Learning Agent Training (D3QN, DoubleDQN, TwinD3QN)
4. **交易模拟和回测评估** / Trading Simulation and Backtesting
5. **集成模型评估** / Ensemble Model Evaluation

## 项目结构 / Project Structure

```
crypto_trading_system/
├── main.py                    # 主程序入口 / Main program entry
├── requirements.txt           # 依赖包列表 / Dependencies
├── .env                      # 环境配置文件 / Environment configuration
├── README.md                 # 项目说明 / Project documentation
│
├── data_processing/          # 数据预处理模块 / Data preprocessing module
│   ├── seq_data.py          # 数据加载和预处理 / Data loading and preprocessing
│   └── data_config.py       # 数据配置 / Data configuration
│
├── sequential_models/        # 序列模型模块 / Sequential models module
│   ├── seq_run.py           # 序列模型训练 / Sequential model training
│   ├── seq_net.py           # 网络架构定义 / Network architecture
│   └── seq_record.py        # 训练记录 / Training records
│
├── reinforcement_learning/   # 强化学习模块 / Reinforcement learning module
│   ├── erl_run.py           # RL训练主程序 / RL training main
│   ├── erl_agent.py         # RL智能体 / RL agents
│   ├── erl_config.py        # RL配置 / RL configuration
│   ├── erl_net.py           # RL网络 / RL networks
│   ├── erl_evaluator.py     # RL评估器 / RL evaluator
│   └── erl_replay_buffer.py # 经验回放缓冲区 / Experience replay buffer
│
├── trading_simulation/       # 交易模拟模块 / Trading simulation module
│   ├── trade_simulator.py   # 交易环境模拟器 / Trading environment simulator
│   └── validate_model.py    # 模型验证 / Model validation
│
├── ensemble_evaluation/      # 集成评估模块 / Ensemble evaluation module
│   ├── task2_eval.py        # 主评估程序 / Main evaluation
│   ├── task2_ensemble.py    # 集成模型 / Ensemble models
│   └── benchmark_models.py  # 基准模型 / Benchmark models
│
├── utils/                    # 工具模块 / Utilities module
│   ├── metrics.py           # 评估指标 / Evaluation metrics
│   ├── generate_article_figures.py  # 图表生成 / Figure generation
│   ├── plot_trade_log.py    # 交易日志可视化 / Trade log visualization
│   └── analyze_results.py   # 结果分析 / Result analysis
│
└── config/                   # 配置文件 / Configuration files
    └── .gitignore           # Git忽略文件 / Git ignore
```

## 安装和使用 / Installation and Usage

### 1. 环境设置 / Environment Setup

```bash
# 创建虚拟环境 / Create virtual environment
python -m venv venv

# 激活虚拟环境 / Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖 / Install dependencies
pip install -r requirements.txt
```

### 2. 数据准备 / Data Preparation

将比特币1秒级数据文件 `BTC_1sec.csv` 放置在 `../data/` 目录下，或配置PostgreSQL数据库连接。

Place the Bitcoin 1-second data file `BTC_1sec.csv` in the `../data/` directory, or configure PostgreSQL database connection.

### 3. 运行系统 / Run System

```bash
# 运行完整流程 / Run complete pipeline
python main.py

# 运行特定步骤 / Run specific step
python main.py --step 1  # 数据预处理 / Data preprocessing
python main.py --step 2  # 序列模型训练 / Sequential model training
python main.py --step 3  # 强化学习训练 / RL training
python main.py --step 4  # 模型评估 / Model evaluation
python main.py --step 5  # 集成评估 / Ensemble evaluation

# 指定GPU / Specify GPU
python main.py --gpu_id 0
```

## 核心特性 / Key Features

### 1. 数据预处理和特征工程 / Data Preprocessing and Feature Engineering
- 支持PostgreSQL数据库和CSV文件数据源
- Alpha101因子计算
- 技术指标生成
- 数据标准化和清洗

### 2. 序列模型 / Sequential Models
- **MambaTCN网络**: 结合Mamba状态空间模型和时间卷积网络
- **注意力机制**: 多头注意力机制捕获全局依赖
- **信号平滑**: 可选的信号平滑处理

### 3. 强化学习智能体 / Reinforcement Learning Agents
- **D3QN**: 双重深度Q网络
- **DoubleDQN**: 双Q学习
- **TwinD3QN**: 双胞胎深度Q网络

### 4. 交易模拟 / Trading Simulation
- 高频交易环境模拟
- 滑点和交易成本建模
- 实时风险管理

### 5. 评估和分析 / Evaluation and Analysis
- 多维度性能指标
- 基准模型比较
- 可视化分析报告

## 输出结果 / Output Results

系统运行完成后，将生成以下结果文件：

After system completion, the following result files will be generated:

- `../data/`: 预处理后的数据文件 / Preprocessed data files
- `../output/`: 序列模型输出 / Sequential model outputs
- `../trained_agents/`: 训练好的RL智能体 / Trained RL agents
- `../results/`: 评估结果 / Evaluation results
- `../article_figures/`: 论文图表 / Article figures

## 技术栈 / Technology Stack

- **深度学习框架**: PyTorch
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib
- **数据库**: PostgreSQL (可选)
- **状态空间模型**: Mamba-SSM
- **量化分析**: QuantStats

## 注意事项 / Notes

1. 建议使用GPU加速训练过程
2. 确保有足够的内存和存储空间
3. 首次运行需要下载和预处理数据，可能需要较长时间
4. 生产环境使用前请充分测试和验证

## 许可证 / License

本项目仅供学术研究使用。

This project is for academic research purposes only.

## 联系方式 / Contact

如有问题或建议，请联系项目维护者。

For questions or suggestions, please contact the project maintainer.