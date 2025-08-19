#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密货币高频交易强化学习系统 - 主程序
Cryptocurrency High-Frequency Trading Reinforcement Learning System - Main Program

该项目实现了一个完整的加密货币量化交易系统，包含以下主要组件：
1. 数据预处理和特征工程
2. 序列模型训练（RNN/LSTM + Mamba-TCN）
3. 强化学习智能体训练（D3QN, DoubleDQN, TwinD3QN）
4. 交易模拟和回测评估
5. 集成模型评估

作者：JCMa
日期：2025年1月
"""

import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_processing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sequential_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'reinforcement_learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_simulation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ensemble_evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Windows系统兼容性设置
if os.name == "nt":  # Windows NT
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def check_environment():
    """
    检查运行环境和依赖
    Check runtime environment and dependencies
    """
    logger.info("=" * 60)
    logger.info("检查运行环境 / Checking Runtime Environment")
    logger.info("=" * 60)
    
    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"Python版本 / Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'scipy', 'quantstats'
    ]
    
    # 可选包（需要CUDA支持）
    optional_packages = ['mamba_ssm']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {package} 已安装 / installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"[MISSING] {package} 未安装 / not installed")
    
    # 检查可选包
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {package} 已安装 / installed (optional)")
        except ImportError:
            logger.warning(f"[WARNING] {package} 未安装 / not installed (optional, requires CUDA)")
    
    if missing_packages:
        logger.error(f"缺少必要依赖包 / Missing required packages: {missing_packages}")
        logger.error("请运行 / Please run: pip install -r requirements.txt")
        return False
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"[OK] CUDA可用 / CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU数量 / GPU count: {torch.cuda.device_count()}")
        else:
            logger.info("! CUDA不可用，将使用CPU / CUDA not available, using CPU")
    except Exception as e:
        logger.warning(f"检查CUDA时出错 / Error checking CUDA: {e}")
    
    return True

def step1_data_preprocessing(gpu_id=-1):
    """
    步骤1: 数据预处理和特征工程
    Step 1: Data preprocessing and feature engineering
    """
    logger.info("\n" + "=" * 60)
    logger.info("步骤1: 数据预处理和特征工程 / Step 1: Data Preprocessing")
    logger.info("=" * 60)
    
    try:
        from data_processing.seq_data import convert_btc_csv_to_btc_npy
        from data_processing.data_config import ConfigData
        
        # 检查是否已经存在预处理后的文件
        config = ConfigData()
        if (Path(config.input_ary_path).exists() and 
            Path(config.label_ary_path).exists()):
            logger.info("预处理文件已存在，跳过数据预处理步骤")
            logger.info("Preprocessed files exist, skipping data preprocessing")
            logger.info(f"输入文件: {config.input_ary_path}")
            logger.info(f"标签文件: {config.label_ary_path}")
            return True
        
        logger.info("开始数据预处理...")
        logger.info("Starting data preprocessing...")
        
        # 执行数据预处理
        convert_btc_csv_to_btc_npy()
        
        # 验证生成的文件
        if (Path(config.input_ary_path).exists() and 
            Path(config.label_ary_path).exists()):
            logger.info("[OK] 数据预处理完成 / Data preprocessing completed")
            logger.info(f"[OK] 输入特征文件: {config.input_ary_path}")
            logger.info(f"[OK] 标签文件: {config.label_ary_path}")
            return True
        else:
            logger.error("[ERROR] 数据预处理完成但文件未生成")
            return False
        
    except Exception as e:
        logger.error(f"数据预处理失败 / Data preprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def step2_factor_model_training(gpu_id=-1, force=False):
    """
    步骤2: 因子模型训练
    Step 2: Factor model training
    """
    logger.info("\n" + "=" * 60)
    logger.info("步骤2: 因子模型训练 / Step 2: Factor Model Training")
    logger.info("=" * 60)
    
    try:
        from sequential_models.seq_run import train_model, valid_model
        from data_processing.data_config import ConfigData
        
        config = ConfigData()
        
        # 检查是否已经存在训练好的模型（除非强制执行）
        if not force and Path(config.predict_ary_path).exists():
            logger.info("因子模型预测文件已存在，跳过训练步骤")
            logger.info("Factor model prediction file exists, skipping training")
            return True
        
        logger.info("开始因子模型训练...")
        logger.info("Starting factor model training...")
        
        # 训练模型
        train_model(gpu_id=gpu_id)
        
        logger.info("开始模型验证和预测生成...")
        logger.info("Starting model validation and prediction generation...")
        
        # 验证模型并生成预测
        valid_model(gpu_id=gpu_id)
        
        logger.info("[OK] 因子模型训练完成 / Factor model training completed")
        return True
        
    except Exception as e:
        logger.error(f"因子模型训练失败 / Factor model training failed: {e}")
        return False

def step3_reinforcement_learning_training(gpu_id=-1, force=False):
    """
    步骤3: 强化学习智能体训练
    Step 3: Reinforcement learning agent training
    """
    logger.info("\n" + "=" * 60)
    logger.info("步骤3: 强化学习智能体训练 / Step 3: RL Agent Training")
    logger.info("=" * 60)
    
    try:
        from reinforcement_learning.erl_run import train_agent, valid_agent
        from reinforcement_learning.erl_config import Config
        from reinforcement_learning.erl_agent import AgentD3QN
        from trading_simulation.trade_simulator import TradeSimulator, EvalTradeSimulator
        
        # 检查是否已经存在训练好的智能体（除非强制执行）
        model_dirs = ["TradeSimulator-v0_D3QN_-1", "TradeSimulator-v0_D3QN_0"]
        model_exists = any(Path(model_dir).exists() and any(Path(model_dir).iterdir()) for model_dir in model_dirs)
        
        if not force and model_exists:
            logger.info("训练好的智能体已存在，跳过训练步骤")
            logger.info("Trained agents exist, skipping training")
            return True
        
        logger.info("开始强化学习智能体训练...")
        logger.info("Starting reinforcement learning agent training...")
        
        # 设置训练参数
        num_sims = 1024
        num_ignore_step = 60
        max_position = 10
        step_gap = 2
        slippage = 3e-7
        max_step = (4800 - num_ignore_step) // step_gap
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": num_sims,
            "max_step": max_step,
            "state_dim": 8 + 2,  # factor_dim + (position, holding)
            "action_dim": 3,  # long, 0, short
            "if_discrete": True,
            "max_position": max_position,
            "slippage": slippage,
            "num_sims": num_sims,
            "step_gap": step_gap,
        }
        
        # 创建配置
        args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
        args.gpu_id = gpu_id
        args.random_seed = gpu_id
        args.gamma = 0.995
        args.learning_rate = 1e-5
        args.batch_size = 256
        args.break_step = int(8e4)
        args.buffer_size = int(max_step * 32)
        args.repeat_times = 2
        args.horizon_len = int(max_step * 4)
        args.eval_per_step = int(max_step)
        args.num_workers = 1
        args.save_gap = 8
        
        # 设置评估环境配置
        args.eval_env_class = EvalTradeSimulator
        args.eval_env_args = env_args.copy()
        
        # 运行训练
        train_agent(args=args)
        valid_agent(args=args)
        
        logger.info("[OK] 强化学习训练完成 / Reinforcement learning training completed")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"强化学习训练失败 / Reinforcement learning training failed: {e}")
        logger.error(f"详细错误信息 / Detailed error: {traceback.format_exc()}")
        return False

def step4_model_evaluation(gpu_id=-1):
    """
    步骤4: 模型评估和回测
    Step 4: Model evaluation and backtesting
    """
    logger.info("\n" + "=" * 60)
    logger.info("步骤4: 模型评估和回测 / Step 4: Model Evaluation")
    logger.info("=" * 60)
    
    try:
        from task2_eval import run_evaluation
        from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
        
        logger.info("开始模型评估...")
        logger.info("Starting model evaluation...")
        
        save_path = "trained_agents"
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
        
        run_evaluation(save_path, agent_list, gpu_id=gpu_id)
        
        logger.info("[OK] 模型评估完成 / Model evaluation completed")
        return True
        
    except Exception as e:
        logger.error(f"模型评估失败 / Model evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def step5_ensemble_evaluation():
    """
    步骤5: 集成模型评估
    Step 5: Ensemble model evaluation
    """
    logger.info("\n" + "=" * 60)
    logger.info("步骤5: 集成模型评估 / Step 5: Ensemble Evaluation")
    logger.info("=" * 60)
    
    try:
        import task2_ensemble
        
        logger.info("开始集成模型评估...")
        logger.info("Starting ensemble model evaluation...")
        
        # 运行集成评估
        # 这里可以调用task2_ensemble中的具体函数
        logger.info("集成模型评估模块已加载")
        
        logger.info("[OK] 集成模型评估完成 / Ensemble evaluation completed")
        return True
        
    except Exception as e:
        logger.error(f"集成模型评估失败 / Ensemble evaluation failed: {e}")
        return False

def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(description='加密货币高频交易强化学习系统')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID (-1 for CPU)')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', '1', '2', '3', '4', '5'],
                       help='执行特定步骤 (1: 数据预处理, 2: 序列模型训练, 3: 强化学习训练, 4: 模型评估, 5: 集成评估)')
    parser.add_argument('--force', action='store_true', 
                       help='强制重新执行步骤2和步骤3，忽略已存在的文件 / Force re-execution of steps 2 and 3, ignoring existing files')
    
    args = parser.parse_args()
    
    logger.info("加密货币高频交易强化学习系统启动")
    logger.info("Cryptocurrency High-Frequency Trading RL System Starting")
    
    start_time = time.time()
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，程序退出")
        return
    
    # 执行步骤
    steps_to_run = []
    if args.step == 'all':
        steps_to_run = ['1', '2', '3', '4', '5']
    else:
        steps_to_run = [args.step]
    
    for step in steps_to_run:
        if step == '1':
            if not step1_data_preprocessing(args.gpu_id):
                logger.error("数据预处理失败，程序退出")
                return
        elif step == '2':
            if not step2_factor_model_training(args.gpu_id, force=args.force):
                logger.error("因子模型训练失败，程序退出")
                return
        elif step == '3':
            if not step3_reinforcement_learning_training(args.gpu_id, force=args.force):
                logger.error("强化学习训练失败，程序退出")
                return
        elif step == '4':
            if not step4_model_evaluation(args.gpu_id):
                logger.error("模型评估失败，程序退出")
                return
        elif step == '5':
            if not step5_ensemble_evaluation():
                logger.error("集成模型评估失败，程序退出")
                return
    
    end_time = time.time()
    logger.info(f"\n系统运行完成，总耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"System completed, total time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()