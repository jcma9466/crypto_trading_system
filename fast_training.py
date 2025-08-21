#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速强化学习训练脚本
使用最小化测试验证的极小参数配置，确保快速初始化和训练
"""

import os
import sys
import torch
import numpy as np
import logging

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_processing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sequential_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'reinforcement_learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_simulation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ensemble_evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fast_rl_training(gpu_id=0, force=True):
    """
    快速强化学习训练
    使用极小参数配置，确保快速初始化和训练完成
    """
    logger.info("开始快速强化学习训练...")
    
    try:
        from reinforcement_learning.erl_run import train_agent, valid_agent
        from reinforcement_learning.erl_config import Config
        from reinforcement_learning.erl_agent import AgentD3QN
        from trading_simulation.trade_simulator import TradeSimulator, EvalTradeSimulator
        
        # 快速训练参数（基于最小化测试的成功配置）
        num_sims = 8  # 极小的并行环境数量
        num_ignore_step = 60
        max_position = 5
        step_gap = 2
        slippage = 3e-7
        max_step = 500  # 大幅减少最大步数
        
        logger.info(f"使用快速参数: num_sims={num_sims}, max_step={max_step}")
        
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
        args.learning_rate = 1e-4
        args.batch_size = 32  # 小批次大小
        args.break_step = int(2e3)  # 大幅减少训练步数
        args.buffer_size = int(max_step * 4)  # 极小缓冲区
        args.repeat_times = 1
        args.horizon_len = int(max_step)  # 减少horizon长度
        args.eval_per_step = int(max_step // 2)
        args.num_workers = 1
        args.save_gap = 2
        
        # 设置评估环境配置
        args.eval_env_class = EvalTradeSimulator
        args.eval_env_args = env_args.copy()
        
        logger.info("开始快速训练...")
        # 运行训练
        train_agent(args=args)
        logger.info("训练完成，开始验证...")
        valid_agent(args=args)
        
        logger.info("[OK] 快速强化学习训练完成")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"快速训练失败: {e}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False

if __name__ == '__main__':
    fast_rl_training()