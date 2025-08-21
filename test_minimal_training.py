#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化强化学习训练测试
用于快速验证训练流程是否正常工作
"""

import os
import sys
import torch
import numpy as np

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_processing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sequential_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'reinforcement_learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_simulation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ensemble_evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_minimal_rl_training():
    """
    最小化强化学习训练测试
    使用极小的参数进行快速验证
    """
    print("开始最小化强化学习训练测试...")
    
    try:
        from reinforcement_learning.erl_run import train_agent, valid_agent
        from reinforcement_learning.erl_config import Config
        from reinforcement_learning.erl_agent import AgentD3QN
        from trading_simulation.trade_simulator import TradeSimulator, EvalTradeSimulator
        
        # 极小的训练参数
        num_sims = 4  # 只使用4个并行环境
        num_ignore_step = 60
        max_position = 1
        step_gap = 2
        slippage = 3e-7
        max_step = 100  # 极小的步数
        
        print(f"使用参数: num_sims={num_sims}, max_step={max_step}")
        
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
        args.gpu_id = 0
        args.random_seed = 0
        args.gamma = 0.995
        args.learning_rate = 1e-4
        args.batch_size = 16  # 极小的批次大小
        args.break_step = 1000  # 极小的训练步数
        args.buffer_size = 500  # 极小的缓冲区大小
        args.repeat_times = 1
        args.horizon_len = 200
        args.eval_per_step = 500
        args.num_workers = 1
        args.save_gap = 2
        
        # 设置评估环境配置
        args.eval_env_class = EvalTradeSimulator
        args.eval_env_args = env_args.copy()
        
        print("开始训练...")
        # 运行训练
        train_agent(args=args)
        print("训练完成，开始验证...")
        valid_agent(args=args)
        
        print("[OK] 最小化强化学习训练测试完成")
        return True
        
    except Exception as e:
        import traceback
        print(f"最小化训练测试失败: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        return False

if __name__ == '__main__':
    test_minimal_rl_training()