#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置管理
Project Configuration Management
"""

import os
from pathlib import Path
from typing import Dict, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"
OUTPUT_ROOT = PROJECT_ROOT.parent / "output"
RESULTS_ROOT = PROJECT_ROOT.parent / "results"
TRAINED_AGENTS_ROOT = PROJECT_ROOT.parent / "trained_agents"
ARTICLE_FIGURES_ROOT = PROJECT_ROOT.parent / "article_figures"

# 确保目录存在
for directory in [DATA_ROOT, OUTPUT_ROOT, RESULTS_ROOT, TRAINED_AGENTS_ROOT, ARTICLE_FIGURES_ROOT]:
    directory.mkdir(exist_ok=True)

class ProjectConfig:
    """
    项目配置类
    Project configuration class
    """
    
    # 数据配置
    DATA_CONFIG = {
        "btc_csv_path": DATA_ROOT / "BTC_1sec.csv",
        "input_ary_path": DATA_ROOT / "BTC_1sec_input.npy",
        "label_ary_path": DATA_ROOT / "BTC_1sec_label.npy",
        "predict_ary_path": DATA_ROOT / "BTC_1sec_predict.npy",
        "predict_pth_path": DATA_ROOT / "BTC_1sec_predict.pth",
    }
    
    # 模型配置
    MODEL_CONFIG = {
        "seq_model_output_dir": OUTPUT_ROOT,
        "rl_model_output_dir": TRAINED_AGENTS_ROOT,
        "checkpoint_interval": 32,
        "max_epochs": 6000,
    }
    
    # 强化学习配置
    RL_CONFIG = {
        "num_sims": 512,
        "num_ignore_step": 60,
        "max_position": 1,
        "step_gap": 2,
        "slippage": 7e-7,
        "gamma": 0.995,
        "learning_rate": 2e-6,
        "batch_size": 512,
        "break_step": int(8e4),
        "repeat_times": 2,
        "num_workers": 1,
        "save_gap": 8,
    }
    
    # 网络架构配置
    NETWORK_CONFIG = {
        "seq_model": {
            "inp_dim": 8,
            "mid_dim": 128,
            "out_dim": 1,
            "num_layers": 4,
            "use_signal_smoothing": True,
            "smoothing_kernel": 5,
        },
        "rl_model": {
            "state_dim": 10,  # 8 + 2 (position, holding)
            "action_dim": 3,  # long, 0, short
            "net_dims": (128, 128),
            "if_discrete": True,
        }
    }
    
    # 数据库配置（可选）
    DATABASE_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "crypto_data"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "table_name": os.getenv("DB_TABLE", "btc_1sec"),
    }
    
    # 日志配置
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "log_file": PROJECT_ROOT / "trading_system.log",
    }
    
    # 评估配置
    EVALUATION_CONFIG = {
        "metrics": ["sharpe_ratio", "max_drawdown", "return_over_max_drawdown", "total_return"],
        "benchmark_models": ["buy_and_hold", "random_walk", "moving_average"],
        "output_formats": ["csv", "json", "png"],
    }
    
    @classmethod
    def get_config(cls, section: str) -> Dict[str, Any]:
        """
        获取指定配置节
        Get specified configuration section
        """
        config_map = {
            "data": cls.DATA_CONFIG,
            "model": cls.MODEL_CONFIG,
            "rl": cls.RL_CONFIG,
            "network": cls.NETWORK_CONFIG,
            "database": cls.DATABASE_CONFIG,
            "logging": cls.LOGGING_CONFIG,
            "evaluation": cls.EVALUATION_CONFIG,
        }
        return config_map.get(section, {})
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有配置
        Get all configurations
        """
        return {
            "data": cls.DATA_CONFIG,
            "model": cls.MODEL_CONFIG,
            "rl": cls.RL_CONFIG,
            "network": cls.NETWORK_CONFIG,
            "database": cls.DATABASE_CONFIG,
            "logging": cls.LOGGING_CONFIG,
            "evaluation": cls.EVALUATION_CONFIG,
        }

# 全局配置实例
config = ProjectConfig()

# 常用路径快捷方式
DATA_PATHS = config.get_config("data")
MODEL_PATHS = config.get_config("model")
RL_PARAMS = config.get_config("rl")
NETWORK_PARAMS = config.get_config("network")