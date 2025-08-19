import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reinforcement_learning.erl_config import Config, build_env
from trading_simulation.trade_simulator import EvalTradeSimulator
from reinforcement_learning.erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from utils.metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown

# 导入基准模型和图片生成模块
try:
    from benchmark_models import run_benchmark_comparison
except ImportError:
    print("Warning: benchmark_models.py not found. Benchmark comparison will be skipped.")
    run_benchmark_comparison = None

try:
    from generate_article_figures import generate_all_article_figures
except ImportError:
    print("Warning: generate_article_figures.py not found. Figure generation will be skipped.")
    generate_all_article_figures = None

try:
    from data_config import ConfigData
except ImportError:
    print("Warning: data_config.py not found. Using fallback configuration.")
    ConfigData = None


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config):
        # 设置随机种子
        x = 42
        torch.manual_seed(x)
        np.random.seed(x)
        torch.cuda.manual_seed_all(x)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        self.save_path = save_path
        self.agent_classes = agent_classes

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args or {}, gpu_id=args.gpu_id)

        self.current_btc = 0
        # 使用固定的初始资金，因为 Config 类没有 starting_cash 属性
        starting_cash = 1e6
        self.cash = [starting_cash]
        self.btc_assets = [0]
        # self.net_assets = [torch.tensor(starting_cash, device=self.device)]
        self.net_assets = [starting_cash]
        self.starting_cash = starting_cash

        # Ensure state dimensions match
        if args.env_args and self.state_dim != args.env_args.get("state_dim", self.state_dim):
            print(f"Warning: Agent state_dim ({self.state_dim}) != Environment state_dim ({args.env_args['state_dim']})")
            self.state_dim = args.env_args["state_dim"]
        self.trade_log_path = os.path.join(save_path, "trade_log.csv")
        # 创建CSV文件并写入表头
        with open(self.trade_log_path, 'w', encoding='utf-8') as f:
            f.write("Step,Action,Price,Cash,BTC,BTC_Value,Total\n")
        self.signal_count = 0  # 连续信号计数
        self.last_signal = 0   # 上一个信号
        self.signal_window = []  # 存储移动窗口内的信号
        self.window_size = 79   # 移动窗口大小
        self.lock_signal = 0    # 锁定的信号
        self.lock_steps = 0     # 锁定剩余步数

    def load_agents(self):
        args = self.args
        for agent_class in self.agent_classes:
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            print(f"Loading agent from: {os.path.abspath(cwd)}")
            # 检查模型文件是否存在
            model_path = os.path.join(cwd, "act.pth")
            if os.path.exists(model_path):
                print(f"Found model file: {model_path}")
            else:
                print(f"Warning: Model file not found at {model_path}")
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)

    def multi_trade(self):
        """Evaluation loop using ensemble of agents"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for _ in range(trade_env.max_step-1):
            actions = []
            intermediate_state = last_state

            # Collect actions from each agent
            for agent in agents:
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                tensor_action = tensor_q_values.argmax(dim=1)
                action = tensor_action.detach().cpu().unsqueeze(1)
                actions.append(action)

            # Debug agent decisions
            print(f"Individual agent actions: {[a.item() for a in actions]}")
            action = self._ensemble_action(actions=actions)
            action_int = action.item() - 1

            state, reward, done, _ = trade_env.step(action=action)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            mid_price = trade_env.price_ary[trade_env.step_i, 2]
            # 确保 mid_price 是 tensor 类型，如果不是则转换
            if not torch.is_tensor(mid_price):
                mid_price = torch.tensor(mid_price, device=self.device)
            else:
                mid_price = mid_price.to(self.device)
            mid_price_value = mid_price.item()

            new_cash = self.cash[-1]
            act1=0
            act2=0
            # 信号锁定逻辑
            if self.lock_steps > 0:
                action_int = self.lock_signal
                self.lock_steps -= 1
            else:
                # 信号锁定逻辑
                original_action = action_int  # 保存原始信号
                
                # 更新移动窗口（无论是否处于锁定状态都继续更新）
                self.signal_window.append(original_action)
                if len(self.signal_window) > self.window_size:
                    self.signal_window.pop(0)
                
                # 计算窗口内的空头信号数量
                short_signals = sum(1 for x in self.signal_window if x < 0)
                
                # 检查是否需要锁定或继续锁定
                if short_signals >= 9:
                    self.lock_steps = 500
                    action_int = -1  # 强制执行空头
                    print(f"Locking/Maintaining short position due to {short_signals} short signals in window")
                elif self.lock_steps > 0:
                    self.lock_steps -= 1
                    action_int = -1  # 维持空头

            # Modified trading logic with better price checks
            if action_int > 0:  # Buy signal
                act1 += 1
                if self.current_btc <0:
                    slippage = self.args.env_args.get("slippage", 0) if self.args.env_args else 0
                    trade_value = 32*mid_price_value * (1 + slippage)
                    new_cash -= trade_value
                    self.current_btc += 32
                    print(f"Executed BUY at {mid_price_value:.2f}")
                elif self.current_btc == 0:  # Can afford to buy
                    slippage = self.args.env_args.get("slippage", 0) if self.args.env_args else 0
                    trade_value = 16*mid_price_value * (1 + slippage)
                    new_cash -= trade_value
                    self.current_btc += 16
                    print(f"Executed BUY at {mid_price_value:.2f}")
            elif action_int < 0:  # Sell signal
                act2 += 1
                if self.current_btc > 0:
                    slippage = self.args.env_args.get("slippage", 0) if self.args.env_args else 0
                    trade_value = 32*mid_price_value * (1 - slippage)
                    new_cash += trade_value
                    self.current_btc -= 32
                    print(f"Executed SELL at {mid_price_value:.2f}")
                elif self.current_btc == 0:
                    slippage = self.args.env_args.get("slippage", 0) if self.args.env_args else 0
                    trade_value = 16*mid_price_value * (1 - slippage)
                    new_cash += trade_value
                    self.current_btc -= 16
                    print(f"Executed SELL at {mid_price_value:.2f}")


            self.cash.append(new_cash)
            btc_value = self.current_btc * mid_price.item()  # Calculate BTC value in cash
            self.btc_assets.append(btc_value)
            self.net_assets.append(new_cash + btc_value)  # Total assets = cash + BTC value

            # 记录每一步的交易信息（无论是否有实际交易）
            step_num = len(self.net_assets)
            btc_log = (f"Step {step_num}: Action={action_int}, "
                  f"Price={mid_price.item():.2f}, Cash={new_cash:.2f}, "
                  f"BTC={self.current_btc}, BTC Value={btc_value:.2f}, "
                  f"Total={self.net_assets[-1]:.2f}")
            
            # 只打印前几步和有实际交易的步骤
            if step_num <= 10 or action_int != 0:
                print(btc_log)
            
            # 将每一步的信息都写入CSV文件
            with open(self.trade_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{step_num},{action_int},{mid_price.item():.2f},"
                       f"{new_cash:.2f},{self.current_btc},{btc_value:.2f},{self.net_assets[-1]:.2f}\n")
            
            last_state = state

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)

            last_price = mid_price
            current_btcs.append(self.current_btc)

        # Convert positions to CPU before saving if they're on GPU
        positions_np = []
        for p in positions:
            if torch.is_tensor(p):
                positions_np.append(p.cpu().numpy())
            elif isinstance(p, np.ndarray):
                positions_np.append(p)
            else:
                positions_np.append(np.array(p))
        
        # Save results
        np.save(f"{self.save_path}_positions.npy", np.array(positions_np, dtype=object))
        np.save(f"{self.save_path}_net_assets.npy", np.array(self.net_assets, dtype=np.float64))
        np.save(f"{self.save_path}_btc_positions.npy", np.array(self.btc_assets, dtype=np.float64))
        np.save(f"{self.save_path}_correct_predictions.npy", np.array(correct_pred, dtype=np.int32))

        # Compute metrics
        # 在计算指标之前添加调试信息
        print(f"Net assets history: {self.net_assets}")
        print(f"Returns statistics:")
        
        # 确保 net_assets 是 numpy 数组
        net_assets_array = np.array(self.net_assets, dtype=np.float64)
        
        # 计算收益率，避免除零错误
        if len(net_assets_array) > 1:
            # 过滤掉零值或负值，避免除零错误
            valid_indices = net_assets_array[:-1] > 0
            if np.any(valid_indices):
                # 确保索引维度匹配
                diff_values = np.diff(net_assets_array)
                base_values = net_assets_array[:-1]
                returns = diff_values[valid_indices] / base_values[valid_indices]
            else:
                returns = np.array([])
        else:
            returns = np.array([])
            
        print(f"- Mean return: {np.mean(returns) if len(returns) > 0 else 0}")
        print(f"- Std return: {np.std(returns) if len(returns) > 0 else 0}")
        print(f"- Min return: {np.min(returns) if len(returns) > 0 else 0}")
        print(f"- Max return: {np.max(returns) if len(returns) > 0 else 0}")
        print(f"- Number of trades: {len([x for x in action_ints if x != 0])}")
        print(f"-act1: {act1}，act2: {act2}")
        
        # 添加检查，避免除以零导致的无穷大
        if len(returns) == 0 or np.std(returns) == 0:
            print("Warning: Returns have zero standard deviation or no valid returns!")
            final_sharpe_ratio = 0
        else:
            final_sharpe_ratio = sharpe_ratio(returns)
            
        if len(returns) == 0:
            print("Warning: No trading returns generated!")
            final_max_drawdown = 0
            final_roma = 0
        else:
            final_max_drawdown = max_drawdown(returns)
            final_roma = return_over_max_drawdown(returns)
    
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")

    def _ensemble_action(self, actions):
        """Returns the majority action among agents. Our code uses majority voting, you may change this to increase performance."""
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)


def run_evaluation(save_path, agent_list, gpu_id=-1):
    print(f"\nInitializing evaluation...")
    
    # 检查数据配置
    from data_processing.data_config import ConfigData
    config_data = ConfigData()
    print(f"\nData Configuration:")
    print(f"Factor array path: {os.path.abspath(config_data.predict_ary_path)}")
    print(f"Price data path: {os.path.abspath(config_data.csv_path)}")
    
    # 检查数据文件是否存在
    for path in [config_data.predict_ary_path, config_data.csv_path]:
        if os.path.exists(path):
            print(f"Found data file: {path}")
            print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
        else:
            print(f"Warning: Data file not found at {path}")
    
    model_dir = os.path.abspath(save_path)
    print(f"\nLoading models from: {model_dir}")
    if os.path.exists(model_dir):
        print(f"Model directory contents:")
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                print(f"  Directory: {item}")
                for subitem in os.listdir(item_path):
                    print(f"    - {subitem}")
            else:
                print(f"  File: {item}")
    else:
        print(f"Warning: Model directory not found at {model_dir}")
    
    print(f"Agent list: {[agent.__name__ for agent in agent_list]}")

    print(f"Using GPU ID: {gpu_id}")

    num_sims = 1
    num_ignore_step = 800
    max_position = 1
    step_gap = 16
    slippage = 7e-7

    max_step = (64000 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "dataset_path": config_data.csv_path,  # 使用配置中的路径
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (64, 32)  # 修正为正确的元组大小
    # 移除 starting_cash 属性，因为 Config 类没有这个属性

    ensemble_evaluator = EnsembleEvaluator(
        save_path,
        agent_list,
        args,
    )
    print("\nLoading trained agents...")
    ensemble_evaluator.load_agents()
    print("Starting trading simulation...")
    ensemble_evaluator.multi_trade()


def run_benchmark_evaluation():
    """
    运行基准模型评估和图片生成
    """
    print("\n=== Running Benchmark Model Evaluation ===")
    
    # 检查数据文件
    if ConfigData is None:
        print("Error: data_config module not available")
        return None
    config_data = ConfigData()
    
    # 加载价格数据
    try:
        # 优先从数据库加载15分钟K线数据
        print("Loading price data from PostgreSQL database...")
        price_data = config_data.load_btc_data_from_db()
        
        # 如果数据库加载失败，回退到CSV文件
        if price_data is None:
            print("Database loading failed, trying CSV file...")
            if not os.path.exists(config_data.csv_path):
                print(f"Error: Price data file not found at {config_data.csv_path}")
                return None
            print(f"Loading price data from {config_data.csv_path}...")
            price_data = pd.read_csv(config_data.csv_path)
        
        print(f"Loaded {len(price_data)} rows of price data")
        print(f"Columns: {list(price_data.columns)}")
        
        # 检查数据格式并提取价格序列
        if 'midpoint' in price_data.columns:
            # 使用midpoint作为价格序列
            prices = price_data['midpoint'].values
            prices_array = np.array(prices)
            print(f"Using 'midpoint' column as price data. Price range: {np.min(prices_array):.2f} - {np.max(prices_array):.2f}")
        elif 'close' in price_data.columns:
            # 使用close价格
            prices = price_data['close'].values
            prices_array = np.array(prices)
            print(f"Using 'close' column as price data. Price range: {np.min(prices_array):.2f} - {np.max(prices_array):.2f}")
        else:
            print("Error: No suitable price column found. Expected 'midpoint' or 'close' column.")
            return None
            
        # 检查价格数据的有效性
        prices_array = np.array(prices)
        if len(prices_array) == 0 or np.any(np.isnan(prices_array)) or np.any(prices_array <= 0):
            print("Error: Invalid price data (empty, NaN, or non-positive values)")
            return None
            
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None
    
    # 运行基准模型比较
    benchmark_results = None
    if run_benchmark_comparison is not None:
        try:
            print("\nRunning benchmark model comparison...")
            benchmark_results = run_benchmark_comparison(prices_array)
            
            # 保存基准模型结果
            results_dir = "./results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存为JSON格式
            import json
            with open(os.path.join(results_dir, "benchmark_results.json"), 'w') as f:
                # 转换numpy数组为列表以便JSON序列化
                json_results = {}
                for strategy, result in benchmark_results.items():
                    json_results[strategy] = {}
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            json_results[strategy][key] = value.tolist()
                        else:
                            json_results[strategy][key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
                json.dump(json_results, f, indent=2)
            
            print(f"[OK] Benchmark results saved to {results_dir}/benchmark_results.json")
            
            # 打印基准模型性能摘要
            print("\n=== Benchmark Model Performance Summary ===")
            for strategy, result in benchmark_results.items():
                print(f"\n{strategy}:")
                print(f"  Total Return: {result['total_return']*100:.2f}%")
                print(f"  Annual Return: {result['annual_return']*100:.2f}%")
                print(f"  Volatility: {result['volatility']*100:.2f}%")
                print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
                
        except Exception as e:
            print(f"Error running benchmark comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping benchmark comparison (module not available)")
    
    # 生成文章图片
    if generate_all_article_figures is not None:
        try:
            print("\nGenerating article figures...")
            if benchmark_results is not None:
                success = generate_all_article_figures(benchmark_results)
                if success:
                    print("[OK] All article figures generated successfully")
                else:
                    print("[WARNING] Some figures failed to generate")
            else:
                print("[WARNING] No benchmark results to generate figures")
        except Exception as e:
            print(f"Error generating figures: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping figure generation (module not available)")
    
    return benchmark_results

def run_complete_evaluation(save_path="trained_agents", agent_list=None, include_benchmarks=True, gpu_id=-1):
    """
    运行完整的评估，包括强化学习模型和基准模型
    
    Args:
        save_path: 训练好的智能体保存路径
        agent_list: 智能体类列表
        include_benchmarks: 是否包含基准模型评估
    """
    if agent_list is None:
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    print("\n" + "="*60)
    print("COMPLETE EVALUATION PIPELINE")
    print("="*60)
    
    # 1. 运行强化学习模型评估
    print("\n1. Running Reinforcement Learning Model Evaluation...")
    try:
        run_evaluation(save_path, agent_list, gpu_id)
        print("[OK] RL model evaluation completed")
    except Exception as e:
        print(f"Error in RL evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 运行基准模型评估（如果启用）
    benchmark_results = None
    if include_benchmarks:
        print("\n2. Running Benchmark Model Evaluation...")
        try:
            benchmark_results = run_benchmark_evaluation()
            if benchmark_results:
                print("[OK] Benchmark evaluation completed")
            else:
                print("[WARNING] Benchmark evaluation failed or skipped")
        except Exception as e:
            print(f"Error in benchmark evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n2. Skipping benchmark evaluation (disabled)")
    
    print("\n" + "="*60)
    print("EVALUATION PIPELINE COMPLETED")
    print("="*60)
    
    return benchmark_results

if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark-only":
        # 仅运行基准模型评估
        run_benchmark_evaluation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--rl-only":
        # 仅运行强化学习模型评估
        save_path = "trained_agents"
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
        run_evaluation(save_path, agent_list)
    else:
        # 运行完整评估
        save_path = "trained_agents"
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
        run_complete_evaluation(save_path, agent_list, include_benchmarks=True)
    