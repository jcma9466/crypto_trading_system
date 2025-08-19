#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的交易策略实现
Improved Trading Strategy Implementation

主要改进:
1. 风险调整奖励函数
2. 动态仓位管理
3. 智能信号锁定
4. 增强状态空间
5. 基于置信度的集成决策
"""

import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Dict
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> float:
        """计算MACD指标"""
        if len(prices) < slow:
            return 0.0
        
        ema_fast = pd.Series(prices).ewm(span=fast).mean().iloc[-1]
        ema_slow = pd.Series(prices).ewm(span=slow).mean().iloc[-1]
        
        return ema_fast - ema_slow
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """计算平均真实波幅"""
        if len(close) < 2:
            return 0.01
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        
        return atr

class EnhancedStateSpace:
    """增强状态空间"""
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.price_history = deque(maxlen=lookback_window * 2)
        self.volume_history = deque(maxlen=lookback_window)
        
    def update_history(self, price: float, volume: float = 0.0):
        """更新历史数据"""
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def get_enhanced_state(self, current_position: float = 0.0) -> np.ndarray:
        """获取增强状态特征"""
        if len(self.price_history) < 10:
            # 如果历史数据不足，返回零向量
            return np.zeros(25)
        
        prices = np.array(list(self.price_history))
        
        # 基础价格特征
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            recent_returns = returns[-10:] if len(returns) >= 10 else returns
            # 填充到10个元素
            if len(recent_returns) < 10:
                recent_returns = np.pad(recent_returns, (10 - len(recent_returns), 0), 'constant')
        else:
            recent_returns = np.zeros(10)
        
        # 技术指标
        rsi = TechnicalIndicators.rsi(prices) / 100.0  # 归一化到[0,1]
        macd = TechnicalIndicators.macd(prices)
        
        # 布林带
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices)
        current_price = prices[-1]
        bb_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        # 波动性特征
        volatility = np.std(returns[-self.lookback_window:]) if len(returns) >= self.lookback_window else 0.01
        
        # 趋势特征
        if len(prices) >= 5:
            short_trend = (prices[-1] - prices[-5]) / prices[-5]
        else:
            short_trend = 0.0
            
        if len(prices) >= 20:
            long_trend = (prices[-1] - prices[-20]) / prices[-20]
        else:
            long_trend = 0.0
        
        # 动量特征
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
        
        # 组合状态向量
        state = np.concatenate([
            recent_returns,  # 10个元素
            [rsi, macd / 1000.0, bb_position],  # 3个元素，MACD缩放
            [volatility * 100, short_trend, long_trend],  # 3个元素
            [momentum * 100, current_position],  # 2个元素
            [len(self.price_history) / self.lookback_window]  # 1个元素，数据完整性
        ])
        
        # 确保状态向量长度为25
        if len(state) != 25:
            state = np.pad(state, (0, max(0, 25 - len(state))), 'constant')[:25]
        
        return state

class RiskAdjustedReward:
    """风险调整奖励函数"""
    
    def __init__(self, transaction_cost_rate: float = 0.001, risk_aversion: float = 0.5):
        self.transaction_cost_rate = transaction_cost_rate
        self.risk_aversion = risk_aversion
        self.return_history = deque(maxlen=100)
        
    def calculate_reward(self, old_asset: float, new_asset: float, action: int, 
                        volatility: float, position_change: float) -> float:
        """计算风险调整奖励"""
        if old_asset <= 0:
            return 0.0
        
        # 基础收益率
        raw_return = (new_asset - old_asset) / old_asset
        self.return_history.append(raw_return)
        
        # 交易成本惩罚
        transaction_cost = abs(position_change) * self.transaction_cost_rate
        
        # 风险惩罚（基于波动性）
        risk_penalty = volatility * self.risk_aversion * abs(position_change)
        
        # 夏普比率奖励
        sharpe_bonus = 0.0
        if len(self.return_history) >= 10:
            returns_array = np.array(list(self.return_history))
            if np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array)
                sharpe_bonus = max(0, sharpe_ratio) * 0.1
        
        # 持仓时间奖励（鼓励适度持仓）
        holding_bonus = 0.0
        if abs(position_change) < 0.1:  # 小幅调整或持仓
            holding_bonus = 0.001
        
        # 综合奖励
        total_reward = raw_return - transaction_cost - risk_penalty + sharpe_bonus + holding_bonus
        
        return total_reward

class VolatilityBasedPositionSizing:
    """基于波动性的仓位管理"""
    
    def __init__(self, base_position: float = 50000, max_position: float = 500000, 
                 min_position: float = 5000):
        self.base_position = base_position
        self.max_position = max_position
        self.min_position = min_position
        
    def calculate_position_size(self, volatility: float, confidence: float, 
                               account_balance: float) -> float:
        """计算动态仓位大小"""
        # 波动性调整因子（波动性越高，仓位越小）
        vol_factor = min(1.0, 0.02 / max(volatility, 0.001))
        
        # 置信度调整因子
        conf_factor = max(0.1, min(1.0, confidence))
        
        # 账户余额调整因子（避免过度杠杆）
        balance_factor = min(1.0, account_balance / 100000)  # 假设基准账户为10万
        
        # 计算仓位大小
        position_size = self.base_position * vol_factor * conf_factor * balance_factor
        
        # 限制在合理范围内
        position_size = max(self.min_position, min(position_size, self.max_position))
        
        # 确保不超过账户余额的50%
        max_allowed = account_balance * 0.8
        position_size = min(position_size, max_allowed)
        
        return position_size

class AdaptiveSignalLock:
    """自适应信号锁定机制"""
    
    def __init__(self, window_size: int = 20, volatility_threshold: float = 0.02):
        self.window_size = window_size
        self.volatility_threshold = volatility_threshold
        self.signal_window = deque(maxlen=window_size)
        self.lock_steps = 0
        self.lock_signal = 0
        
    def should_lock_signal(self, current_signal: int, market_volatility: float, 
                          trend_strength: float) -> Tuple[bool, int, int]:
        """判断是否应该锁定信号"""
        # 更新信号窗口
        self.signal_window.append(current_signal)
        
        # 如果当前处于锁定状态
        if self.lock_steps > 0:
            self.lock_steps -= 1
            return True, self.lock_signal, self.lock_steps
        
        # 检查是否需要新的锁定
        if len(self.signal_window) < self.window_size:
            return False, current_signal, 0
        
        # 计算信号强度
        signals = list(self.signal_window)
        signal_strength = abs(sum(signals)) / len(signals)
        
        # 动态锁定条件
        should_lock = False
        lock_duration = 0
        
        # 条件1：强烈的单向信号且低波动性
        if signal_strength > 0.6 and market_volatility < self.volatility_threshold:
            should_lock = True
            lock_duration = min(50, int(signal_strength * 100))  # 最多锁定50步
        
        # 条件2：极强的趋势信号
        if signal_strength > 0.8 and abs(trend_strength) > 0.01:
            should_lock = True
            lock_duration = min(30, int(signal_strength * 50))
        
        if should_lock:
            self.lock_steps = lock_duration
            self.lock_signal = 1 if sum(signals) > 0 else -1
            return True, self.lock_signal, self.lock_steps
        
        return False, current_signal, 0

class ConfidenceBasedEnsemble:
    """基于置信度的集成决策"""
    
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.performance_history = {i: deque(maxlen=100) for i in range(num_agents)}
        self.confidence_threshold = 0.3  # 降低置信度阈值
        
    def update_performance(self, agent_id: int, performance: float):
        """更新智能体性能历史"""
        self.performance_history[agent_id].append(performance)
    
    def calculate_dynamic_weights(self) -> np.ndarray:
        """计算动态权重"""
        weights = np.ones(self.num_agents)
        
        for agent_id in range(self.num_agents):
            if len(self.performance_history[agent_id]) > 10:
                recent_performance = list(self.performance_history[agent_id])[-20:]
                avg_performance = np.mean(recent_performance)
                weights[agent_id] = max(0.1, avg_performance + 1.0)  # 确保权重为正
        
        # 归一化权重
        weights = weights / np.sum(weights)
        return weights
    
    def ensemble_decision(self, actions: List[int], confidences: List[float]) -> int:
        """集成决策"""
        if len(actions) != self.num_agents or len(confidences) != self.num_agents:
            return 0  # 默认持仓
        
        # 获取动态权重
        weights = self.calculate_dynamic_weights()
        
        # 置信度加权
        conf_weights = np.array(confidences) * weights
        
        # 加权投票
        action_scores = {-1: 0, 0: 0, 1: 0}
        for i, action in enumerate(actions):
            action_scores[action] += conf_weights[i]
        
        # 选择得分最高的动作
        best_action = max(action_scores.keys(), key=lambda k: action_scores[k])
        
        # 如果最高得分的动作置信度太低，且不是持仓动作，则选择持仓
        max_score = action_scores[best_action]
        if max_score < self.confidence_threshold and best_action != 0:
            return 0
        
        return best_action

class DynamicStopLoss:
    """动态止损管理"""
    
    def __init__(self, atr_multiplier: float = 2.0, max_loss_pct: float = 0.05):
        self.atr_multiplier = atr_multiplier
        self.max_loss_pct = max_loss_pct
        self.entry_prices = {}
        
    def set_entry_price(self, position_id: str, entry_price: float, atr: float):
        """设置入场价格和止损"""
        stop_loss_distance = atr * self.atr_multiplier
        self.entry_prices[position_id] = {
            'entry_price': entry_price,
            'stop_loss_distance': stop_loss_distance
        }
    
    def should_stop_loss(self, position_id: str, current_price: float, 
                        position_type: str) -> bool:
        """判断是否应该止损"""
        if position_id not in self.entry_prices:
            return False
        
        entry_info = self.entry_prices[position_id]
        entry_price = entry_info['entry_price']
        stop_distance = entry_info['stop_loss_distance']
        
        if position_type == 'long':
            stop_price = entry_price - stop_distance
            return current_price <= stop_price
        elif position_type == 'short':
            stop_price = entry_price + stop_distance
            return current_price >= stop_price
        
        return False

class ImprovedTradingStrategy:
    """改进的交易策略主类"""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_position = 0.0
        
        # 组件初始化
        self.state_space = EnhancedStateSpace()
        self.reward_calculator = RiskAdjustedReward()
        self.position_sizer = VolatilityBasedPositionSizing()
        self.signal_lock = AdaptiveSignalLock()
        self.ensemble = ConfidenceBasedEnsemble()
        self.stop_loss = DynamicStopLoss()
        
        # 交易记录
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def process_market_data(self, price: float, volume: float = 0.0) -> np.ndarray:
        """处理市场数据并返回状态"""
        self.state_space.update_history(price, volume)
        return self.state_space.get_enhanced_state(self.current_position)
    
    def make_trading_decision(self, agent_actions: List[int], agent_confidences: List[float],
                            current_price: float, market_volatility: float) -> Dict:
        """做出交易决策"""
        # 集成决策
        ensemble_action = self.ensemble.ensemble_decision(agent_actions, agent_confidences)
        
        # 计算趋势强度
        prices = list(self.state_space.price_history)
        if len(prices) >= 10:
            trend_strength = (prices[-1] - prices[-10]) / prices[-10]
        else:
            trend_strength = 0.0
        
        # 信号锁定检查
        is_locked, final_action, lock_steps = self.signal_lock.should_lock_signal(
            ensemble_action, market_volatility, trend_strength
        )
        
        # 计算仓位大小
        avg_confidence = np.mean(agent_confidences)
        position_size = self.position_sizer.calculate_position_size(
            market_volatility, avg_confidence, self.current_balance
        )
        
        # 执行交易
        trade_result = self.execute_trade(final_action, position_size, current_price)
        
        return {
            'action': final_action,
            'position_size': position_size,
            'is_locked': is_locked,
            'lock_steps': lock_steps,
            'trade_result': trade_result,
            'ensemble_action': ensemble_action,
            'avg_confidence': avg_confidence
        }
    
    def execute_trade(self, action: int, position_size: float, current_price: float) -> Dict:
        """执行交易"""
        old_balance = self.current_balance
        old_position = self.current_position
        
        # 计算新仓位
        if action == 1:  # 买入
            target_position = position_size / current_price
        elif action == -1:  # 卖出
            target_position = -position_size / current_price
        else:  # 持仓
            target_position = self.current_position
        
        # 计算仓位变化
        position_change = target_position - self.current_position
        
        # 检查资金是否足够
        required_cash = abs(position_change) * current_price
        if required_cash > self.current_balance * 0.95:  # 保留5%现金
            position_change = 0  # 资金不足，不执行交易
            target_position = self.current_position
        
        # 更新仓位和余额
        self.current_position = target_position
        self.current_balance -= position_change * current_price * (1 + 0.001)  # 包含交易成本
        
        # 计算总资产
        total_asset = self.current_balance + self.current_position * current_price
        
        # 计算奖励
        old_asset = old_balance + old_position * current_price
        volatility = np.std(list(self.state_space.price_history)[-20:]) if len(self.state_space.price_history) >= 20 else 0.01
        reward = self.reward_calculator.calculate_reward(
            old_asset, total_asset, action, volatility, abs(position_change)
        )
        
        # 记录交易
        if abs(position_change) > 0.001:  # 有实际交易发生
            self.trade_history.append({
                'price': current_price,
                'action': action,
                'position_change': position_change,
                'balance': self.current_balance,
                'total_asset': total_asset,
                'reward': reward
            })
            self.performance_metrics['total_trades'] += 1
        
        return {
            'position_change': position_change,
            'new_balance': self.current_balance,
            'new_position': self.current_position,
            'total_asset': total_asset,
            'reward': reward
        }
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if len(self.trade_history) == 0:
            return self.performance_metrics
        
        # 计算总收益率
        final_asset = self.trade_history[-1]['total_asset']
        total_return = (final_asset - self.initial_balance) / self.initial_balance
        
        # 计算胜率
        winning_trades = sum(1 for trade in self.trade_history if trade['reward'] > 0)
        win_rate = winning_trades / len(self.trade_history) if self.trade_history else 0
        
        # 计算夏普比率
        returns = [trade['reward'] for trade in self.trade_history]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # 计算最大回撤
        assets = [trade['total_asset'] for trade in self.trade_history]
        peak = np.maximum.accumulate(assets)
        drawdown = (assets - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        self.performance_metrics.update({
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_asset': final_asset,
            'total_trades': len(self.trade_history)
        })
        
        return self.performance_metrics

if __name__ == "__main__":
    # 示例使用
    strategy = ImprovedTradingStrategy(initial_balance=100000)
    
    # 模拟市场数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    for i, price in enumerate(prices):
        # 处理市场数据
        state = strategy.process_market_data(price)
        
        # 模拟智能体决策
        agent_actions = [np.random.choice([-1, 0, 1]) for _ in range(3)]
        agent_confidences = [np.random.uniform(0.3, 0.9) for _ in range(3)]
        market_volatility = 0.02
        
        # 做出交易决策
        decision = strategy.make_trading_decision(
            agent_actions, agent_confidences, price, market_volatility
        )
        
        if i % 100 == 0:
            print(f"Step {i}: Price={price:.2f}, Action={decision['action']}, "
                  f"Asset={decision['trade_result']['total_asset']:.2f}")
    
    # 打印性能摘要
    performance = strategy.get_performance_summary()
    print("\n=== 性能摘要 ===")
    for key, value in performance.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")