#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected Portfolio Analysis Script
Analyzes portfolio performance with proper position scaling
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set English fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_trading_results():
    """Load trading results data"""
    results = {}
    
    # Find all result files
    result_files = ['trained_agents']
    
    for prefix in result_files:
        try:
            # Load NAV data
            nav_file = f"{prefix}_net_assets.npy"
            if os.path.exists(nav_file):
                nav_data = np.load(nav_file, allow_pickle=True)
                
                # Load position data
                position_file = f"{prefix}_positions.npy"
                btc_position_file = f"{prefix}_btc_positions.npy"
                
                positions = None
                btc_positions = None
                
                if os.path.exists(position_file):
                    positions = np.load(position_file, allow_pickle=True)
                        
                if os.path.exists(btc_position_file):
                    btc_positions = np.load(btc_position_file, allow_pickle=True)
                
                # Load action data
                action_file = f"{prefix}_actions.npy"
                actions = None
                if os.path.exists(action_file):
                    actions = np.load(action_file, allow_pickle=True)
                
                # Load correct predictions data
                correct_pred_file = f"{prefix}_correct_predictions.npy"
                correct_predictions = None
                if os.path.exists(correct_pred_file):
                    correct_predictions = np.load(correct_pred_file, allow_pickle=True)
                
                results[prefix] = {
                    'nav': nav_data,
                    'positions': positions,
                    'btc_positions': btc_positions,
                    'actions': actions,
                    'correct_predictions': correct_predictions
                }
                print(f"Loaded {prefix} data successfully: NAV length={len(nav_data)}")
                
                # Print data shape information
                print(f"  NAV data shape: {nav_data.shape if hasattr(nav_data, 'shape') else type(nav_data)}")
                if positions is not None:
                    print(f"  Position data shape: {positions.shape if hasattr(positions, 'shape') else type(positions)}")
                if btc_positions is not None:
                    print(f"  BTC position data shape: {btc_positions.shape if hasattr(btc_positions, 'shape') else type(btc_positions)}")
                    print(f"  BTC position range: {np.min(btc_positions):.2f} to {np.max(btc_positions):.2f}")
                    
            else:
                print(f"File not found: {nav_file}")
        except Exception as e:
            print(f"Failed to load {prefix} data: {e}")
    
    return results

def load_price_data():
    """Load BTC price data"""
    csv_files = [
        'data/BTC_15m.csv',
        'BTC_15m.csv',
        'btc15min.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                print(f"Loading price data from {csv_file}")
                df = pd.read_csv(csv_file)
                print(f"CSV columns: {list(df.columns)}")
                
                # Find timestamp column
                timestamp_cols = ['system_time', 'datetime', 'timestamp', 'time', 'date']
                timestamp_col = None
                for col in timestamp_cols:
                    if col in df.columns:
                        timestamp_col = col
                        break
                
                # Find price column
                price_cols = ['midpoint', 'mid', 'close', 'price']
                price_col = None
                for col in price_cols:
                    if col in df.columns:
                        price_col = col
                        break
                
                if price_col:
                    print(f"Using column '{price_col}' as price data")
                    if timestamp_col:
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                        return df[[timestamp_col, price_col]].rename(columns={timestamp_col: 'timestamp', price_col: 'price'})
                    else:
                        return pd.DataFrame({'price': df[price_col]})
                else:
                    print(f"No price column found in {csv_file}")
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    print("No price data found")
    return None

def calculate_performance_metrics(nav_data):
    """Calculate performance metrics"""
    if len(nav_data) == 0:
        return {}
    
    initial_nav = nav_data[0]
    final_nav = nav_data[-1]
    
    # Calculate returns
    returns = np.diff(nav_data) / nav_data[:-1]
    total_return = (final_nav - initial_nav) / initial_nav * 100
    
    # Calculate max drawdown
    cumulative_max = np.maximum.accumulate(nav_data)
    drawdowns = (nav_data - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns) * 100
    
    # Calculate Sharpe ratio (assuming daily data)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0
    
    # Calculate volatility
    volatility = np.std(returns) * 100 if len(returns) > 1 else 0
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'final_nav': final_nav
    }

def correct_btc_positions(btc_positions, nav_data, price_data):
    """Correct BTC positions based on reasonable portfolio allocation"""
    if price_data is None or len(price_data) == 0:
        print("Warning: No price data available for position correction")
        return btc_positions
    
    # Align price data with position data
    nav_length = len(nav_data)
    if len(price_data) >= nav_length:
        aligned_prices = price_data['price'].iloc[:nav_length].values
    else:
        # Extend price data if needed
        aligned_prices = np.full(nav_length, price_data['price'].iloc[-1])
        aligned_prices[:len(price_data)] = price_data['price'].values
    
    corrected_positions = np.zeros_like(btc_positions)
    
    for i in range(len(btc_positions)):
        current_nav = nav_data[i]
        current_price = aligned_prices[i]
        original_position = btc_positions[i]
        
        # Calculate what the position should be based on NAV allocation
        # Assume maximum 80% of portfolio can be in BTC
        max_btc_value = current_nav * 0.8
        max_btc_amount = max_btc_value / current_price
        
        # Scale down the original position if it's unreasonably large
        if original_position > max_btc_amount:
            # Scale down proportionally
            scale_factor = max_btc_amount / original_position
            corrected_positions[i] = original_position * scale_factor
        else:
            corrected_positions[i] = original_position
    
    print(f"Position correction applied:")
    print(f"  Original max position: {np.max(btc_positions):.2f} BTC")
    print(f"  Corrected max position: {np.max(corrected_positions):.2f} BTC")
    print(f"  Original avg position: {np.mean(btc_positions):.2f} BTC")
    print(f"  Corrected avg position: {np.mean(corrected_positions):.2f} BTC")
    
    return corrected_positions

def plot_corrected_analysis(results, price_data=None):
    """Plot corrected portfolio analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Corrected Portfolio Performance Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create time axis if price data is available
    time_axis = None
    if price_data is not None and 'timestamp' in price_data.columns:
        nav_length = len(list(results.values())[0]['nav'])
        if len(price_data) >= nav_length:
            time_axis = price_data['timestamp'].iloc[:nav_length]
        else:
            # Extend time axis if needed
            last_time = price_data['timestamp'].iloc[-1]
            time_diff = price_data['timestamp'].iloc[-1] - price_data['timestamp'].iloc[-2]
            extended_times = [last_time + time_diff * i for i in range(1, nav_length - len(price_data) + 1)]
            time_axis = pd.concat([price_data['timestamp'], pd.Series(extended_times)])
    
    # 1. NAV Comparison
    ax1 = axes[0, 0]
    for i, (strategy, data) in enumerate(results.items()):
        nav = data['nav']
        if len(nav) > 0:
            if time_axis is not None:
                ax1.plot(time_axis, nav, label=f'{strategy.replace("_", " ").title()}', 
                        color=colors[i], linewidth=2)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax1.plot(nav, label=f'{strategy.replace("_", " ").title()}', 
                        color=colors[i], linewidth=2)
    
    ax1.set_title('Net Asset Value Comparison', fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('NAV (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns Distribution
    ax2 = axes[0, 1]
    returns_data = []
    strategy_names = []
    
    for strategy, data in results.items():
        nav = data['nav']
        if len(nav) > 1:
            returns = np.diff(nav) / nav[:-1] * 100
            returns_data.append(returns)
            strategy_names.append(strategy.replace('_', ' ').title())
    
    if returns_data:
        bp = ax2.boxplot(returns_data, labels=strategy_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(returns_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title('Returns Distribution', fontweight='bold')
        ax2.set_ylabel('Returns (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # 3. Corrected BTC Position Changes with Price
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    
    for i, (strategy, data) in enumerate(results.items()):
        if data['btc_positions'] is not None:
            btc_pos = data['btc_positions']
            nav = data['nav']
            
            # Apply position correction
            corrected_pos = correct_btc_positions(btc_pos, nav, price_data)
            
            if len(corrected_pos) > 0:
                if time_axis is not None:
                    ax3.plot(time_axis, corrected_pos, label=f'{strategy.replace("_", " ").title()} BTC Position', 
                            color=colors[i], alpha=0.8, linewidth=2)
                else:
                    ax3.plot(corrected_pos, label=f'{strategy.replace("_", " ").title()} BTC Position', 
                            color=colors[i], alpha=0.8, linewidth=2)
    
    # Plot BTC price if available
    if price_data is not None and len(price_data) > 0:
        nav_length = len(list(results.values())[0]['nav'])
        if len(price_data) >= nav_length:
            price_subset = price_data.iloc[:nav_length]
            if time_axis is not None:
                ax3_twin.plot(time_axis, price_subset['price'], color='black', alpha=0.6, 
                             linewidth=1, linestyle='--', label='BTC Price')
            else:
                ax3_twin.plot(price_subset['price'], color='black', alpha=0.6, 
                             linewidth=1, linestyle='--', label='BTC Price')
            ax3_twin.set_ylabel('BTC Price (USD)', color='black')
            ax3_twin.tick_params(axis='y', labelcolor='black')
    
    ax3.set_title('Corrected BTC Position Changes vs Price', fontweight='bold')
    if time_axis is not None:
        ax3.set_xlabel('Time')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('BTC Amount')
    ax3.legend(loc='upper left')
    if price_data is not None:
        ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Trading Frequency Analysis
    ax4 = axes[1, 0]
    trade_frequencies = []
    strategy_labels = []
    
    for strategy, data in results.items():
        if data['actions'] is not None:
            actions = data['actions']
            if len(actions) > 0:
                non_hold_actions = np.sum(actions != 0)
                trade_freq = non_hold_actions / len(actions) * 100
                trade_frequencies.append(trade_freq)
                strategy_labels.append(strategy.replace('_', ' ').title())
    
    if trade_frequencies:
        bars = ax4.bar(strategy_labels, trade_frequencies, color=colors[:len(trade_frequencies)], alpha=0.7)
        ax4.set_title('Trading Frequency Comparison', fontweight='bold')
        ax4.set_ylabel('Trading Frequency (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, freq in zip(bars, trade_frequencies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{freq:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative Returns
    ax5 = axes[1, 1]
    for i, (strategy, data) in enumerate(results.items()):
        nav = data['nav']
        if len(nav) > 0:
            cumulative_returns = (nav / nav[0] - 1) * 100
            if time_axis is not None:
                ax5.plot(time_axis, cumulative_returns, label=f'{strategy.replace("_", " ").title()}', 
                        color=colors[i], linewidth=2)
            else:
                ax5.plot(cumulative_returns, label=f'{strategy.replace("_", " ").title()}', 
                        color=colors[i], linewidth=2)
    
    ax5.set_title('Cumulative Returns', fontweight='bold')
    if time_axis is not None:
        ax5.set_xlabel('Time')
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Cumulative Return (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Position vs Price Correlation
    ax6 = axes[1, 2]
    if price_data is not None:
        for i, (strategy, data) in enumerate(results.items()):
            if data['btc_positions'] is not None:
                btc_pos = data['btc_positions']
                nav = data['nav']
                corrected_pos = correct_btc_positions(btc_pos, nav, price_data)
                
                nav_length = len(nav)
                if len(price_data) >= nav_length:
                    price_subset = price_data['price'].iloc[:nav_length].values
                    
                    if len(corrected_pos) == len(price_subset):
                        ax6.scatter(price_subset[::50], corrected_pos[::50], 
                                   alpha=0.6, color=colors[i], s=20,
                                   label=f'{strategy.replace("_", " ").title()}')
    
    ax6.set_title('Position vs Price Correlation', fontweight='bold')
    ax6.set_xlabel('BTC Price (USD)')
    ax6.set_ylabel('BTC Position')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corrected_portfolio_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_corrected_performance_summary(results, price_data=None):
    """Print corrected performance summary"""
    print("\n" + "="*80)
    print("Corrected Trading Strategy Performance Summary")
    print("="*80)
    
    # Performance table
    performance_data = []
    for strategy, data in results.items():
        metrics = calculate_performance_metrics(data['nav'])
        if metrics:
            performance_data.append([
                strategy.replace('_', ' ').title(),
                f"{metrics['total_return']:.4f}",
                f"{metrics['max_drawdown']:.4f}",
                f"{metrics['sharpe_ratio']:.4f}",
                f"{metrics['volatility']:.4f}",
                f"{metrics['final_nav']:.2f}"
            ])
    
    if performance_data:
        headers = ['Strategy', 'Total Return(%)', 'Max Drawdown(%)', 'Sharpe Ratio', 'Volatility(%)', 'Final NAV']
        col_widths = [max(len(str(row[i])) for row in [headers] + performance_data) + 2 for i in range(len(headers))]
        
        # Print header
        header_row = ''.join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
        print(header_row)
        
        # Print data
        for row in performance_data:
            data_row = ''.join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
            print(data_row)
    
    print("\n" + "="*80)
    print("Corrected Returns Analysis")
    print("="*80)
    
    for strategy, data in results.items():
        nav = data['nav']
        btc_pos = data['btc_positions']
        
        if len(nav) > 0:
            print(f"\n{strategy.replace('_', ' ').title()}:")
            print(f"  Initial NAV: {nav[0]:.2f}")
            print(f"  Final NAV: {nav[-1]:.2f}")
            print(f"  NAV Change: {nav[-1] - nav[0]:.2f}")
            print(f"  Return Rate: {(nav[-1] - nav[0]) / nav[0] * 100:.4f}%")
            
            if btc_pos is not None:
                corrected_pos = correct_btc_positions(btc_pos, nav, price_data)
                print(f"  Original Avg BTC Position: {np.mean(btc_pos):.6f}")
                print(f"  Corrected Avg BTC Position: {np.mean(corrected_pos):.6f}")
                print(f"  Original Max BTC Position: {np.max(btc_pos):.6f}")
                print(f"  Corrected Max BTC Position: {np.max(corrected_pos):.6f}")
                print(f"  Position Std Dev: {np.std(corrected_pos):.6f}")

def main():
    """Main function"""
    print("Starting corrected portfolio NAV and position analysis...")
    
    # Load data
    results = load_trading_results()
    price_data = load_price_data()
    
    if not results:
        print("No trading results data found")
        return
    
    # Print corrected performance summary
    print_corrected_performance_summary(results, price_data)
    
    # Plot corrected analysis charts
    try:
        plot_corrected_analysis(results, price_data)
        print("\nCorrected charts saved as corrected_portfolio_analysis.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nCorrected analysis completed!")

if __name__ == "__main__":
    main()