import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except Exception:
    sns = None
from scipy import stats
import matplotlib as mpl
from matplotlib.font_manager import FontProperties


def _setup_fonts():
    """Robust font setup that works on Linux/Windows without hard-coded file paths."""
    # Prefer common Chinese fonts if available by name; otherwise fall back gracefully.
    preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = preferred
    plt.rcParams['axes.unicode_minus'] = False
    # Do not bind to a specific font file path to avoid FileNotFoundError
    return FontProperties(size=10)


def plot_equity_and_positions(save_path="trained_agents", secondary_axis=False):
    """
    Generate equity curve and position change plot, and save as PNG.

    Parameters:
        save_path: Prefix saved during evaluation (e.g., "trained_agents"), function will read
                   f"{save_path}_net_assets.npy" and f"{save_path}_btc_positions.npy"
        secondary_axis: If True, plot on same subplot with dual axes;
                        If False, use two subplots (top and bottom)
    """
    # Setup fonts (cross-platform robust)
    font = _setup_fonts()

    # Load data
    net_assets_path = f"{save_path}_net_assets.npy"
    btc_positions_path = f"{save_path}_btc_positions.npy"

    try:
        net_assets = np.load(net_assets_path)
    except Exception as e:
        print(f"[WARN] Failed to read net assets file: {net_assets_path}, error: {e}")
        return

    try:
        btc_positions = np.load(btc_positions_path)
    except Exception as e:
        print(f"[WARN] Failed to read BTC positions file: {btc_positions_path}, error: {e}")
        return

    # Create time series index for plotting
    # 考虑step_gap=16，每个交易步骤代表16个15分钟间隔
    step_gap = 16  # 从task2_eval.py中获取的step_gap值
    timestamps = pd.date_range(start='2024-07-01', periods=len(net_assets), freq=f'{15*step_gap}min')

    plt.style.use('seaborn-v0_8-whitegrid')

    if secondary_axis:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        color1 = '#1f77b4'
        color2 = '#d62728'
        ln1 = ax1.plot(timestamps, net_assets, color=color1, label='Equity', linewidth=1.5)
        ax1.set_xlabel('Time', fontproperties=font)
        ax1.set_ylabel('Net Asset Value', color=color1, fontproperties=font)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ln2 = ax2.plot(timestamps, btc_positions, color=color2, label='BTC Holdings', linewidth=1.2, alpha=0.8)
        ax2.set_ylabel('BTC Holdings', color=color2, fontproperties=font)
        ax2.tick_params(axis='y', labelcolor=color2)

        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', prop=font)

        plt.title('Equity and Position', fontproperties=font)
        plt.tight_layout()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(timestamps, net_assets, color='#1f77b4', linewidth=1.5)
        ax1.set_title('Equity Curve', fontproperties=font)
        ax1.set_ylabel('Net Asset Value', fontproperties=font)
        ax1.grid(True, alpha=0.3)

        ax2.plot(timestamps, btc_positions, color='#d62728', linewidth=1.2)
        ax2.set_title('Position Changes (BTC Holdings)', fontproperties=font)
        ax2.set_xlabel('Time', fontproperties=font)
        ax2.set_ylabel('BTC Holdings', fontproperties=font)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    out_path = f"{save_path}_equity_positions.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved equity and positions figure: {out_path}")


def analyze_trading_results(save_path="trained_agents"):
    """
    Analyze trading evaluation results
    
    Parameters:
        save_path: Path prefix for saving results, same as used in task2_eval.py
    """
    # Setup Chinese font support (cross-platform)
    font = _setup_fonts()
    
    # Load saved data
    try:
        positions = np.load(f"{save_path}_positions.npy", allow_pickle=True)
        net_assets = np.load(f"{save_path}_net_assets.npy")
        btc_positions = np.load(f"{save_path}_btc_positions.npy")
        correct_predictions = np.load(f"{save_path}_correct_predictions.npy")
        print(f"Data loaded successfully:")
        print(f"  - net_assets shape: {net_assets.shape}")
        print(f"  - positions shape: {positions.shape}")
        print(f"  - btc_positions shape: {btc_positions.shape}")
        print(f"  - correct_predictions shape: {correct_predictions.shape}")
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None
    
    # Create time series index based on net_assets length
    # 考虑step_gap=16，每个交易步骤代表16个15分钟间隔
    time_steps = len(net_assets)
    step_gap = 16  # 从task2_eval.py中获取的step_gap值
    time_index = pd.date_range(start='2024-07-01', periods=time_steps, freq=f'{15*step_gap}min')
    
    # Handle different array lengths
    positions_flat = positions.flatten() if positions.ndim > 1 else positions
    
    # Pad or truncate arrays to match net_assets length
    if len(positions_flat) < time_steps:
        positions_flat = np.pad(positions_flat, (0, time_steps - len(positions_flat)), 'constant', constant_values=0)
    elif len(positions_flat) > time_steps:
        positions_flat = positions_flat[:time_steps]
    
    if len(correct_predictions) < time_steps - 1:
        correct_predictions = np.pad(correct_predictions, (0, time_steps - 1 - len(correct_predictions)), 'constant', constant_values=0)
    elif len(correct_predictions) > time_steps - 1:
        correct_predictions = correct_predictions[:time_steps - 1]
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame({
        'timestamp': time_index,
        'net_assets': net_assets,
        'btc_positions': btc_positions,
        'positions': positions_flat,
        'correct_predictions': np.concatenate([[np.nan], correct_predictions])  # Add NaN for first step
    })
    
    # Use actual timestamps for plotting
    timestamps = time_index
    
    # Calculate returns
    returns = np.diff(net_assets) / net_assets[:-1]
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')  # Use modern style
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Net asset value changes
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(timestamps, net_assets, 'b-', linewidth=1.5)
    ax1.set_title('Net Asset Value', fontproperties=font)
    ax1.set_xlabel('Time', fontproperties=font)
    ax1.set_ylabel('Net Asset Value', fontproperties=font)
    ax1.ticklabel_format(style='plain', axis='y')  # Avoid scientific notation
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative returns
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(timestamps[1:], cumulative_returns * 100, 'g-', linewidth=1.5)
    ax2.set_title('Cumulative Return (%)', fontproperties=font)
    ax2.set_xlabel('Time', fontproperties=font)
    ax2.set_ylabel('Cumulative Return (%)', fontproperties=font)
    ax2.grid(True, alpha=0.3)
    
    # 3. Position changes
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(timestamps, btc_positions, 'r-', linewidth=1.5)
    ax3.set_title('BTC Position (Holdings)', fontproperties=font)
    ax3.set_xlabel('Time', fontproperties=font)
    ax3.set_ylabel('BTC Holdings', fontproperties=font)
    ax3.grid(True, alpha=0.3)
    
    # 4. Prediction accuracy analysis
    win_count = np.sum(correct_predictions == 1)
    loss_count = np.sum(correct_predictions == -1)
    neutral_count = np.sum(correct_predictions == 0)
    total_trades = len(correct_predictions) - neutral_count
    
    if total_trades > 0:
        win_rate = win_count / total_trades * 100
    else:
        win_rate = 0
    
    labels = ['Correct', 'Incorrect', 'No Trade']
    sizes = [win_count, loss_count, neutral_count]
    colors = ['#66b3ff', '#ff9999', '#99ff99']
    explode = (0.1, 0, 0)
    
    ax4 = fig.add_subplot(3, 2, 4)
    wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=None, colors=colors, 
                                      autopct='%1.1f%%', shadow=True, startangle=90)
    # Set legend separately to avoid Chinese display issues
    ax4.legend(wedges, labels, loc="center right", bbox_to_anchor=(1.1, 0.5), fontsize=9, prop=font)
    ax4.axis('equal')
    ax4.set_title(f'Prediction Result Distribution (Win rate: {win_rate:.2f}%)', fontproperties=font)
    
    # 5. Return distribution
    ax5 = fig.add_subplot(3, 2, 5)
    if sns is not None:
        sns.histplot(returns, kde=True, ax=ax5, color='blue', alpha=0.6)
    else:
        ax5.hist(returns, bins=50, color='blue', alpha=0.6)
    ax5.set_title('Return Distribution', fontproperties=font)
    ax5.set_xlabel('Return', fontproperties=font)
    ax5.set_ylabel('Frequency', fontproperties=font)
    
    # Add return statistics
    mean_return = np.mean(returns) * 100
    std_return = np.std(returns) * 100
    skew = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    stats_text = (f"Mean Return: {mean_return:.4f}%\n"
                 f"Std Dev of Return: {std_return:.4f}%\n"
                 f"Skewness: {skew:.4f}\n"
                 f"Kurtosis: {kurtosis:.4f}")
    
    ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontproperties=font)
    
    # 6. Trading behavior analysis
    # Fix trading behavior statistics method
    # Should not use position_changes, but use action_ints or correct_predictions for statistics
    
    # Method 1: Infer trading behavior from correct_predictions
    # Both correct and incorrect predictions indicate trades occurred
    buys = 0
    sells = 0
    holds = 0
    
    # Iterate through correct_predictions to count different types of trades
    for i in range(len(correct_predictions)):
        if correct_predictions[i] == 1 or correct_predictions[i] == -1:
            # Determine if it's buy or sell
            # We need to judge based on position changes
            if i > 0 and positions_flat[i] > positions_flat[i-1]:
                buys += 1
            elif i > 0 and positions_flat[i] < positions_flat[i-1]:
                sells += 1
            else:
                # Attempted to trade but unsuccessful (possibly insufficient funds or other reasons)
                holds += 1
        else:
            # correct_predictions == 0 means no trade
            holds += 1
    
    # Check if data is valid to avoid pie chart drawing errors
    if buys + sells + holds > 0:  # Ensure there's data to plot
        labels = ['Buy', 'Sell', 'Hold']
        sizes = [buys, sells, holds]
        # Filter out categories with 0 count
        filtered_labels = []
        filtered_sizes = []
        filtered_colors = []
        colors = ['#66b3ff', '#ff9999', '#99ff99']
        
        for i, size in enumerate(sizes):
            if size > 0:
                filtered_labels.append(labels[i])
                filtered_sizes.append(size)
                filtered_colors.append(colors[i])
        
        ax6 = fig.add_subplot(3, 2, 6)
        if filtered_sizes:  # Ensure there's data after filtering
            wedges, texts, autotexts = ax6.pie(filtered_sizes, labels=None, colors=filtered_colors, 
                                             autopct='%1.1f%%', shadow=True, startangle=90)
            # Set legend separately to avoid Chinese display issues
            ax6.legend(wedges, filtered_labels, loc="center right", bbox_to_anchor=(1.1, 0.5), fontsize=9, prop=font)
            ax6.axis('equal')
            ax6.set_title('Trade Action Distribution', fontproperties=font)
        else:
            ax6.text(0.5, 0.5, 'Insufficient trade data', 
                    horizontalalignment='center', verticalalignment='center',
                    fontproperties=font)
            ax6.set_title('Trade Action Distribution (No Data)', fontproperties=font)
    else:
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.text(0.5, 0.5, 'No trade data', 
                horizontalalignment='center', verticalalignment='center',
                fontproperties=font)
        ax6.set_title('Trade Action Distribution (No Data)', fontproperties=font)
    
    # Add more statistical information to chart title
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns) * 100
    
    # Modify Sharpe ratio calculation method, import same function as task2_eval.py
    try:
        # Try to import same metrics module
        from metrics import sharpe_ratio as metrics_sharpe_ratio
        sharpe = metrics_sharpe_ratio(returns)
    except:
        # If import fails, use original calculation method
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assume 252 trading days/year
        print("Warning: Could not import sharpe_ratio from metrics, using fallback calculation")
    
    plt.suptitle(f'Trading Strategy Evaluation\nMax Drawdown: {max_drawdown:.2f}%, Sharpe Ratio: {sharpe:.4f}',
                 fontsize=16, fontproperties=font)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_path}_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n===== Trading Strategy Evaluation Results =====")
    print(f"Total trades: {total_trades}")
    print(f"Correct predictions: {win_count}")
    print(f"Incorrect predictions: {loss_count}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Mean return: {mean_return:.4f}%")
    print(f"Std dev of return: {std_return:.4f}%")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print(f"Buy count: {buys}")
    print(f"Sell count: {sells}")
    print(f"Hold count: {holds}")
    print(f"Initial equity: {net_assets[0]:.2f}")
    print(f"Final equity: {net_assets[-1]:.2f}")
    print(f"Total return: {(net_assets[-1]/net_assets[0] - 1) * 100:.2f}%")
    
    # Save statistics to CSV file
    stats_df = pd.DataFrame({
        'Metric': ['Total trades', 'Correct predictions', 'Incorrect predictions', 'Win rate (%)', 
                   'Mean return (%)', 'Std dev of return (%)', 'Sharpe ratio', 'Max drawdown (%)',
                   'Skewness', 'Kurtosis', 'Buy count', 'Sell count', 'Hold count',
                   'Initial equity', 'Final equity', 'Total return (%)'],
        'Value': [total_trades, win_count, loss_count, win_rate,
                  mean_return, std_return, sharpe, max_drawdown,
                  skew, kurtosis, buys, sells, holds,
                  net_assets[0], net_assets[-1], (net_assets[-1]/net_assets[0] - 1) * 100]
    })
    
    stats_df.to_csv(f"{save_path}_stats.csv", index=False)
    print(f"\nSaved analysis stats to {save_path}_stats.csv")
    print(f"Saved analysis figure to {save_path}_analysis.png")
    
    return stats_df


if __name__ == "__main__":
    save_path = "trained_agents"
    # 1) Equity & Positions figure (English titles/labels)
    plot_equity_and_positions(save_path=save_path, secondary_axis=False)
    print(f"Saved equity and positions figure: {save_path}_equity_positions.png")

    # 2) Full analysis figure and stats (English titles/labels)
    analyze_trading_results(save_path)
    print(f"Saved analysis figure to {save_path}_analysis.png")
    print(f"Saved analysis stats to {save_path}_stats.csv")