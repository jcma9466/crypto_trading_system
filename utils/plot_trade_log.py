import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

def _setup_fonts():
    """Setup Chinese font support (cross-platform)"""
    try:
        # Try to use system fonts
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            'C:/Windows/Fonts/msyh.ttc',  # Windows
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = FontProperties(fname=path)
                break
        
        if font is None:
            # Fallback to matplotlib default
            mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            mpl.rcParams['axes.unicode_minus'] = False
            font = FontProperties()
        
        return font
    except Exception:
        return FontProperties()

def plot_trade_analysis(save_path="trained_agents"):
    """
    Plot trading analysis using .npy data files
    
    Parameters:
        save_path: Path prefix for loading data files
    """
    # Setup Chinese font support
    font = _setup_fonts()
    
    # Load data files
    try:
        net_assets = np.load(f"{save_path}_net_assets.npy")
        positions = np.load(f"{save_path}_positions.npy", allow_pickle=True)
        btc_positions = np.load(f"{save_path}_btc_positions.npy")
        correct_predictions = np.load(f"{save_path}_correct_predictions.npy")
        
        print(f"Data loaded successfully:")
        print(f"  - net_assets shape: {net_assets.shape}")
        print(f"  - positions shape: {positions.shape}")
        print(f"  - btc_positions shape: {btc_positions.shape}")
        print(f"  - correct_predictions shape: {correct_predictions.shape}")
        
    except Exception as e:
        print(f"Error loading data files: {e}")
        # Create empty plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        for ax in [ax1, ax2, ax3]:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center', va='center')
        ax1.set_title('Net Assets Over Time')
        ax2.set_title('BTC Holdings Over Time')
        ax3.set_title('Trading Actions Over Time')
        plt.tight_layout()
        plt.savefig(f'{save_path}_trade_analysis.png')
        plt.close()
        return
    
    # Handle different array lengths and create timestamps
    # 考虑step_gap=16，每个交易步骤代表16个15分钟间隔
    time_steps = len(net_assets)
    step_gap = 16  # 从task2_eval.py中获取的step_gap值
    timestamps = pd.date_range(start='2024-07-01', periods=time_steps, freq=f'{15*step_gap}min')
    
    # Flatten positions if needed
    positions_flat = positions.flatten() if positions.ndim > 1 else positions
    
    # Pad or truncate arrays to match net_assets length
    if len(positions_flat) < time_steps:
        positions_flat = np.pad(positions_flat, (0, time_steps - len(positions_flat)), 'constant', constant_values=0)
    elif len(positions_flat) > time_steps:
        positions_flat = positions_flat[:time_steps]
    
    # Create synthetic price data based on net assets and btc positions
    # This is an approximation since we don't have actual price data
    estimated_price = net_assets / (btc_positions + 1e-8)  # Avoid division by zero
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Net Assets Over Time with Trading Points
    ax1.plot(timestamps, net_assets, color='blue', label='Net Assets', linewidth=2)
    
    # Identify buy and sell points based on position changes
    buy_points = []
    sell_points = []
    
    for i in range(1, len(positions_flat)):
        if positions_flat[i] > positions_flat[i-1]:  # Buy
            buy_points.append(i)
        elif positions_flat[i] < positions_flat[i-1]:  # Sell
            sell_points.append(i)
    
    # Plot buy points
    if buy_points:
        buy_timestamps = [timestamps[i] for i in buy_points]
        buy_assets = [net_assets[i] for i in buy_points]
        ax1.scatter(buy_timestamps, buy_assets, color='red', marker='^', label='Buy', s=100, alpha=0.7)
    
    # Plot sell points
    if sell_points:
        sell_timestamps = [timestamps[i] for i in sell_points]
        sell_assets = [net_assets[i] for i in sell_points]
        ax1.scatter(sell_timestamps, sell_assets, color='green', marker='v', label='Sell', s=100, alpha=0.7)
    
    ax1.set_title('Net Assets Over Time (with Trading Points)', fontproperties=font, fontsize=14)
    ax1.set_xlabel('Time', fontproperties=font)
    ax1.set_ylabel('Net Assets', fontproperties=font)
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop=font)
    
    # Plot 2: BTC Holdings Over Time with Trading Points
    ax2.plot(timestamps, btc_positions, color='orange', label='BTC Holdings', linewidth=2)
    
    # Plot buy points
    if buy_points:
        buy_btc = [btc_positions[i] for i in buy_points]
        ax2.scatter(buy_timestamps, buy_btc, color='red', marker='^', label='Buy', s=100, alpha=0.7)
    
    # Plot sell points
    if sell_points:
        sell_btc = [btc_positions[i] for i in sell_points]
        ax2.scatter(sell_timestamps, sell_btc, color='green', marker='v', label='Sell', s=100, alpha=0.7)
    
    ax2.set_title('BTC Holdings Over Time (with Trading Points)', fontproperties=font, fontsize=14)
    ax2.set_xlabel('Time', fontproperties=font)
    ax2.set_ylabel('BTC Amount', fontproperties=font)
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop=font)
    
    # Plot 3: Trading Actions Over Time
    ax3.plot(timestamps, positions_flat, color='purple', label='Trading Actions', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax3.set_title('Trading Actions Time Series', fontproperties=font, fontsize=14)
    ax3.set_xlabel('Time', fontproperties=font)
    ax3.set_ylabel('Position', fontproperties=font)
    ax3.grid(True, alpha=0.3)
    ax3.legend(prop=font)
    
    # Calculate and display statistics
    total_trades = len(buy_points) + len(sell_points)
    final_return = (net_assets[-1] - net_assets[0]) / net_assets[0] * 100
    max_drawdown = np.min(net_assets / np.maximum.accumulate(net_assets) - 1) * 100
    
    # Add statistics text
    stats_text = f"Total Trades: {total_trades}\nTotal Return: {final_return:.2f}%\nMax Drawdown: {max_drawdown:.2f}%"
    fig.text(0.02, 0.02, stats_text, fontproperties=font, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for statistics
    
    # Save the plot
    output_path = f'{save_path}_trade_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trading analysis completed, chart saved to {output_path}")
    print(f"Statistical analysis results:")
    print(f"Total trades: {total_trades}")
    print(f"Buy count: {len(buy_points)}")
    print(f"Sell count: {len(sell_points)}")
    print(f"Total return: {final_return:.2f}%")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    
    return output_path

if __name__ == "__main__":
    save_path = "trained_agents"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Generate trade analysis plot
    plot_trade_analysis(save_path)