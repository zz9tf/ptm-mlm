#!/usr/bin/env python3
"""
Parse training log and plot loss, accuracy, and perplexity curves.
Three separate plots: train, validation (with test markers).
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

def smooth_data(data, window_size=100):
    """
    Smooth data using moving average.
    
    @param {list} data - List of values to smooth
    @param {int} window_size - Size of the moving average window
    @returns {np.array} Smoothed data
    """
    if len(data) < window_size:
        return np.array(data)
    return uniform_filter1d(np.array(data), size=window_size, mode='nearest')

def parse_log_file(log_path):
    """
    Parse log file to extract training, validation, and test metrics.
    Groups training data by epoch and calculates averages per epoch.
    
    @param {str} log_path - Path to the log file
    @returns {dict} Dictionary containing lists of epoch, train/val/test loss, acc, and perplexity values
    """
    # Training data - store raw data first, then aggregate by epoch
    train_data_by_epoch = {}  # {epoch: {'losses': [], 'accs': [], 'perplexities': []}}
    
    # Validation data
    val_losses = []
    val_accs = []
    val_perplexities = []
    val_epochs = []
    
    # Test data (usually only one point at the end)
    test_loss = None
    test_acc = None
    test_perplexity = None
    test_epoch = None
    
    # Pattern to match training metrics with epoch in dict: {'Epoch': 969, 'Step': 0, 'train_loss': 2.3646, ...}
    train_pattern_with_epoch = re.compile(r"['\"]?Epoch['\"]?\s*:\s*(\d+).*?train_loss['\"]?\s*:\s*([\d.]+).*?train_preplexity['\"]?\s*:\s*([\d.]+).*?train_acc['\"]?\s*:\s*([\d.]+)")
    
    # Pattern to match training metrics without epoch in dict, extract from line start: "Epoch 1/100: ... {'train_loss': ...}"
    train_pattern_no_epoch = re.compile(r"Epoch\s+(\d+)/.*?train_loss['\"]?\s*:\s*([\d.]+).*?train_preplexity['\"]?\s*:\s*([\d.]+).*?train_acc['\"]?\s*:\s*([\d.]+)")
    
    # Pattern to match validation metrics: 📊 Avg Validation - Loss: 2.8704, Acc: 0.1292, PTM Acc: 0.3980, PPL: 17.65
    val_pattern = re.compile(r"📊 Avg Validation - Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+).*?PPL:\s*([\d.]+)")
    
    # Pattern to match epoch from dictionary before validation line: {'Epoch': 1093, 'Step': 1000, 'avg_val_loss': ...}
    val_epoch_dict_pattern = re.compile(r"['\"]?Epoch['\"]?\s*:\s*(\d+).*?avg_val_loss")
    
    # Pattern to match test metrics: 📊 Avg Test - Loss: 2.5764, Acc: 0.2275, PTM Acc: 0.4953, PPL: 13.18
    test_pattern = re.compile(r"📊 Avg Test - Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+).*?PPL:\s*([\d.]+)")
    test_epoch_pattern = re.compile(r"Epoch\s+(\d+)/")
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        prev_line = ""
        for i, line in enumerate(lines):
            # Try to find training metrics with epoch in dict first
            train_match = train_pattern_with_epoch.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                perplexity = float(train_match.group(3))
                acc = float(train_match.group(4))
            else:
                # Try to find training metrics without epoch in dict, extract from line start
                train_match = train_pattern_no_epoch.search(line)
                if train_match:
                    epoch = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    perplexity = float(train_match.group(3))
                    acc = float(train_match.group(4))
                else:
                    train_match = None
            
            if train_match:
                if epoch not in train_data_by_epoch:
                    train_data_by_epoch[epoch] = {'losses': [], 'accs': [], 'perplexities': []}
                
                train_data_by_epoch[epoch]['losses'].append(loss)
                train_data_by_epoch[epoch]['accs'].append(acc)
                train_data_by_epoch[epoch]['perplexities'].append(perplexity)
            
            # Try to find validation metrics
            val_match = val_pattern.search(line)
            if val_match:
                # Try to extract epoch from the previous line (which usually contains the dict with epoch)
                epoch_match = val_epoch_dict_pattern.search(prev_line)
                if epoch_match:
                    val_epoch = int(epoch_match.group(1))
                else:
                    # If not found, use sequential numbering based on validation count
                    val_epoch = len(val_epochs) if len(val_epochs) > 0 else 0
                
                loss = float(val_match.group(1))
                acc = float(val_match.group(2))
                perplexity = float(val_match.group(3))
                
                val_losses.append(loss)
                val_accs.append(acc)
                val_perplexities.append(perplexity)
                val_epochs.append(val_epoch)
            
            # Try to find test metrics (usually only one at the end)
            test_match = test_pattern.search(line)
            if test_match:
                # Try to extract epoch from the line or previous line
                epoch_match = test_epoch_pattern.search(line)
                if not epoch_match:
                    epoch_match = val_epoch_dict_pattern.search(prev_line)
                if epoch_match:
                    test_epoch = int(epoch_match.group(1))
                elif len(val_epochs) > 0:
                    test_epoch = val_epochs[-1]
                else:
                    test_epoch = None
                
                test_loss = float(test_match.group(1))
                test_acc = float(test_match.group(2))
                test_perplexity = float(test_match.group(3))
            
            # Store current line as previous for next iteration
            prev_line = line
    
    # Aggregate training data by epoch (calculate averages)
    train_epochs = sorted(train_data_by_epoch.keys())
    train_losses = []
    train_accs = []
    train_perplexities = []
    
    for epoch in train_epochs:
        data = train_data_by_epoch[epoch]
        train_losses.append(np.mean(data['losses']))
        train_accs.append(np.mean(data['accs']))
        train_perplexities.append(np.mean(data['perplexities']))
    
    return {
        'train_epochs': train_epochs,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_perplexities': train_perplexities,
        'val_epochs': val_epochs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_perplexities': val_perplexities,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_perplexity': test_perplexity,
        'test_epoch': test_epoch
    }

def plot_train_curves(data, output_path='training_curves_train.png', smooth_window=5):
    """
    Plot training curves with loss, accuracy, and perplexity on one figure.
    Uses three y-axes to accommodate different scales.
    
    @param {dict} data - Dictionary containing train epochs, losses, accs, and perplexities
    @param {str} output_path - Path to save the plot
    @param {int} smooth_window - Window size for smoothing (0 to disable smoothing)
    """
    train_epochs = data['train_epochs']
    train_losses = data['train_losses']
    train_accs = data['train_accs']
    train_perplexities = data['train_perplexities']
    
    if not train_epochs:
        print("❌ No training data found!")
        return
    
    # Apply smoothing if requested
    if smooth_window > 0 and len(train_epochs) > smooth_window:
        train_losses_smooth = smooth_data(train_losses, smooth_window)
        train_accs_smooth = smooth_data(train_accs, smooth_window)
        train_perplexities_smooth = smooth_data(train_perplexities, smooth_window)
        print(f"✨ Applied smoothing with window size: {smooth_window}")
    else:
        train_losses_smooth = train_losses
        train_accs_smooth = train_accs
        train_perplexities_smooth = train_perplexities
    
    print(f"📊 Found {len(train_epochs)} training epochs")
    print(f"📉 Train Loss range: {min(train_losses):.4f} - {max(train_losses):.4f}")
    print(f"📈 Train Acc range: {min(train_accs):.4f} - {max(train_accs):.4f}")
    print(f"📊 Train Perplexity range: {min(train_perplexities):.4f} - {max(train_perplexities):.4f}")
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold', y=0.995)
    
    # First y-axis: Loss (left, blue)
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(train_epochs, train_losses_smooth, color=color1, linewidth=2.5, alpha=0.9, 
                     label='Train Loss', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Second y-axis: Accuracy (right, green)
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(train_epochs, train_accs_smooth, color=color2, linewidth=2.5, alpha=0.9,
                     label='Train Acc', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=10)
    
    # Third y-axis: Perplexity (right, shifted, orange)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:orange'
    ax3.set_ylabel('Perplexity', fontsize=12, fontweight='bold', color=color3)
    line3 = ax3.plot(train_epochs, train_perplexities_smooth, color=color3, linewidth=2.5, alpha=0.9,
                     label='Train Perplexity', linestyle=':')
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=10)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add statistics text
    train_stats = (f'Train Loss: {train_losses[-1]:.4f} (min: {min(train_losses):.4f})\n'
                   f'Train Acc: {train_accs[-1]:.4f} (max: {max(train_accs):.4f})\n'
                   f'Train PPL: {train_perplexities[-1]:.4f} (min: {min(train_perplexities):.4f})')
    ax1.text(0.02, 0.98, train_stats, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Train plot saved to: {output_path}")

def plot_val_curves(data, output_path='training_curves_validation.png', smooth_window=3):
    """
    Plot validation curves with loss, accuracy, and perplexity.
    Also mark test metrics as points on the plot.
    
    @param {dict} data - Dictionary containing val/test epochs, losses, accs, and perplexities
    @param {str} output_path - Path to save the plot
    @param {int} smooth_window - Window size for smoothing (0 to disable smoothing)
    """
    val_epochs = data['val_epochs']
    val_losses = data['val_losses']
    val_accs = data['val_accs']
    val_perplexities = data['val_perplexities']
    test_loss = data['test_loss']
    test_acc = data['test_acc']
    test_perplexity = data['test_perplexity']
    test_epoch = data.get('test_epoch', None)
    
    if not val_epochs:
        print("❌ No validation data found!")
        return
    
    # Apply smoothing if requested
    if smooth_window > 0 and len(val_epochs) > smooth_window:
        val_losses_smooth = smooth_data(val_losses, smooth_window)
        val_accs_smooth = smooth_data(val_accs, smooth_window)
        val_perplexities_smooth = smooth_data(val_perplexities, smooth_window)
        print(f"✨ Applied smoothing with window size: {smooth_window}")
    else:
        val_losses_smooth = val_losses
        val_accs_smooth = val_accs
        val_perplexities_smooth = val_perplexities
    
    print(f"📊 Found {len(val_epochs)} validation epochs")
    print(f"📉 Val Loss range: {min(val_losses):.4f} - {max(val_losses):.4f}")
    print(f"📈 Val Acc range: {min(val_accs):.4f} - {max(val_accs):.4f}")
    print(f"📊 Val Perplexity range: {min(val_perplexities):.4f} - {max(val_perplexities):.4f}")
    
    if test_loss is not None:
        print(f"✅ Test metrics found - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, PPL: {test_perplexity:.4f}")
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle('Validation Curves (with Test Markers)', fontsize=16, fontweight='bold', y=0.995)
    
    # First y-axis: Loss (left, red)
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(val_epochs, val_losses_smooth, color=color1, linewidth=3, alpha=0.9,
                     label='Val Loss', marker='o', markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Second y-axis: Accuracy (right, green)
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(val_epochs, val_accs_smooth, color=color2, linewidth=3, alpha=0.9,
                     label='Val Acc', marker='s', markersize=6, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=10)
    
    # Third y-axis: Perplexity (right, shifted, orange)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:orange'
    ax3.set_ylabel('Perplexity', fontsize=12, fontweight='bold', color=color3)
    line3 = ax3.plot(val_epochs, val_perplexities_smooth, color=color3, linewidth=3, alpha=0.9,
                     label='Val Perplexity', marker='^', markersize=6, linestyle=':')
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=10)
    
    # Mark test metrics as points (use test_epoch if available, otherwise last validation epoch)
    test_markers = []
    if test_loss is not None and len(val_epochs) > 0:
        # Use test_epoch if available, otherwise use the last validation epoch
        test_x = test_epoch if test_epoch is not None else val_epochs[-1]
        
        # Test loss marker
        marker1 = ax1.scatter([test_x], [test_loss], color='darkred', s=200, 
                             marker='*', zorder=5, label='Test Loss', edgecolors='black', linewidths=1.5)
        test_markers.append(marker1)
        
        # Test acc marker
        marker2 = ax2.scatter([test_x], [test_acc], color='darkgreen', s=200,
                             marker='*', zorder=5, label='Test Acc', edgecolors='black', linewidths=1.5)
        test_markers.append(marker2)
        
        # Test perplexity marker
        marker3 = ax3.scatter([test_x], [test_perplexity], color='darkorange', s=200,
                             marker='*', zorder=5, label='Test Perplexity', edgecolors='black', linewidths=1.5)
        test_markers.append(marker3)
    
    # Combine legends
    lines = line1 + line2 + line3 + test_markers
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add statistics text
    val_stats = (f'Val Loss: {val_losses[-1]:.4f} (min: {min(val_losses):.4f})\n'
                 f'Val Acc: {val_accs[-1]:.4f} (max: {max(val_accs):.4f})\n'
                 f'Val PPL: {val_perplexities[-1]:.4f} (min: {min(val_perplexities):.4f})')
    if test_loss is not None:
        val_stats += f'\n\nTest Loss: {test_loss:.4f}\nTest Acc: {test_acc:.4f}\nTest PPL: {test_perplexity:.4f}'
    ax1.text(0.02, 0.98, val_stats, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Validation plot saved to: {output_path}")

if __name__ == '__main__':
    log_path = '/home/zz/.task_manager/logs/00015.log'
    train_output_path = '/home/zz/training_curves_train.png'
    val_output_path = '/home/zz/training_curves_validation.png'
    
    print(f"📖 Reading log file: {log_path}")
    data = parse_log_file(log_path)
    
    # Plot training curves (smooth with window size 5 for epoch-based data)
    if data['train_epochs']:
        train_count = len(data['train_epochs'])
        print(f"📊 Plotting {train_count} training epochs...")
        plot_train_curves(data, train_output_path, smooth_window=5)
    else:
        print("❌ No training data found!")
    
    # Plot validation curves with test markers (smooth with window size 3)
    if data['val_epochs']:
        val_count = len(data['val_epochs'])
        print(f"📊 Plotting {val_count} validation epochs...")
        plot_val_curves(data, val_output_path, smooth_window=3)
    else:
        print("❌ No validation data found!")
