# Updated evaluation.py acc to gemini

import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import argparse
import time

def load_pkl_file(file_path):
    """Load the pkl file and return its contents."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def ensure_tensor(data):
    """recursively converts numpy arrays in a dict/list to torch tensors."""
    if isinstance(data, dict):
        return {k: ensure_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_tensor(v) for v in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return data

# ==============================================================================
# 1. TEMPORAL METRICS (Vectorized)
# ==============================================================================
def compute_temporal_metrics(trajectory):
    """
    Computes RMSE and Max Error over time for the entire trajectory at once.
    """
    # Extract & Move to CPU
    # Shapes: [Time, Nodes, 3] or [Time, Nodes, 1]
    # We flatten Nodes to calculate global statistics per time step

    # 1. Identify Mask (Metal Nodes Only)
    # Assumes node_type is [Time, Nodes, 1] or [1, Nodes, 1] tiled
    node_type = trajectory['node_type'].to('cpu')
    if node_type.dim() == 3:
        mask = (node_type[0, :, 0] == 0) # Use first frame mask for all steps (topology constant)
    else:
        mask = (node_type[:, 0] == 0)

    gt_pos = trajectory['gt_pos'].to('cpu')[:, mask, :]   # [Time, Metal_Nodes, 3]
    pred_pos = trajectory['pred_pos'].to('cpu')[:, mask, :]
    mesh_pos = trajectory['mesh_pos'].to('cpu')[:, mask, :]

    # 2. Calculate Deformation (Y-axis displacement from original mesh pos)
    # We focus on Y-axis (height/depth) as it's the primary deformation axis in stamping
    gt_deform_y = torch.abs(gt_pos[..., 1] - mesh_pos[..., 1])
    pred_deform_y = torch.abs(pred_pos[..., 1] - mesh_pos[..., 1])

    # 3. Calculate Error (RMSE between Prediction and GT)
    # Error tensor: [Time, Metal_Nodes]
    diff_y = gt_deform_y - pred_deform_y
    squared_error = diff_y ** 2

    # Aggregation over Nodes (dim=1) -> Result is [Time]
    rmse_per_step = torch.sqrt(torch.mean(squared_error, dim=1))
    max_error_per_step = torch.max(torch.abs(diff_y), dim=1)[0]

    # Stats for Plotting (Mean deformation magnitude)
    gt_mean_deform = torch.mean(gt_deform_y, dim=1)
    pred_mean_deform = torch.mean(pred_deform_y, dim=1)

    metrics = {
        'loss_rmse': rmse_per_step.numpy(),
        'loss_max': max_error_per_step.numpy(),
        'gt_mean': gt_mean_deform.numpy(),
        'pred_mean': pred_mean_deform.numpy()
    }

    return metrics

def plot_temporal_error(metrics, output_dir):
    """Generates the GIF showing error evolution."""
    steps = np.arange(len(metrics['loss_rmse']))

    keys_to_plot = [
        ('RMSE', metrics['loss_rmse'], metrics['gt_mean'], metrics['pred_mean']),
        ('Max_Error', metrics['loss_max'], None, None) # calculating max of max is noisy, simplifying
    ]

    os.makedirs(os.path.join(output_dir, "temporal_plots"), exist_ok=True)

    # Pre-calculate Y-limits for stability
    y_max = max(metrics['loss_rmse'].max(), metrics['gt_mean'].max()) * 1.1

    frames = []
    # Skip frames for faster GIF generation (every 2nd frame)
    plot_indices = range(0, len(steps), 2)

    print("Generating temporal error GIF...")
    for step in plot_indices:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot full history up to current step
        ax.plot(steps[:step], metrics['gt_mean'][:step], 'g-', label='GT Mean Deform')
        ax.plot(steps[:step], metrics['pred_mean'][:step], 'b--', label='Pred Mean Deform')
        ax.plot(steps[:step], metrics['loss_rmse'][:step], 'r-', linewidth=2, label='Model RMSE')

        ax.set_xlim(0, len(steps))
        ax.set_ylim(0, y_max)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Deformation (mm)')
        ax.set_title(f'Model Error over Time (Frame {step})')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Save Frame
        filename = os.path.join(output_dir, "temporal_plots", f"frame_{step:03d}.png")
        plt.savefig(filename)
        plt.close()
        frames.append(Image.open(filename))

    # Save GIF
    if frames:
        gif_path = os.path.join(output_dir, "temporal_error.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=60, loop=0)
        print(f"Saved: {gif_path}")

# ==============================================================================
# 2. SPATIAL METRICS (Domain-Wise)
# ==============================================================================
def compute_spatial_metrics(trajectory, axis=0, discrete_size=20):
    """
    Slices the mesh along an axis (default X=0) and computes error per slice.
    Diagnoses if error is concentrated in specific regions (e.g. center vs edges).
    """
    mesh_pos = trajectory['mesh_pos'].to('cpu') # [Time, Nodes, 3]
    gt_pos = trajectory['gt_pos'].to('cpu')
    pred_pos = trajectory['pred_pos'].to('cpu')
    node_type = trajectory['node_type'].to('cpu')

    # Use the LAST frame for spatial analysis (cumulative error)
    # or flatten all frames. Using Last Frame is standard for "final shape quality".
    last_idx = -1

    m_pos = mesh_pos[last_idx] # [Nodes, 3]
    g_pos = gt_pos[last_idx]
    p_pos = pred_pos[last_idx]
    n_type = node_type[last_idx]

    # Filter Metal Only
    mask = (n_type.flatten() == 0)
    m_pos = m_pos[mask]
    g_pos = g_pos[mask]
    p_pos = p_pos[mask]

    # Define Bins
    x_coords = m_pos[:, axis]
    x_min, x_max = x_coords.min().item(), x_coords.max().item()

    bins = []
    curr = x_min
    while curr < x_max:
        bins.append((curr, min(curr + discrete_size, x_max)))
        curr += discrete_size

    results = {
        'ranges': [],
        'rmse': [],
        'gt_mean': [],
        'pred_mean': []
    }

    for (b_min, b_max) in bins:
        # Mask for this spatial bin
        bin_mask = (x_coords >= b_min) & (x_coords < b_max)

        if bin_mask.sum() == 0:
            continue # Empty bin (no nodes here)

        # Slice Data
        p_slice = p_pos[bin_mask]
        g_slice = g_pos[bin_mask]

        # Compute RMSE for this slice (Vector magnitude of error vector)
        error_vec = p_slice - g_slice
        mse = torch.mean(error_vec**2)
        rmse = torch.sqrt(mse).item()

        # Store
        results['ranges'].append(f"{b_min:.0f}-{b_max:.0f}")
        results['rmse'].append(rmse)
        results['gt_mean'].append(torch.mean(g_slice[:, 1]).item()) # Monitor Y-height
        results['pred_mean'].append(torch.mean(p_slice[:, 1]).item())

    return results

def plot_spatial_error(results, output_dir):
    """Bar chart of error across the X-axis of the part."""
    labels = results['ranges']
    rmse = results['rmse']

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap coloring
    norm = plt.Normalize(min(rmse), max(rmse))
    colors = plt.cm.coolwarm(norm(rmse))

    bars = ax.bar(x, rmse, width, color=colors, edgecolor='black')

    ax.set_ylabel('RMSE (mm)')
    ax.set_xlabel('Position along Part Length (X-axis)')
    ax.set_title('Error Distribution across the Part (Spatial Analysis)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Error Intensity')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "spatial_error_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================
def evaluate_rollout(pkl_path, output_dir):
    print(f"Loading: {pkl_path}")
    if not os.path.exists(pkl_path):
        print("File not found.")
        return

    data = load_pkl_file(pkl_path)
    # Ensure list format even if single trajectory
    if isinstance(data, dict): data = [data]

    data = ensure_tensor(data)

    os.makedirs(output_dir, exist_ok=True)
    summary_metrics = []

    for i, traj in enumerate(data):
        print(f"\n--- Processing Trajectory {i} ---")
        traj_dir = os.path.join(output_dir, str(i))
        os.makedirs(traj_dir, exist_ok=True)

        # 1. Temporal Analysis
        t_metrics = compute_temporal_metrics(traj)
        plot_temporal_error(t_metrics, traj_dir)

        # 2. Spatial Analysis
        s_metrics = compute_spatial_metrics(traj)
        plot_spatial_error(s_metrics, traj_dir)

        # 3. Logging
        avg_rmse = np.mean(t_metrics['loss_rmse'])
        max_rmse = np.max(t_metrics['loss_rmse'])
        print(f"  -> Avg RMSE: {avg_rmse:.4f}")
        print(f"  -> Max RMSE: {max_rmse:.4f}")

        summary_metrics.append({
            'id': i,
            'avg_rmse': float(avg_rmse),
            'max_rmse': float(max_rmse)
        })

    # Save Summary JSON
    with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    print("\n✅ Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pkl file")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    evaluate_rollout(args.input, args.output)