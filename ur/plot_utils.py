import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
 

def visualize_correspondences(
    rgb: torch.Tensor,
    uv: torch.Tensor,
    up_flow: torch.Tensor,
    flow_mask: torch.Tensor,
    n_samples: int = 50,
    uncertainty: torch.Tensor = None,
    normalize_images: bool = True,
    mean_torch: torch.Tensor = None,
    std_torch: torch.Tensor = None,
    title: str = "CMRNext Correspondences",
    figsize: tuple = (16, 9),
    seed: int = None,
) -> plt.Figure:
    """
    Visualize LiDAR-to-image correspondences predicted by CMRNext (RAFT).
    
    All valid LiDAR points are shown as small dots. A random subset of these
    points also displays flow vectors (arrows/lines) to the corrected positions.
    
    For each sampled LiDAR point:
      - A circle marks the projected source position (where the point lands
        under the *initial*, possibly miscalibrated, extrinsic).
      - An arrow (or line) shows the predicted flow displacement to the
        corrected pixel position.
      - Colors encode the displacement magnitude (colormap: plasma).
    
    Args:
        rgb:              Image tensor, shape (3, H, W), float, on any device.
                          May be either raw [0,1] or ImageNet-normalized.
        uv:               LiDAR pixel coordinates, shape (N, 2), int/float.
                          Column-0 = x (width), column-1 = y (height).
        up_flow:          Predicted flow from the network, shape (H, W, 2),
                          float, same device as uv.
        flow_mask:        Binary mask of valid LiDAR pixels, shape (H, W), int.
        n_samples:        How many random correspondences to draw flow lines for.
        uncertainty:      Optional per-pixel uncertainty, shape (H, W).
                          When provided, point alpha is modulated by confidence.
        normalize_images: If True, un-normalize rgb using mean/std before display.
        mean_torch:       ImageNet mean used during pre-processing, shape (3,).
                          Defaults to [0.485, 0.456, 0.406].
        std_torch:        ImageNet std used during pre-processing, shape (3,).
                          Defaults to [0.229, 0.224, 0.225].
        title:            Figure title string.
        figsize:          Matplotlib figure size in inches.
        seed:             Random seed for reproducible sampling. None = random.
 
    Returns:
        fig: The matplotlib Figure object (caller can call plt.show() or
             fig.savefig() on it).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
 
    # ------------------------------------------------------------------ #
    # 1. Prepare the background image                                      #
    # ------------------------------------------------------------------ #
    if mean_torch is None:
        mean_torch = torch.tensor([0.485, 0.456, 0.406])
    if std_torch is None:
        std_torch = torch.tensor([0.229, 0.224, 0.225])
 
    img_np = rgb.detach().cpu().float()  # (3, H, W)
    if normalize_images:
        # Undo ImageNet normalization: x = (x_norm * std) + mean
        mean = mean_torch.view(3, 1, 1).cpu()
        std = std_torch.view(3, 1, 1).cpu()
        img_np = img_np * std + mean
 
    img_np = img_np.permute(1, 2, 0).numpy()  # (H, W, 3)
    img_np = np.clip(img_np, 0.0, 1.0)
 
    H, W = img_np.shape[:2]
 
    # ------------------------------------------------------------------ #
    # 2. Move tensors to CPU / numpy                                       #
    # ------------------------------------------------------------------ #
    uv_np = uv.detach().cpu().numpy().astype(np.float32)          # (N, 2): [x, y]
    flow_np = up_flow.detach().cpu().float().numpy()               # (H, W, 2)
    mask_np = flow_mask.detach().cpu().numpy().astype(bool)        # (H, W)
 
    unc_np = None
    if uncertainty is not None:
        unc_np = uncertainty.detach().cpu().float().numpy()        # (H, W)
 
    # ------------------------------------------------------------------ #
    # 3. Select valid LiDAR points                                        #
    # ------------------------------------------------------------------ #
    # Keep all points that (a) are within image bounds and (b) have a
    # valid entry in the flow mask.
    valid = (
        (uv_np[:, 0] >= 0) & (uv_np[:, 0] < W) &
        (uv_np[:, 1] >= 0) & (uv_np[:, 1] < H) &
        mask_np[uv_np[:, 1].astype(int), uv_np[:, 0].astype(int)]
    )
    valid_uv = uv_np[valid]  # (M, 2)
    M = valid_uv.shape[0]
 
    if M == 0:
        raise ValueError("No valid LiDAR points found in flow_mask.")
 
    # Get flow for ALL valid points (for coloring all source points)
    src_x_all = valid_uv[:, 0].astype(int)
    src_y_all = valid_uv[:, 1].astype(int)
    dx_all = flow_np[src_y_all, src_x_all, 0]
    dy_all = flow_np[src_y_all, src_x_all, 1]
    magnitudes_all = np.sqrt(dx_all ** 2 + dy_all ** 2)
 
    # ------------------------------------------------------------------ #
    # 4. Sample a random subset for flow line visualization               #
    # ------------------------------------------------------------------ #
    n_draw = min(n_samples, M)
    chosen = np.array(random.sample(range(M), n_draw))
    
    src_pts = valid_uv[chosen]  # (n_draw, 2) — source: x, y
    dx = dx_all[chosen]
    dy = dy_all[chosen]
    dst_pts = src_pts + np.stack([dx, dy], axis=1)  # (n_draw, 2) — destination
    magnitudes = magnitudes_all[chosen]
 
    # ------------------------------------------------------------------ #
    # 5. Optionally derive per-point alpha from uncertainty               #
    # ------------------------------------------------------------------ #
    alphas_all = np.ones(M)
    alphas_samples = np.ones(n_draw)
    
    if unc_np is not None:
        conf_all = unc_np[src_y_all, src_x_all]
        conf_samples = conf_all[chosen]
        
        # Low uncertainty → high confidence → high alpha
        conf_min_all, conf_max_all = conf_all.min(), conf_all.max()
        if conf_max_all > conf_min_all:
            alphas_all = 1.0 - (conf_all - conf_min_all) / (conf_max_all - conf_min_all)
            alphas_samples = 1.0 - (conf_samples - conf_min_all) / (conf_max_all - conf_min_all)
        
        alphas_all = np.clip(0.3 + 0.7 * alphas_all, 0.3, 1.0)
        alphas_samples = np.clip(0.3 + 0.7 * alphas_samples, 0.3, 1.0)
 
    # ------------------------------------------------------------------ #
    # 6. Build the figure                                                  #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.imshow(img_np, interpolation='bilinear', aspect='auto')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # image coords: y increases downward
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=10, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
 
    # Color map: map displacement magnitude → color (using all points for scaling)
    cmap = matplotlib.colormaps.get_cmap('plasma')
    mag_min = magnitudes_all.min()
    mag_max = max(magnitudes_all.max(), 1e-6)
    
    # Colors for ALL source points
    norm_mag_all = (magnitudes_all - mag_min) / (mag_max - mag_min)
    colors_all = cmap(norm_mag_all)
    
    # Colors for sampled points (for lines and destination markers)
    norm_mag_samples = (magnitudes - mag_min) / (mag_max - mag_min)
    colors_samples = cmap(norm_mag_samples)
 
    # ------------------------------------------------------------------ #
    # 7. Draw ALL source points (small dots, no flow lines)               #
    # ------------------------------------------------------------------ #
    ax.scatter(
        valid_uv[:, 0], valid_uv[:, 1],
        c=colors_all,
        s=8,  # Smaller size for background points
        marker='o',
        linewidths=0.3,
        edgecolors='white',
        alpha=alphas_all * 0.5,  # More transparent for background points
        zorder=2,
        label=f'LiDAR source ({M} pts)',
    )
 
    # ------------------------------------------------------------------ #
    # 8. Draw flow lines and destination points for sampled subset        #
    # ------------------------------------------------------------------ #
    # Build a LineCollection for the sampled points
    segments = [
        [[sx, sy], [dx_, dy_]]
        for (sx, sy), (dx_, dy_) in zip(src_pts, dst_pts)
    ]
    lc = LineCollection(
        segments,
        colors=[(*c[:3], a * 0.9) for c, a in zip(colors_samples, alphas_samples)],
        linewidths=1.2,
        zorder=3,
    )
    ax.add_collection(lc)
 
    # Source points (highlighted) — slightly larger circles
    ax.scatter(
        src_pts[:, 0], src_pts[:, 1],
        c=colors_samples,
        s=22,
        marker='o',
        linewidths=0.8,
        edgecolors='white',
        alpha=alphas_samples,
        zorder=4,
        label=f'Highlighted source ({n_draw} pts)',
    )
 
    # Destination points (after applying predicted flow) — filled diamonds
    ax.scatter(
        dst_pts[:, 0], dst_pts[:, 1],
        c=colors_samples,
        s=16,
        marker='D',
        linewidths=0,
        alpha=alphas_samples,
        zorder=5,
        label='Predicted target',
    )
 
    # ------------------------------------------------------------------ #
    # 9. Colorbar and legend                                               #
    # ------------------------------------------------------------------ #
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=mag_min, vmax=mag_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=30)
    cbar.set_label('Flow magnitude (px)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
 
    src_patch = mpatches.Patch(facecolor='none', edgecolor='white',
                               linewidth=0.8, label=f'All source points ({M} pts)')
    highlighted_patch = mpatches.Patch(facecolor='none', edgecolor='white',
                                       linewidth=1.5, label=f'Flow vectors ({n_draw} pts)')
    dst_patch = mpatches.Patch(facecolor='white', edgecolor='none',
                               label='Predicted target')
    line_patch = mpatches.Patch(facecolor='none', edgecolor='grey',
                                linewidth=1.0, label='Flow vector')
    ax.legend(
        handles=[src_patch, highlighted_patch, dst_patch, line_patch],
        loc='lower left',
        fontsize=8,
        framealpha=0.5,
        facecolor='#1a1a1a',
        labelcolor='white',
    )
 
    fig.tight_layout(pad=0.5)
    return fig


def plot_dense_depth_map(depth_img_no_occlusion):
    """
    Plots the dense depth map (depth_img_no_occlusion).
    
    Args:
        depth_img_no_occlusion (torch.Tensor or np.ndarray): Dense depth map.
    """
    if isinstance(depth_img_no_occlusion, torch.Tensor):
        depth_map = depth_img_no_occlusion.detach().cpu().numpy()
    else:
        depth_map = depth_img_no_occlusion

    # Remove batch and channel dimensions if present
    depth_map = np.squeeze(depth_map)

    # If it still has 3 dimensions (e.g., C, H, W), take the first channel
    # This handles cases where reflectance data might be concatenated
    if depth_map.ndim == 3:
        depth_map = depth_map[0]

    # Ensure it's a float array so we can use NaN
    depth_map = depth_map.astype(float)
    # Mask zero values (which represent no depth) for better visualization
    depth_map[depth_map == 0] = np.nan

    plt.figure(figsize=(10, 5))
    
    # Create a colormap with black for NaN values to make points visible
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='black')
    
    # depth_img_no_occlusion is often already normalized between 0 and 1,
    # but the jet colormap will automatically scale to the max value present
    plt.imshow(depth_map, cmap=cmap)
    plt.colorbar(label='Normalized Depth')
    plt.title('Dense Depth Map (No Occlusion)')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
