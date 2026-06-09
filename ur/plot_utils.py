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
    Visualize random LiDAR-to-image correspondences predicted by CMRNext (RAFT).
 
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
        n_samples:        How many random correspondences to draw.
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
 
    Example usage inside evaluate_calibration(), after the model forward pass:
 
        up_flow_hw2 = predicted_flow[-1][0].permute(1, 2, 0)  # (H, W, 2)
        fig = visualize_correspondences(
            rgb=sample['rgb'][idx],
            uv=uv,
            up_flow=up_flow_hw2,
            flow_mask=flow_mask,
            n_samples=80,
            uncertainty=predicted_uncertainty[-1][0].sum(0) if _config['uncertainty'] else None,
            normalize_images=_config['normalize_images'],
        )
        plt.show()
        # or: fig.savefig(f"correspondences_{batch_idx}.png", dpi=150, bbox_inches="tight")
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
    # 3. Select valid LiDAR points and sample randomly                    #
    # ------------------------------------------------------------------ #
    # Only keep points that (a) are within image bounds and (b) have a
    # valid entry in the flow mask.
    valid = (
        (uv_np[:, 0] >= 0) & (uv_np[:, 0] < W) &
        (uv_np[:, 1] >= 0) & (uv_np[:, 1] < H) &
        mask_np[uv_np[:, 1].astype(int), uv_np[:, 0].astype(int)]
    )
    valid_uv = uv_np[valid]  # (M, 2)
 
    if valid_uv.shape[0] == 0:
        raise ValueError("No valid LiDAR points found in flow_mask.")
 
    n_draw = min(n_samples, valid_uv.shape[0])
    chosen = np.array(random.sample(range(valid_uv.shape[0]), n_draw))
    src_pts = valid_uv[chosen]  # (n_draw, 2) — source: x, y
 
    # ------------------------------------------------------------------ #
    # 4. Look up the predicted flow for each sampled point                #
    # ------------------------------------------------------------------ #
    src_x = src_pts[:, 0].astype(int)
    src_y = src_pts[:, 1].astype(int)
    dx = flow_np[src_y, src_x, 0]  # (n_draw,)
    dy = flow_np[src_y, src_x, 1]
 
    dst_pts = src_pts + np.stack([dx, dy], axis=1)  # (n_draw, 2) — destination
 
    magnitudes = np.sqrt(dx ** 2 + dy ** 2)  # (n_draw,) — for color mapping
 
    # ------------------------------------------------------------------ #
    # 5. Optionally derive per-point alpha from uncertainty               #
    # ------------------------------------------------------------------ #
    alphas = np.ones(n_draw)
    if unc_np is not None:
        conf = unc_np[src_y, src_x]
        # Low uncertainty → high confidence → high alpha
        conf_min, conf_max = conf.min(), conf.max()
        if conf_max > conf_min:
            alphas = 1.0 - (conf - conf_min) / (conf_max - conf_min)
        alphas = np.clip(0.3 + 0.7 * alphas, 0.3, 1.0)
 
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
 
    # Color map: map displacement magnitude → color
    cmap = matplotlib.colormaps.get_cmap('plasma')
    mag_min = magnitudes.min()
    mag_max = max(magnitudes.max(), 1e-6)
    norm_mag = (magnitudes - mag_min) / (mag_max - mag_min)
    colors = cmap(norm_mag)  # (n_draw, 4) RGBA
 
    # ------------------------------------------------------------------ #
    # 7. Draw correspondences as lines + endpoint markers                 #
    # ------------------------------------------------------------------ #
    # Build a LineCollection for efficiency
    segments = [
        [[sx, sy], [dx_, dy_]]
        for (sx, sy), (dx_, dy_) in zip(src_pts, dst_pts)
    ]
    lc = LineCollection(
        segments,
        colors=[(*c[:3], a * 0.75) for c, a in zip(colors, alphas)],
        linewidths=1.2,
        zorder=2,
    )
    ax.add_collection(lc)
 
    # Source points (LiDAR projection under initial calib) — hollow circles
    ax.scatter(
        src_pts[:, 0], src_pts[:, 1],
        c=colors,
        s=18,
        marker='o',
        linewidths=0.8,
        edgecolors='white',
        alpha=alphas,
        zorder=3,
        label='LiDAR source',
    )
 
    # Destination points (after applying predicted flow) — filled diamonds
    ax.scatter(
        dst_pts[:, 0], dst_pts[:, 1],
        c=colors,
        s=14,
        marker='D',
        linewidths=0,
        alpha=alphas,
        zorder=4,
        label='Predicted target',
    )
 
    # ------------------------------------------------------------------ #
    # 8. Colorbar and legend                                               #
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
                               linewidth=0.8, label=f'Source ({n_draw} pts)')
    dst_patch = mpatches.Patch(facecolor='white', edgecolor='none',
                               label='Predicted target')
    line_patch = mpatches.Patch(facecolor='none', edgecolor='grey',
                                linewidth=1.0, label='Flow vector')
    ax.legend(
        handles=[src_patch, dst_patch, line_patch],
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
