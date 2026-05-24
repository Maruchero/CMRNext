import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def plot_bev_camera_correspondences(inlier_pc, rgb_tensor, inlier_uv, max_dist=80.0):
    """
    Plots the Bird's-Eye View of LiDAR points and traces their structural 
    associations directly to the deep learning feature-predicted camera pixels.
    """
    pc = np.array(inlier_pc)
    uv = np.array(inlier_uv)
    
    # Un-normalize camera image back to standard RGB space
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    rgb = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * std + mean, 0.0, 1.0)

    # Initialize side-by-side workspace
    fig, (ax_bev, ax_cam) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Model Feature Association: 3D LiDAR Point to 2D Camera Pixel", fontsize=14, fontweight='bold')

    # --- LEFT PANEL: Bird's-Eye View (LiDAR Point Source) ---
    x_coords = pc[:, 0]
    z_coords = pc[:, 2] # Depth
    distances = np.sqrt(x_coords**2 + z_coords**2)

    scatter_bev = ax_bev.scatter(x_coords, z_coords, c=distances, cmap='jet', 
                                 vmin=0, vmax=max_dist, s=25, zorder=3, edgecolors='black', linewidths=0.3)
    
    ax_bev.set_title("Source: LiDAR Bird's-Eye View (X vs Z)")
    ax_bev.set_xlabel("X (Left / Right) [meters]")
    ax_bev.set_ylabel("Z (Depth Forward) [meters]")
    ax_bev.grid(True, linestyle='--', alpha=0.5)
    ax_bev.set_xlim([-15, 15])
    ax_bev.set_ylim([0, max_dist])
    ax_bev.invert_xaxis() # Invert to align spatial left/right with camera perspective

    # --- RIGHT PANEL: Camera Image View (Model Prediction Target) ---
    ax_cam.imshow(rgb)
    ax_cam.scatter(uv[:, 0], uv[:, 1], c=distances, cmap='jet', 
                   vmin=0, vmax=max_dist, s=25, edgecolors='black', linewidths=0.5, zorder=3)
    ax_cam.set_title("Assignment: Network Feature-Predicted Pixels")
    ax_cam.axis('off')

    # Colorbar reference for depth context
    cbar = fig.colorbar(scatter_bev, ax=ax_cam, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('LiDAR Target Distance [meters]')

    # --- DRAW ASSOCIATION ASSIGNMENTS ---
    # Map a physical connection line between each source 3D point and target 2D image pixel
    for i in range(len(pc)):
        bev_source_pt = (x_coords[i], z_coords[i])
        cam_target_pt = (uv[i, 0], uv[i, 1])

        con = ConnectionPatch(xyA=bev_source_pt, xyB=cam_target_pt, 
                              coordsA="data", coordsB="data",
                              axesA=ax_bev, axesB=ax_cam, 
                              color="red", lw=0.6, alpha=0.7)
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()