import open3d as o3d
import numpy as np
import torch

def interpolate_color(color1, color2, weight):
    return [c1 * (1 - weight) + c2 * weight for c1, c2 in zip(color1, color2)]

def transfer_weight_to_color_heatmap(xyz, heatmap, color0=[0, 0, 0.4], color1 = [1, 0.1, 0]):
    assert np.max(heatmap) <= 1 and np.min(heatmap) >= 0
    colors = np.array([interpolate_color(color0, color1, hm) for hm in heatmap])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def vis_global_heatmap(pcd_path):
    scene_heatmap = o3d.io.read_point_cloud(pcd_path)
    scene_heatmap_pcd = transfer_weight_to_color_heatmap(np.asarray(scene_heatmap.points), np.asarray(scene_heatmap.colors).mean(axis=-1))
    o3d.visualization.draw_geometries([scene_heatmap_pcd])

if __name__ == "__main__":
    pcd_path = "your_pcd_path"