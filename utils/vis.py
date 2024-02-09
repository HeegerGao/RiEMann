import open3d as o3d
import torch
import numpy as np
import colorsys

def interpolate_color(weight, color1=[0, 0, 0.4], color2=[1, 0.1, 0]):
    return [c1 * (1 - weight) + c2 * weight for c1, c2 in zip(color1, color2)]

def save_pcd_as_pcd(xyz, rgb, save_file="./pcd/test.pcd", draw_heatmap=False):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()

    pcd=o3d.geometry.PointCloud()

    if draw_heatmap:
        rgb = np.array([interpolate_color(hm) for hm in rgb.mean(axis=-1)])

    pcd.points=o3d.utility.Vector3dVector(xyz)
    pcd.colors=o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(save_file, pcd)

def vis_result(xyz, rgb, pred_pos, pred_rot, point_size=10.0, color_enhance=True):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()
        pred_pos = pred_pos.detach().cpu().numpy()
        pred_rot = pred_rot.detach().cpu().numpy()

    pred_trans = np.identity(4)
    pred_trans[:3, :3] = pred_rot
    pred_trans[:3, 3] = pred_pos

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    coor_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    coor_pred = coor_pred.transform(pred_trans)

    if color_enhance:
        # enhance color for better visualization
        colors = np.asarray(pcd.colors)
        hsv_colors = np.array([colorsys.rgb_to_hsv(*color) for color in colors])
        hsv_colors[:, 1] *= 1.4
        hsv_colors[:, 1] = np.clip(hsv_colors[:, 1], 0, 1)
        adjusted_colors = np.array([colorsys.hsv_to_rgb(*color) for color in hsv_colors])
        pcd.colors = o3d.utility.Vector3dVector(adjusted_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coor_pred)
    opt = vis.get_render_option()
    opt.point_size = point_size   # change point size

    vis.run()
    vis.destroy_window()
