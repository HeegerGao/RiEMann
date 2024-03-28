import open3d as o3d
import numpy as np
import torch

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def draw_arrow(unit_vector, translation, color):
    # Default arrow direction is along Z-axis
    default_arrow_direction = np.array([0, 0, 1])
    R = rotation_matrix_from_vectors(default_arrow_direction, unit_vector)
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00025,
                                            cone_radius=0.0009,
                                            cylinder_height=np.linalg.norm(unit_vector) * 0.007,
                                            cone_height=np.linalg.norm(unit_vector) * 0.004)
    arrow.paint_uniform_color(color)
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(translation)

    return arrow

def draw_ori_feature(ball_pcd, ball_oris):
    arrows = []
    for i in range(ball_oris.shape[0]):
        # Your desired vector
        red_vector = ball_oris[i][0:3].numpy()
        green_vector = ball_oris[i][3:6].numpy()
        blue_vector = ball_oris[i][6:9].numpy()
        unit_red_vector = red_vector / np.linalg.norm(red_vector)
        unit_green_vector = green_vector / np.linalg.norm(green_vector)
        unit_blue_vector = blue_vector / np.linalg.norm(blue_vector)

        arrows.append(draw_arrow(unit_red_vector, np.asarray(ball_pcd.points)[i], (1, 0, 0)))
        arrows.append(draw_arrow(unit_green_vector, np.asarray(ball_pcd.points)[i], (0, 1, 0)))
        arrows.append(draw_arrow(unit_blue_vector, np.asarray(ball_pcd.points)[i], (0, 0, 1)))

    # Visualize the point cloud and the vectors
    o3d.visualization.draw_geometries([ball_pcd, *arrows])


if "__name__" == "__main__":
    ball_pcd = o3d.io.read_point_cloud("eval.pcd")
    ori_feature = torch.load('ori_feature_{pcd_name}.pt', map_location=torch.device('cpu')).detach()
    draw_ori_feature(ball_pcd, ori_feature)