import numpy as np
import trimesh
from scipy.spatial import cKDTree

import os
import argparse
from PIL import Image

POINTS_TO_SAMPLE = None
P2M_SCALING_FACTOR = 0.22

#########################################################
def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def inverse_transform(train_data, param):
    # Unpack camera parameters
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])
    camY = param[3] * np.sin(phi)
    temp = param[3] * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    # Compute camera matrix
    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)
    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])

    # Extract transformed positions and normals
    pt_trans = train_data[:, :3]

    # Inverse transformation for positions
    position = np.dot(pt_trans, cam_mat) + cam_pos

    # Inverse transformation for normals
    return position
#########################################################

def normalize_points(points):
    # Normalize points into the cube [-1, 1]^3
    min_val = np.min(points, axis=0)
    max_val = np.max(points, axis=0)
    points = 2 * (points - min_val) / (max_val - min_val) - 1
    return points

def chamfer_distance(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    closest_p2_to_p1, _ = tree1.query(p2)
    closest_p1_to_p2, _ = tree2.query(p1)
    cd = np.mean(closest_p2_to_p1) + np.mean(closest_p1_to_p2)
    return cd / 2

def f_score(p1, p2, threshold=0.2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    recall = np.mean(tree1.query(p2)[0] < threshold)
    precision = np.mean(tree2.query(p1)[0] < threshold)
    if recall + precision == 0:
        return 0
    fs = 2 * precision * recall / (precision + recall)
    return fs

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, help='Path to input .obj directory.')
# parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
args = parser.parse_args()


input_files = [
    os.path.join(args.input_path, file)
    for file in os.listdir(args.input_path) 
    if file.endswith('.obj')
]

results = {}

for obj_filename in input_files:
    png_name, ext = os.path.splitext(os.path.basename(obj_filename))
    path_loc = os.path.splitext(png_name)[0]
    category, id, png_idx = tuple(path_loc.split('.'))
    print("category:", category, "id:", id, "png_idx:", png_idx)
    dat_filename = f'/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/{category}/{id}/rendering/{png_idx}.dat'
    rendering_metadata = f'/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/{category}/{id}/rendering/rendering_metadata.txt'

    render_meta = np.loadtxt(rendering_metadata)
    dat_points = np.load(dat_filename, allow_pickle=True, encoding='bytes')
    param = render_meta[int(png_idx)]

    # Load and sample meshes
    dat_points = inverse_transform(dat_points, param)
    # need to transpose the matrix to get the correct rotation
    rad_rotation_1 = -np.deg2rad(75)
    rot_mat_1 = np.array([[np.cos(rad_rotation_1), -np.sin(rad_rotation_1), 0],
                        [np.sin(rad_rotation_1), np.cos(rad_rotation_1), 0],
                        [0, 0, 1]]).T
    # need to transpose the matrix to get the correct rotation
    rad_rotation_2 = -np.deg2rad(90)
    rot_mat_2 = np.array([[np.cos(rad_rotation_2), 0, -np.sin(rad_rotation_2)],
                        [0, 1, 0],
                        [np.sin(rad_rotation_2), 0, np.cos(rad_rotation_2)]]).T

    dat_points = ((dat_points @ rot_mat_1) @ rot_mat_2) / P2M_SCALING_FACTOR
    # shift the points to the origin
    dat_points -= np.mean(dat_points, axis=0)

    # need to transpose the matrix around z axis to get the correct rotation (FOR OBJ FILE)
    rad_rotation_3 = -np.deg2rad(15)
    rot_mat_3 = np.array([
        [np.cos(rad_rotation_3), -np.sin(rad_rotation_3), 0],
        [np.sin(rad_rotation_3), np.cos(rad_rotation_3), 0],
        [0, 0, 1]
    ]).T
    # need to transpose the matrix around x axis to get even better rotation (FOR OBJ FILE)
    rad_rotation_4 = np.deg2rad(15)
    rot_mat_4 = np.array([
        [1, 0, 0],
        [0, np.cos(rad_rotation_4), -np.sin(rad_rotation_4)],
        [0, np.sin(rad_rotation_4), np.cos(rad_rotation_4)]
    ]).T

    mesh = trimesh.load(obj_filename, process=True)
    mesh_points, _ = trimesh.sample.sample_surface(mesh, min(mesh.vertices.shape[0], dat_points.shape[0]))
    mesh_points = (mesh_points @ rot_mat_3) @ rot_mat_4
    # shift the points to the origin
    mesh_points -= np.mean(mesh_points, axis=0)

    # Normalize points
    mesh_points = normalize_points(mesh_points)
    dat_points = normalize_points(dat_points)

    # Compute metrics
    cd = chamfer_distance(mesh_points, dat_points)
    fs = f_score(mesh_points, dat_points)

    print("Chamfer Distance:", cd)
    print("F-Score:", fs)

    if category not in results:
        results[category] = {"CD": [], "FS": []}
    results[category]["CD"].append((id, png_idx, cd))
    results[category]["FS"].append((id, png_idx, fs))


print("RESULTS:\n", results)
print("AVERAGE CD PER CATEGORY:\n", {category: np.mean([cd for _, _, cd in results[category]["CD"]]) for category in results})
print("AVERAGE FS PER CATEGORY:\n", {category: np.mean([fs for _, _, fs in results[category]["FS"]]) for category in results})


import json
with open("eval_results.json", "w") as file:
    # Convert the dictionary to a JSON string
    json_string = json.dumps(results)
    # Write the JSON string to the file
    file.write(json_string)
