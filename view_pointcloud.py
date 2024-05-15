import numpy as np
import trimesh

PHOTO_IDX = 3

DAT_FILE = f"/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/0{PHOTO_IDX}.dat"
OBJ_FILE = f"/om/user/evan_kim/InstantMesh/outputs/instant-mesh-large/meshes/02691156.98b163efbbdf20c898dc7d57268f30d4.0{PHOTO_IDX}.png.obj"
RENDERING_METADATA = "/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/rendering_metadata.txt"
SCALING_FACTOR = 0.20

# new_order = [0, 2, 1]
# new_order = [1, 0, 2]
# new_order = [1, 2, 0]
# new_order = [2, 0, 1]
# new_order = [2, 1, 0]

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

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

render_meta = np.loadtxt(RENDERING_METADATA)
dat_points = np.load(DAT_FILE, allow_pickle=True, encoding='bytes')
param = render_meta[PHOTO_IDX]

dat_points = inverse_transform(dat_points, param)
# need to transpose the matrix to get the correct rotation
rad_rotation_1 = -np.deg2rad(75)
rot_mat_1 = np.array([
    [np.cos(rad_rotation_1), -np.sin(rad_rotation_1), 0],
    [np.sin(rad_rotation_1), np.cos(rad_rotation_1), 0],
    [0, 0, 1]
]).T
# need to transpose the matrix to get the correct rotation
rad_rotation_2 = -np.deg2rad(90)
rot_mat_2 = np.array([
    [np.cos(rad_rotation_2), 0, -np.sin(rad_rotation_2)],
    [0, 1, 0],
    [np.sin(rad_rotation_2), 0, np.cos(rad_rotation_2)]
]).T

dat_points = ((dat_points @ rot_mat_1) @ rot_mat_2) / SCALING_FACTOR
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

mesh = trimesh.load(OBJ_FILE, process=True)
mesh_points, _ = trimesh.sample.sample_surface(mesh, 8179)
mesh_points = (mesh_points @ rot_mat_3) @ rot_mat_4
# shift the points to the origin
mesh_points -= np.mean(mesh_points, axis=0)

##############################################3
#Take transpose as columns should be the points
# p1 = dat_points.transpose()
# p2 = mesh_points.transpose()

# #Calculate centroids
# p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
# p2_c = np.mean(p2, axis = 1).reshape((-1,1))

# #Subtract centroids
# q1 = p1-p1_c
# q2 = p2-p2_c

# #Calculate covariance matrix
# H=np.matmul(q1,q2.transpose())

# #Calculate singular value decomposition (SVD)
# U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

# #Calculate rotation matrix
# R = np.matmul(V_t.transpose(),U.transpose())

# assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

# #Calculate translation matrix
# T = p2_c - np.matmul(R,p1_c)

# #Check result
# result = T + np.matmul(R,p1)
#########################################
# dat_points = result.transpose()
point_cloud_dat = trimesh.points.PointCloud(dat_points, colors=[255, 0, 0])
point_cloud_mesh = trimesh.points.PointCloud(mesh_points, colors=[0, 255, 0])

# Visualize the point clouds
scene = trimesh.Scene()
scene.add_geometry(point_cloud_dat)
scene.add_geometry(point_cloud_mesh)
scene.show()
