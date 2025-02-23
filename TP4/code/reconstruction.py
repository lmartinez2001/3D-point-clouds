#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    grid_shape = scalar_field.shape
    grid_coords = np.indices((grid_resolution,) * 3).transpose(1,2,3,0)
    grid_coords = min_grid + size_voxel * grid_coords
    grid_coords = grid_coords.reshape(-1,3) # (16*16*16, 3)
    tree = KDTree(points) # point cloud in kdtree for nneighbors
    
    _, ind = tree.query(grid_coords, k=1) # (16*16*16, 1)
    selected_normals, selected_points = normals[ind.reshape(-1)], points[ind.reshape(-1)]

    new_field = np.sum(selected_normals * (grid_coords - selected_points), axis=1)
    scalar_field += new_field.reshape(grid_shape)

    return
    

# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
    def theta(x: np.ndarray, h: int) -> np.ndarray:
        return np.exp(-(np.linalg.norm(x, axis=-1) / h)**2)

    grid_shape = scalar_field.shape
    grid_coords = np.indices((grid_resolution,) * 3).transpose(1,2,3,0)
    grid_coords = min_grid + size_voxel * grid_coords
    grid_coords = grid_coords.reshape(-1,3) # (16*16*16, 3)
    tree = KDTree(points) # point cloud in kdtree for nneighbors
    
    _, ind = tree.query(grid_coords, k=knn) # (16*16*16, knn)
    selected_normals, selected_points = normals[ind], points[ind] # (4096,knn,3)
    diff = grid_coords[:,None,:] - selected_points
    all_theta = theta(diff, 0.1) # (4096,30)
    new_field = np.sum(np.sum(selected_normals * diff, axis=-1) * all_theta, axis=-1) / np.sum(all_theta, axis=-1)
    scalar_field += new_field.reshape(grid_shape)
    
    return



if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
    print("size_voxel: ", size_voxel)
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

    print(scalar_field.shape , normals.shape)

	# # Compute the scalar field in the grid
    # compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
    verts += min_grid
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj=f'../bunny_mesh_imls_{grid_resolution}.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	


