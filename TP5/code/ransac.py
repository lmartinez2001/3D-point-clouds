#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#
import os
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    point_plane = np.zeros((1,3))
    normal_plane = np.zeros((1,3))
    v1 = points[0] - points[1]
    v2 = points[2] - points[1]

    normal_plane = np.cross(v1, v2)[None,:] # (1, 3)
    normal_plane /= np.linalg.norm(normal_plane) 

    point_plane = points[0].reshape(1,-1) # (1, 3)
    return point_plane, normal_plane



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1, use_normals=False, normals=None, normals_threshold=0.1):
    if normals is None and use_normals: raise AssertionError("use_normals is set to true but no normals were provided")

    indexes = np.zeros(len(points), dtype=bool)
    distances = np.abs(np.sum((points-pt_plane) * normal_plane, axis=1)) # dot product
    # Filter by distances first
    indexes = distances < threshold_in
    
    # Filter by normals if applicable
    if use_normals:
        dot_products = np.abs(np.sum(normals * normal_plane, axis=1))
        normals_aligned = dot_products > (1 - normals_threshold)
        indexes = indexes & normals_aligned
    return indexes



def RANSAC(points, normals=None, nb_draws=100, threshold_in=0.1, use_normals=False,prop = 1,kdtree =None,og_pc=None):
    if normals is None and use_normals: raise AssertionError("use_normals is set to true but no normals were provided")
    best_vote = 3
    best_pt_plane = np.zeros((1,3))
    best_normal_plane = np.zeros((1,3))
    
    if kdtree : 
        pass
    else :
        candidates_idx = np.random.choice(len(points), size=(3 * nb_draws), replace=False)
        candidates = points[candidates_idx].reshape(nb_draws, 3, -1) # (n_draws, n_points_per_draw, points_coords)

    for draw in range(nb_draws):
        if kdtree : 
            r = np.random.choice(len(points), size=(1), replace=False)
            ind = kdtree.query(points[r], k=30, return_distance=False)
            candidates_idx = np.random.choice(ind[0], size=(3), replace=False)
            pt_plane, normal_plane = compute_plane(og_pc[candidates_idx])
        else : pt_plane, normal_plane = compute_plane(candidates[draw])
        votes = in_plane(points, pt_plane, normal_plane, threshold_in=threshold_in, use_normals=use_normals, normals=normals).sum()
        if votes > best_vote:
            best_vote = votes
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane
        if best_vote/len(points) > prop:
            break

    return best_pt_plane, best_normal_plane, best_vote,draw


# ================ RANSAC WITH NORMALS COMPUTED WITH PCA ==================
def compute_local_PCA(query_points, cloud_points, neighbors=50):
    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))
    pca = PCA(n_components=3)

    tree = KDTree(cloud_points)
    nneigh = tree.query(query_points, k=neighbors, return_distance=False) # array([array([...]), ...])

    for i, neigh_points_idx in tqdm(enumerate(nneigh)):
        neigh_points = cloud_points[neigh_points_idx]
        pca.fit(neigh_points)
        all_eigenvalues[i], all_eigenvectors[i] = pca.singular_values_, pca.components_.T
    
    return all_eigenvalues, all_eigenvectors



def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, use_normals=False, normals=None, normals_threshold=0.1,prop=1,kdtree = True,save_file = None):
    if normals is None and use_normals: raise AssertionError("use_normals is set to true but no normals were provided")

    nb_points = len(points)
    plane_inds = np.empty(0, dtype=np.int32)
    plane_labels = np.empty(0, dtype=np.int32)
    remaining_inds = np.arange(0,nb_points) # All the points are candidates at first

    if kdtree:
        tree = KDTree(points)
    else :
        tree = None


	
    for plane_idx in  tqdm(range(nb_planes)):
        best_pt_plane, best_normal_plane, _,needed = RANSAC(points=points[remaining_inds], 
                                                     nb_draws=nb_draws, 
                                                     threshold_in=threshold_in, 
                                                     use_normals=use_normals, 
                                                     normals=normals[remaining_inds],prop=prop,kdtree = tree,og_pc=points)
        points_in_plane = in_plane(points=points[remaining_inds], 
                                   pt_plane=best_pt_plane, 
                                   normal_plane=best_normal_plane, 
                                   threshold_in=threshold_in, 
                                   use_normals=use_normals, 
                                   normals=normals[remaining_inds],
                                   normals_threshold=normals_threshold)
        if save_file : print(needed,file = save_file)

        plane_inds = np.append(plane_inds, remaining_inds[points_in_plane])
        plane_labels = np.append(plane_labels, np.zeros(len(points_in_plane.nonzero()[0])) + plane_idx) 
        remaining_inds = remaining_inds[~points_in_plane] # keep only the points which were not selected    
    
    
    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file

    file_path = '../data/indoor_scan_normals.ply' #File with precomputed normal with the last block
    #file_path = '../data/Lille_street_small.ply'

    # Load point cloud
    data = read_ply(file_path)
    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = None
    use_normals = False
    if not "Lille" in file_path and "normals" not in file_path: # data not included
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        labels = data['label']
    
    if "normals" in file_path:
        normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        use_normals = True
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
    nb_points = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    if False:
        print('\n--- 1) and 2) ---\n')
        
        # Define parameter
        threshold_in = 0.10

        # Take randomly three points
        pts = points[np.random.randint(0, nb_points, size=3)]
        
        # Computes the plane passing through the 3 points
        t0 = time.time()
        pt_plane, normal_plane = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))
        
        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        # Save extracted indoor_scan_normalsane and remaining points
        write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    if False:
        print('\n--- 3) ---\n')

        # Define parameters of RANSAC
        nb_draws = 100
        threshold_in = 0.10

        # Find best plane by RANSAC
        t0 = time.time()
        best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
        
        # Find points in the plane and others
        points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        # Save the best extracted plane and remaining points
        if not "Lille" in file_path:
            write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
            write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        else:
            write_ply('../best_plane_lille.ply', [points[plane_inds]], ['x', 'y', 'z'])
            write_ply('../remaining_points_best_plane_lille.ply', [points[remaining_inds]], ['x', 'y', 'z'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #

    if True:
        print('\n--- 4) ---\n')
        
        # Define parameters of recursive_RANSAC
        if "Lille" not in file_path:
            nb_draws = 300
            threshold_in = 0.10
            nb_planes = 5
            normals_threshold = 0.1
            prop = 0.05
        else:
            nb_draws = 300
            threshold_in = 0.20
            nb_planes = 5
        
        # Recursively find best plane by RANSAC
        for i in range(10): #Multiple runs
            t0 = time.time()
            plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points=points, 
                                                                        nb_draws=nb_draws, 
                                                                        threshold_in=threshold_in, 
                                                                        nb_planes=nb_planes, 
                                                                        normals=normals, 
                                                                        use_normals=use_normals,
                                                                        normals_threshold=normals_threshold,prop = prop,kdtree=True,save_file=open("temp.txt",mode='a'))
            t1 = time.time()
            print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                        
        # Save the best planes and remaining points
        if "Lille" not in file_path:
            write_ply('../best_planes_normals_rec_fast.ply', [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
            write_ply('../remaining_points_best_planes_normals_rec_fast.ply', [points[remaining_inds], colors[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue'])
            # write_ply('../best_planes_rec.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
            # write_ply('../remaining_points_best_planes_rec.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        else:
            write_ply('../best_planes_rec_lille.ply', [points[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'plane_label'])
            write_ply('../remaining_points_best_planes_rec_lille.ply', [points[remaining_inds]], ['x', 'y', 'z'])
        
        print('Done')

    # ==> Code to find the radius to compute the normals of the point_cloud
    if False:
        neigh = 20
        print("Computing normals")
        t0 = time.time()
        _, all_eigenvectors = compute_local_PCA(points, points, neighbors=neigh)
        normals = all_eigenvectors[:, :, 2]
        t1 = time.time()
        write_ply('../data/indoor_scan_normals.ply', [points, colors, normals], ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz'])
        print(f"Normals computed in {t1-t0} seconds")
