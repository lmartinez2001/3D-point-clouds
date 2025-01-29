#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
import random

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

from tqdm import tqdm
np.random.seed(0)
random.seed(0)

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

# Utility functions
def plot_with_margins(x, arr, n_runs, save_name=None, xlabel=None, ylabel=None):
    avg_arr = arr.mean(axis=0)
    std_arr = arr.std(axis=0) / np.sqrt(n_runs)
    
    plt.figure()
    plt.plot(x, avg_arr, label='Mean', color='red')
    plt.fill_between(x, avg_arr - std_arr, avg_arr + std_arr, color='blue', alpha=0.3, label='Standard Error')
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.legend()
    if save_name: 
        plt.savefig(save_name)
        print(f'Plot saved at {save_name}')
    plt.show()


def compute_distances(queries, supports):
    return np.linalg.norm(queries[:,None,:] - supports[None,:,:], axis=2) # n_queries, n_support


def brute_force_spherical(queries, supports, radius):
    distances = compute_distances(queries, supports)
    neighborhoods = []
    
    for query_dist in distances:
        neigh = np.argwhere(query_dist < radius)
        neighborhoods.append(list(neigh))
        
    return neighborhoods


def brute_force_KNN(queries, supports, k):
    distances = compute_distances(queries, supports)
    neighborhoods = []
    for query_dist in distances:
        neigh = np.argpartition(query_dist, k)
        neighborhoods.append(list(neigh))
    return neighborhoods


def impl_KDD(queries, supports, radius, leaf_size=5):
    t0 = time.time()    
    tree = KDTree(supports, leaf_size=leaf_size)
    t1 = time.time()         
    ind = tree.query_radius(queries, r=0.3)

    query_time = time.time()-t1
    creation_time = t1-t0
    
    return ind,query_time,creation_time


def compute_opt_leaf_size(queries, points, radius, n_runs, save_query='query_plot.png', save_creation='creation_plot.png'):
    print(queries.shape)
    leaf_size_candidates = np.arange(1, 401, 20, dtype=np.uint16)
    query_times, creation_times = np.zeros((n_runs, len(leaf_size_candidates))), np.zeros((n_runs, len(leaf_size_candidates)))
    
    for run in tqdm(range(n_runs)):
        for idx, ls in enumerate(leaf_size_candidates):
            neighborhoods, query_time, creation_time = impl_KDD(queries, points, radius, leaf_size=ls)
            query_times[run, idx] = query_time
            creation_times[run, idx] = creation_time

    query_means = query_times.mean(axis=0)
    opt_leaf_size = leaf_size_candidates[np.argmin(query_means)]
    print(f'Min time reached with {opt_leaf_size}')
    print(query_times.shape)
    plot_with_margins(leaf_size_candidates, query_times, n_runs, save_name=save_query, xlabel='Leaf size', ylabel='Time (s)')
    plot_with_margins(leaf_size_candidates, creation_times, n_runs, save_name=save_creation, xlabel='Leaf size', ylabel='Time (s)')
    return opt_leaf_size


def compute_radius_effects(queries, points, n_runs, leaf_size, save_name='radius_plot.png'):
    print(f'Computing radius effect with optimal leaf size {leaf_size}...')
    radius_candidates = np.arange(0.01, 0.3, 0.01)
    query_times = np.zeros((n_runs, len(radius_candidates))) 
    tree = KDTree(points, leaf_size=leaf_size)

    for run in tqdm(range(n_runs)):
        for r_idx, radius in enumerate(radius_candidates):
            t = time.time()         
            ind = tree.query_radius(queries, r=radius)
            query_time = time.time() - t
            query_times[run, r_idx] = query_time

    query_means = query_times.mean(axis=0)
    query_std = query_times.std(axis=0) / np.sqrt(n_runs)
    print(f'Min reached for r={radius_candidates[np.argmin(query_means)]}')
    plot_with_margins(radius_candidates, query_times, n_runs, save_name=save_name, xlabel='Radius', ylabel='Time (s)')



# ------------------------------------------------------------------------------------------
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
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # Define the search parameters
    neighbors_num = 100
    radius = 0.2
    num_queries = 10

    n_runs = 10 # Number of time experiements are repeated to get meaningful statistics
    
    print("Size of Point Cloud %d"%points.shape[0])

    # Pick random queries
    random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
    queries = points[random_indices, :]
    
    if False:

        # Search spherical
        times_spherical = np.zeros(n_runs)
        for run in range(n_runs):
            t0 = time.time()
            neighborhoods = brute_force_spherical(queries, points, radius)
            t1 = time.time()
            times_spherical[run] = t1-t0
        mean_spherical = times_spherical.mean()
        std_spherical = times_spherical.std() / np.sqrt(n_runs)
        print(f'{num_queries} spherical neighborhoods computed in {mean_spherical:.3f} seconds (+/- {std_spherical:.3f})')

        # Search KNN
        times_knn = np.zeros(n_runs)
        for run in tqdm(range(n_runs)):
            t0 = time.time()
            neighborhoods = brute_force_KNN(queries, points, neighbors_num)
            t1 = time.time()
            times_knn[run] = t1-t0
        mean_knn = times_knn.mean()
        std_knn = times_knn.std() / np.sqrt(n_runs)
        print(f'{num_queries} KNN computed in {mean_knn:.3f} seconds (+/- {std_knn:.3f})')

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * mean_spherical / num_queries
        total_KNN_time = points.shape[0] * mean_knn / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))



        # KDTree neighborhoods
        # ********************
        query_times, creation_times = np.zeros(n_runs), np.zeros(n_runs)
        for run in tqdm(range(n_runs)):
            neighborhoods, query_time, creation_time = impl_KDD(queries, points, radius,leaf_size=100)
            query_times[run] = query_time
            creation_times[run] = creation_time
        mean_query = query_times.mean()
        std_query = query_times.std() / np.sqrt(n_runs)

        mean_creation = creation_times.mean()
        std_creation = creation_times.std() / np.sqrt(n_runs)
        total_kdtree_query_time  = points.shape[0] * mean_query / num_queries
        
        print('{:d} KDD (Query Only) computed in {:.3f} seconds'.format(num_queries, mean_query))
        print('Computing spherical neighborhoods on whole cloud with KDTree (Query Only): {:.01f} hours'.format(total_kdtree_query_time / 3600))

    num_queries = 1000
    n_runs = 5

    random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
    queries = points[random_indices, :]

    opt_leaf_size = compute_opt_leaf_size(queries, points, radius, n_runs)
    compute_radius_effects(queries, points, n_runs, leaf_size=opt_leaf_size)
        
        
        
        