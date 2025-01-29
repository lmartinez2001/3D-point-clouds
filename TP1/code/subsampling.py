#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from collections import defaultdict

from tqdm import tqdm


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, colors, labels, box_size=np.array([80,200,300])):

    # Compute the AABB box
    bounding_box_min = points.min(axis=0)
    bounding_box_max = points.max(axis=0)

    # Associate each point to a box 
    norm = bounding_box_max-bounding_box_min
    normalized_points = (points - bounding_box_min)/norm
    associated_box =(normalized_points * box_size).astype('int')

    n1,n2,n3 = box_size
    associated_box_flattened = associated_box[:,0] * (n2 * n3) \
                + associated_box[:,1] * n3 + associated_box[:,2] 
    
    # sum the points associated to the same box
    label_sum = np.bincount(associated_box_flattened, weights=labels)

    R,G,B = (np.bincount(associated_box_flattened, weights=colors[:,i]) 
                        for i in range(3))
    X,Y,Z = (np.bincount(associated_box_flattened, weights=points[:,i]) 
                        for i in range(3))

    # Count the points associated to the same box and compute the mean
    grid_count = np.bincount(associated_box_flattened)

    decimated_points = np.column_stack((X,Y,Z))[grid_count>0]
    decimated_colors = np.column_stack((R,G,B))[grid_count>0]
    decimated_labels = label_sum[grid_count>0].reshape(-1, 1)

    grid_count_norm = grid_count[grid_count>0].reshape(-1, 1)

    decimated_points /= grid_count_norm
    decimated_colors /= grid_count_norm
    decimated_labels /= grid_count_norm

    # Correct dtypes
    decimated_colors = decimated_colors.astype('uint8')
    decimated_labels = decimated_labels.astype('int32')

    return decimated_points, decimated_colors, decimated_labels





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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 10

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Grid subsampling
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = grid_subsampling(points, colors, labels, box_size=np.array([80,200,300]))
    t1 = time.time()
    print('Grid subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_sub.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    print('Done')
