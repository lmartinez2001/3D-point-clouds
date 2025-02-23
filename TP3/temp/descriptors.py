#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
import matplotlib.pyplot as plt
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm

# --------------------------------centered_p@centered_p.T----------------------------------------------------------
#
#           Functions
#    centered_p@centered_p.T   \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



import numpy as np

def PCA(points):
    # Check if input is a single point or a batch
    no_batch = False
    if points.ndim == 2:
        # Single point case
        no_batch = True
        points = np.expand_dims(points, axis=0)
    
    # Center the points by subtracting mean
    centered_p = points - points.mean(axis=1,keepdims=True)
    
    # Compute covariance matrix
    cov = (np.transpose(centered_p,axes = (0,2,1)) @ centered_p) / points.shape[1]
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if no_batch :
        eigenvalues, eigenvectors = eigenvalues[0], eigenvectors[0]
    
    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points)

    ind_batches = tree.query_radius(query_points, r=radius)

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))

    for ind,batch in enumerate(tqdm(ind_batches)) :
        all_eigenvalues[ind],all_eigenvectors[ind] = PCA(cloud_points[batch])

    return all_eigenvalues, all_eigenvectors

def compute_local_PCA_neighbors(query_points, cloud_points, neighbors,hist = False):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points)

    dist,ind_batches = tree.query(query_points, k=neighbors)
    if hist :
        plt.hist(dist.flatten(),bins=100)
        plt.xlabel("Distance")
        plt.show()
    
    all_eigenvalues,all_eigenvectors = PCA(cloud_points[ind_batches])

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius,neighbors=None):
    
    if neighbors :
        all_eigenvalues, all_eigenvectors = compute_local_PCA_neighbors(query_points, cloud_points, neighbors)
    else :
        all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)

    lambda_1,lambda_2,lambda_3 = all_eigenvalues[:,-1],all_eigenvalues[:,-2],all_eigenvalues[:,-3]
    normals = all_eigenvectors[:, :, 0]

    verticality = 2*np.arcsin(np.abs(normals[:,2]))/(np.pi)

    lambda_1 += 1e-6
    linearity = 1 - lambda_2/lambda_1
    planarity = (lambda_2 - lambda_3)/lambda_1
    sphericity = lambda_3/lambda_1

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA_neighbors(cloud, cloud, 30,hist=False)
        #all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
        # Add features
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.5 )
        all_features = (cloud, normals,verticality, linearity, planarity, sphericity)
        write_ply('../Lille_street_small_features_rad.ply', all_features, ['x', 'y', 'z', 'nx', 'ny', 'nz','v','l','p','s'])
        
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.5 ,neighbors=30)
        all_features = (cloud, normals,verticality, linearity, planarity, sphericity)
        write_ply('../Lille_street_small_features_neigh.ply', all_features, ['x', 'y', 'z', 'nx', 'ny', 'nz','v','l','p','s'])
