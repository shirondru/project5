import numpy as np
from scipy.spatial.distance import cdist
import random
class KMeans:
    def __init__(
            self,
            k: int = 8,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100,
            random_state: int = 0):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting. Default = 8
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self._tol = tol
        self._max_iter = max_iter
        self.random_state = random_state
        
        
        possible_metrics = [
         'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
         'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 
         'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 
         'sqeuclidean', 'wminkowski', 'yule'
                           ]
        if metric not in possible_metrics:
            raise ValueError(f"The distance metric must be one of:\n {possible_metrics}")
        self._metric = metric
        
        
        
        
    def _MSE(self, mat: np.ndarray, centroid_mat: np.ndarray,cluster_assignments:np.ndarray):
        """
        Calculates Mean squared error between observations and their corresponding cluster centers

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
            centroid_mat: np.ndarray
                A 2D matrix with k rows and mat.shape[1] columns where each rows is a clusters centroid vector. 
            cluster_assignments: np.ndarray
                A 1D array of length(observations) that holds the cluster assignment for each observation

        outputs:
            float
                The mean-squared error between observations and their corresponding cluster center

        """
        
        if centroid_mat.shape[1] != mat.shape[1]:
                raise AssertionError(f"The number of features in the data do not match the number of features model was fitted on")
        
                                
        
        distances = np.ndarray((0,1)) #distances between observations and their cluster centroid will go in here
        for cluster in range(1,self.k+1):
            this_cluster_data = mat[cluster_assignments == cluster,] #get rows with observations assigned to current cluster
            centroid_vector = centroid_mat[cluster - 1,] #get coordinates of corresponding cluster centroid

            #get distance (error) between observations assigned to this cluster and cluster centroid and append 
            #to list of distances for observations corresponding to other clusters
            distances = np.concatenate([cdist(this_cluster_data,np.reshape(centroid_vector,(1,mat.shape[1])),metric = self._metric),distances])

        return np.mean(distances**2)
    
    
    def _check_dimensions(self,mat: np.ndarray):
        """
        Checks that the input matrix `mat` has the correct number of features (columns) for comparisons to be made against the cluster centroids.
        For example, if the model is fitted on a dataset with 5 dimensions, the cluster centroids will be in a 5D space. Attempting to perform some operation,
        such as predictions, on a dataset with less than or more than 5 features will cause this method to raise an error.
        
        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        
        #### Check that the input matrix `mat` has the same number of features as the fitted centroid vectors.

        if self._centroid_locations.shape[1] != mat.shape[1]:
                raise AssertionError(f"The number of features in your new data do not match the number of features model was fitted on")
        
                                       

    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
                
                
                
        What this method is doing:
        1. Initialize a matrix (dim = k,features), `centroid_mat`, to store each of the k clusters' centroid vector
            2a. Do this by selecting k random points from `mat`, without replacement.
        2. Assign each data point to one of the k clusters based on which centroid it is closest to. Store these cluster assignments in `cluster_assignments`
        3. Until the max iterations is reached or the cluster assignments stop changing:
            a. Overwrite `centroid_mat` by introducing each cluster's centroid into a row of centroid_mat. The centroid for cluster 1 goes into row 0
                aa. The centroid is calculated by taking the mean of each feature across all observations currently assigned to that cluster. These means form each component of the centroid vector
            b. Assign each observation to the cluster with the closest centroid. These new assignments are stored in `new_cluster_assignments`
                ba. use scipy.spatial.distance.cdist to compute the distance between each observation vector and each centroid vector
                bb. Here, cdist(mat,centroid_mat) returns a 2D matrix where the (i,j) element is the distance between observation from row i of `mat` and the jth cluster centroid
                bc. np.argmin finds the index of each row corresponding to the centroid with the minimum distance from the observation in that row. This index is stored in an array of length(observations)
                bd. Add 1 to each element in the array to convert the indices (which start at 0) to clusters (which starts at 1)
            c. Check for convergence:
                Two checks for convergence:
                1) if `new_cluster_assignments` is equal to `cluster_assignments` 
                2) if the mean squared error between the observations assigned to a cluster and those cluster centroids in the current iteration is similar to that of the previous iteration (within a tolerance `tol`). 
                ca. Break the loop if the algorithm has converged. The current centroid vectors in `centroid_mat` are the final coordinates for the centroids for each cluster
                cb. If there is no convergence, overwrite `cluster_assignments` with the cluster assignments in `new_cluster_assignments` to calculate new centroids and check if the next iteration has converged
                
        
        """
        
        
        np.random.seed(self.random_state)
        iter_num = 0
        
        ##### Randomly initialize centroids. Each cluster's centroid is stored as a row ######
        centroid_mat = self._centroid_init(mat)
        
        ####### assign all points to closest custer centroid ########
        #use np.argmin to get index of centroid with minimum  distance to that observation
        #add 1 to convert index to cluster (i.e, index 0 ==> cluster 1)
        cluster_assignments = np.argmin(cdist(mat,centroid_mat,metric = self._metric),axis = 1) + 1
        
        
        self.convergence_status = "Max iter reached" #if this is not overwritten below, that means KMeans did not converge and only stopped because max_iter was reached
        while iter_num <= self._max_iter:
            plt.scatter(
                mat[:,0], 
                mat[:,1], 
                c=cluster_assignments)
            plt.scatter(centroid_mat[:,0],centroid_mat[:,1],c='red')
            plt.show()
            plt.close()
            
            if iter_num > 0:
                #get previous mse before overwriting centroid_mat and cluster_assignments
                previous_mse = self._MSE(mat,centroid_mat,cluster_assignments)
            
            ####### recompute each cluster's centroid and store it as a row in centroid_mat ########
            for cluster in range(1,self.k+1):
                this_cluster_data = mat[cluster_assignments == cluster,] #get rows with observations assigned to current cluster
                centroid_mat[cluster - 1,] = this_cluster_data.mean(axis=0) #new centroid for this cluster is a vector with mean of each feature column as its components


            ####### reassign points to closest cluster centroid after re-calculating each cluster's centroid #######
            new_cluster_assignments = np.argmin(cdist(mat,centroid_mat,metric = self._metric),axis = 1) + 1
            
            
            if iter_num > 0:
                #get current mse with new centroid mat and new cluster assignments
                current_mse = self._MSE(mat,centroid_mat,new_cluster_assignments)
                
                #check for convergence
                if np.array_equal(new_cluster_assignments,cluster_assignments): #strict convergence check; stop if cluster assignments aren't changing
                    self.convergence_status = "Strict Convergence"
                    break
                elif abs(previous_mse - current_mse) <= self._tol: # soft convergence check
                    self.convergence_status = "Soft Convergence"
                    break
                    
            cluster_assignments = new_cluster_assignments 
            iter_num +=1
        
        
        self.fitted = True
        print(self.convergence_status)
        
        #save some of the results and training data as a private attribute to be accessed easily by other methods
        self._centroid_locations = centroid_mat
        self._training_clusters = new_cluster_assignments
        self._training_mat = mat
        
        
        return self
    
    
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix by finding the closest cluster centroid to
        each observation in the provided matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features.
                

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        if self.fitted:
            self._check_dimensions(mat) # Check that the input matrix `mat` has the same number of features as the fitted centroid vectors.
                
                
            #assign each observation to the cluster with the closest centroid:
            #use np.argmin to get index of centroid with minimum  distance to that observation
            #add 1 to convert index to cluster (i.e, index 0 ==> cluster 1)
            return np.argmin(cdist(mat,self._centroid_locations,metric = self._metric),axis = 1) + 1

            
            
        else:
            raise AssertionError(f"You must fit the model before attempting any predictions")

    def get_error(self, mat:np.ndarray = None) -> float:
        """
        returns the final mean-squared distance of the fitted model with respect to the data in `mat`
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
                By default this will use the training data and therefore output the training MSE unless a different
                mat is provided

        outputs:
            float
                the mean-squared error (distance) between the observations in `mat` and the cluster centroids
                in the fitted model.
        """
        if self.fitted:
            
            if mat is None:
                mat = self._training_mat
            self._check_dimensions(mat) #Check that the input matrix `mat` has the same number of features as the fitted centroid vectors.
            return self._MSE(mat, self._centroid_locations,self._training_clusters)

        else:
            raise AssertionError(f"You must fit the model to get the cluster centroids")
        
        

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        
        if self.fitted:
            return self._centroid_locations
        else:
            raise AssertionError(f"You must fit the model to get the cluster centroids")
            
            
    def _centroid_init(self,mat: np.ndarray):
        """
        
        Initializes cluster centroids in a smart way to reduce possibility of poor clustering.
        Initialization is done as follows:
        1) First cluster centroid is chosen randomly from one of the data points in mat
        2) Compute the distance from each point (except the point used for the first centroid) to the closest centroid, for all centroids that have been already initialized
        3) Choose the point that is farthest from its nearest centroid as the next centroid
        4) Repeat 2 and 3 
        
        
        Reference: https://en.wikipedia.org/wiki/K-means%2B%2B
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        outputs:
            centroid_mat: np.ndarray
            a `k x m` 2D matrix representing initial cluster centroids
        
        """
        #initialize 2D array to hold centroids in
        centroid_mat = np.ndarray(shape = (self.k,mat.shape[1]))
        
        ##Choose first cluster randomly
        random_point = np.random.choice(mat.shape[0], 1, replace=False)
        centroid_mat[0,:] = mat[random_point, :] 
        
        centroid_idxs = [random_point[0]] #initialize list that will hold the row in `mat` the centroid was in, to prevent its use in steps 2-4
        
        #begin computing the rest of the centroids
        for cluster in range(1,k+1):
            non_centroid_points = [x for x in range(mat.shape[0]) if x not in centroid_idxs] #all other rows in `mat` besides those already being used as centroids
            
            #1) Compute distance between all non-centroid points and each centroid
            #2) For each point, keep the distance between it and the closest centroid
            #3) Of those remaining minimum distances, make the maximum one the next centroid; this is the point that is farthest from it's nearest centroid
            if len(centroid_idxs) == 1: #if only 1 centroid has been defined, reshape that centroid vector to be 2D and compatible with cdist()
                filtered_mat = mat[non_centroid_points,]
                distances = cdist(filtered_mat,centroid_mat[0:len(centroid_idxs),].reshape(1,mat.shape[1]))               
                min_distances_idxs = np.argmin(distances,axis=1)
                min_distances = distances[np.arange(len(distances)),np.argmin(distances,axis=1)]
                new_centroid_idx = np.argmax(min_distances)
                new_centroid = filtered_mat[new_centroid_idx,]
                centroid_mat[cluster,] = new_centroid
                
                #current new_centroid_idx might be shifted in `mat` because point(s) only non_centroid_points were used
                shift = 0
                for centroid_idx in centroid_idxs:
                    if new_centroid_idx < centroid_idx:
                        continue
                    else:
                        shift +=1
                centroid_idxs.append(new_centroid_idx + shift)
                print(new_centroid)
                print(mat[new_centroid_idx + shift,])
                print('*'*20)
                assert(np.array_equal(mat[new_centroid_idx + shift,],new_centroid))
                        
               
            else: #otherwise 
                filtered_mat = mat[non_centroid_points,]
                distances = cdist(filtered_mat,centroid_mat[0:len(centroid_idxs),])
                min_distances_idxs = np.argmin(distances,axis=1)
                min_distances = distances[np.arange(len(distances)),np.argmin(distances,axis=1)]
                new_centroid_idx = np.argmax(min_distances)
                new_centroid = filtered_mat[new_centroid_idx,]
                
                centroid_mat[cluster,] = new_centroid
                
                #current new_centroid_idx might be shifted in `mat` because point(s) only non_centroid_points were used
                shift = 0
                for centroid_idx in centroid_idxs:
                    if new_centroid_idx < centroid_idx:
                        continue
                    else:
                        shift +=1
                centroid_idxs.append(new_centroid_idx + shift)
                assert(np.array_equal(mat[new_centroid_idx + shift,],new_centroid))
                        
            return centroid_mat
                       
                       
                       
   
            
            