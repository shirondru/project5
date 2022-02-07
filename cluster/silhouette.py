import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        possible_metrics = [ #possible metrics supported by scipy.spatial.distance
         'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
         'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 
         'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 
         'sqeuclidean', 'wminkowski', 'yule'
                           ]
        if metric not in possible_metrics:
            raise ValueError(f"The distance metric must be one of:\n {possible_metrics}")
        self._metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        self._check_dimensions(X,y) #check you have a label for every observation
        
        scores = np.ndarray(shape=len(y))
        
        ### get position of each label's centroid
        unique_labels = sorted(list(set(y)))
        centroid_mat = np.ndarray(shape=(len(unique_labels),X.shape[1])) #store each label's centroid in this 2D matrix
        
        for idx,label in enumerate(unique_labels):
            obs_in_cluster = y == label
            cluster_data = X[obs_in_cluster,]
            centroid_mat[idx,] = cluster_data.mean(axis=0) #store centroid for this label. Use idx not label for assignment in case labels are not contiguous and/or do not start at 0
        
        for obs in range(len(y)):
            #### Get mean distance between current point and all other points in the same cluster
            obs_label = y[obs]
            obs_in_cluster = y == obs_label #get bool array representing other observations in same cluster as obs            
            obs_data = X[obs,].reshape(1,X.shape[1]) #reshape to 2D array for cdist compatibility
            rest_of_cluster_data = X[obs_in_cluster,]
            intra_distances = cdist(obs_data,rest_of_cluster_data,metric = self._metric) #get distance between this obs and all obs with same label
            mean_intra_distance = np.sum(intra_distances) / (intra_distances.shape[1] - 1) #intra_distance includes the distance of current obs to itself, so divide by length of points in the cluster minus one to not include that in the average

            #### Get distance between current point and closest label/cluster centroid
            other_cluster_centroid_rows = [x for x in range(len(unique_labs)) if x != unique_labs.index(obs_label)] #this is a list corresponding to all other centroid rows in centroid_mat
            other_cluster_centroids = centroid_mat[other_cluster_centroid_rows,] 
            inter_distances  = cdist(obs_data, other_cluster_centroids) #distance between current point and all other label/cluster centrouds
            nearest_centroid_idx = np.argmin(inter_distances) 
            nearest_centroid_distance = inter_distances[0,nearest_centroid_idx] 
            
            ###calculate silhouette score and add it to scores array in the same order as the observations provided in y
            score = (nearest_centroid_distance - mean_intra_distance) / (max(nearest_centroid_distance,mean_intra_distance))
            scores[obs] = score
        return scores
    
    
    def _check_dimensions(self, X: np.ndarray, y: np.ndarray):
        """
        Checks the number of observations in X is the same as the number of labels in y

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            Raises an error if the number of rows in X is different than the number of labels in y
        """

        if X.shape[0] != len(y):
            raise AssertionError(f"The number of observations is different than the number of provided labels")

            
#     def _get_pairwise_distances(self)


