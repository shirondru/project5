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

        What this method is doing:
        1. initialize a 1D array that will hold all Silhouette scores for each point, in the same order as the points themselves were given
        2. Grab the unique labels/clusters from y
        3. initialize a 2D array, `centroid_mat` that will hold each label's centroid
        4. Loop through each of the unique labels and calculate that label's centroid and store it in `centroid_mat`. 
            4a. Find centroid by taking the mean of each feature across all observations that have that label

        **** The calculations in 5-5b are performed in self._get_silhouette_distances ***
        5. Loop through all observations and get the mean distance between that point and all other points in the same cluster. Also get the distance between the current point and the closest cluster centroid
            5a. Use cdist to get distances between current point and all points with the same label. This includes the distance between the current point and itself
                5ab. The distance between the current point and itself = 0, so to get the mean distance between current point and other points in the same cluster sum the distances and divide by # of points in the cluster - 1
            5b. Grab the centroid vectors for all centroids except the one from the same cluster as the current point from `centroid_mat`. Calculate the distance between the current point and all the other centroids using cdist.
            Save the distance from the point to its closest centroid (from a different cluster)
            5c. Calculate the silhouette score for this point
            5d. save the score in the 1D array, `scores`. Order will be preserved
        

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            scores: np.ndarray
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
            mean_intra_distance,nearest_centroid_distance = self._get_silhouette_distances(X,y,unique_labels,centroid_mat,obs)
            
            ###calculate silhouette score and add it to scores array in the same order as the observations provided in y
            score = (nearest_centroid_distance - mean_intra_distance) / (max(nearest_centroid_distance,mean_intra_distance))
            scores[obs] = score
        return scores

    def _get_silhouette_distances(self,X: np.ndarray, y: np.ndarray, unique_labels: list,centroid_mat: np.ndarray, obs: int):

        """
        For the given observation, `obs` get the the mean intra-cluster distance and the distance between this point and it's nearest cluster centroid.
        Details for what this method is doing can be found in Steps 5-5b in the docstring of `self.score`


        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

            unique_labels: list
                a sorted list storing the names of the cluster labels

            centroid_mat: np.ndarray
                A 2D matrix containing the k cluster centroid vectors in each row
            obs: int
                the index (in y) of the current observation

        outputs:
            mean_intra_distance: float
                The mean distance between the current observation and all other observations in the same cluster
            nearest_centroid_distance: float
                The distance between the current obseration and its closeslt cluster centroid
        """

        #### Get mean distance between current point and all other points in the same cluster
        obs_label = y[obs]
        obs_in_cluster = y == obs_label #get bool array representing other observations in same cluster as obs            
        obs_data = X[obs,].reshape(1,X.shape[1]) #reshape to 2D array for cdist compatibility
        rest_of_cluster_data = X[obs_in_cluster,]
        intra_distances = cdist(obs_data,rest_of_cluster_data,metric = self._metric) #get distance between this obs and all obs with same label
        mean_intra_distance = np.sum(intra_distances) / (intra_distances.shape[1] - 1) #intra_distance includes the distance of current obs to itself, so divide by length of points in the cluster minus one to not include that in the average

        #### Get distance between current point and closest label/cluster centroid
        other_cluster_centroid_rows = [x for x in range(len(unique_labels)) if x != unique_labels.index(obs_label)] #this is a list corresponding to all other centroid rows in centroid_mat
        other_cluster_centroids = centroid_mat[other_cluster_centroid_rows,] 
        inter_distances  = cdist(obs_data, other_cluster_centroids) #distance between current point and all other label/cluster centrouds
        nearest_centroid_idx = np.argmin(inter_distances) 
        nearest_centroid_distance = inter_distances[0,nearest_centroid_idx] 
    
        return mean_intra_distance,nearest_centroid_distance

    
    
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


