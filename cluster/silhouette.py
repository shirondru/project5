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
            5b. Loop through all cluster labels except the one corresponding to the current observation. In each iteration of the loop, find the mean distance between the current observation and all the points in other cluster. 
            Append this mean distance to a list, and return the smallest mean distance after all iterations. This is the smallest distance between the current point and all points in another cluster
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
        self._print_warning = False
        
        scores = np.ndarray(shape=len(y))
        
        ### get position of each label's centroid
        unique_labels = sorted(list(set(y)))
        centroid_mat = np.ndarray(shape=(len(unique_labels),X.shape[1])) #store each label's centroid in this 2D matrix
        
        for idx,label in enumerate(unique_labels):
            obs_in_cluster = y == label
            cluster_data = X[obs_in_cluster,]
            centroid_mat[idx,] = cluster_data.mean(axis=0) #store centroid for this label. Use idx (not label) for assignment in case labels are not contiguous and/or do not start at 0
        
        for obs in range(len(y)):
            mean_intra_distance,min_mean_inter_distance = self._get_silhouette_distances(X,y,unique_labels,centroid_mat,obs)
            
            ###calculate silhouette score and add it to scores array in the same order as the observations provided in y
            score = (min_mean_inter_distance - mean_intra_distance) / (max(min_mean_inter_distance,mean_intra_distance))
            scores[obs] = score

        if self._print_warning: #print a warning if any clusters have only one data point. This would lead to a NaN silhouette score
            print("Warning! One or more clusters contain only one data point. This suggests previous clustering might be  unreliable.\n" \
                  "This could be the number of data points is not much greater than the number of clusters!"\
                 f"# clusters: {len(unique_labels)}\nn_observations:{X.shape[0]}")


        return scores

    def _get_silhouette_distances(self,X: np.ndarray, y: np.ndarray, unique_labels: list,centroid_mat: np.ndarray, obs: int):

        """
        For the given observation, `obs` get the the mean intra-cluster distance and smallest distance between this point and all other points in another cluster.
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

        if rest_of_cluster_data.shape[0] == 1:
            #print a warning after calculating all scores to alert user that having only one data point in a cluster will lead to NaN silhouette scores
            self._print_warning = True


        #get minimum mean distance of current observation to all points in any of the other clusters
        mean_inter_distances = []
        for other_label in [x for x in unique_labels if x != obs_label]: #loop through labels corresponding to other clusters
            mean_inter_distances.append(np.mean(cdist(obs_data,X[y==other_label]))) #append mean distance of current point to all points in other cluster to list
        min_mean_inter_distance = min(mean_inter_distances) 


    
        return mean_intra_distance,min_mean_inter_distance

    
    
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


