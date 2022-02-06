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


