# Write your k-means unit tests here
from cluster import make_clusters, KMeans
import numpy as np
from scipy.spatial.distance import cdist
import pytest

def test_check_easy_cluster():
    """
    This test checks to see that the KMeans algorithm properly clusters a very easily clusterable dataset.
    I use make_clusters with k=3 and a very low scale to ensure the 3 clusters are high distinct. 
    I then loop through each of these truth clusters and identify the samples in these clusters. I then check what cluster(s)
    the KMeans algorithm predicted the same samples are in, and I assert that they are all in only one cluster; because the dataset
    is so easily clusterable, it should have perfectly clustered these samples, even if it didn't name the clusters in the same way.
    """

    # create tight clusters
    k = 3
    clusters, labels = make_clusters(k=k,scale=0.3)

    #fit KMeans and predictions for these samples
    km = KMeans(k=k,random_state = 0)
    km.fit(clusters)
    pred = km.predict(clusters)

    # for each true cluster, create 
    for clust in range(k):
        samples_in_true_cluster = labels==clust 
        pred_labels_for_true_cluster = pred[samples_in_true_cluster,] #get the predicted cluster labels for the same samples
        
         #check the predicted cluster labels for all samples truly in one cluster are actually in just one cluster
        assert len(np.unique(pred_labels_for_true_cluster)) == 1, "Your model was not able to properly cluster an easy dataset in the same way as the truth"
        

def test_centroid_init_randomness():
    """
    Although the centroid initialization `self._centroid_init` is implemented via the KMeans++ method, the very first initialized centroid is chosen randomly from one of the 
    data points, and which point is chosen for the first initial centroid will affect the initialization of the rest of the centroids.
    If this initialization is implemented correctly, it should be pseudo-random, and varying the random seed will affect the centroid initialization.
    Assert that is the case
    """
    k=5
    clusters, labels = make_clusters(k=k,scale=1.5)
    centroid_mats = []
    for seed in [12312,23423523,1313158291]:
        km = KMeans(k=k,random_state = seed)
        centroid_mats.append(km._centroid_init(clusters)) #append 2D matrix with centroids to list

    #assert initial centroids that came from initialization with different random seed are different
    assert not np.allclose(centroid_mats[0],centroid_mats[1]), "Changing random seed did not change centroid initialization!"
    assert not np.allclose(centroid_mats[0],centroid_mats[2]), "Changing random seed did not change centroid initialization!"
    assert not np.allclose(centroid_mats[1],centroid_mats[2]), "Changing random seed did not change centroid initialization!"


def test_overfit_capability():
    """
    As k increases, the training MSE should decrease because the algorithm will begin to overfit to the data. Here, I test that the training MSE
    of the algorithm decreases as I re-fit the model on the same data with increasing k. I use loose clusters that are not easily distinguishable to 
    demonstrate that increasing k will decrease training MSE even in cases that are difficult to cluster. 
    """
    errors = []
    for k in [3,10,25,50]:
        clusters, labels = make_clusters(scale=2,k=4)
        km = KMeans(k=k,random_state = 0)
        km.fit(clusters)
        errors.append(km.get_error())

    #assert training MSE decreases as you increase k
    assert errors[0] > errors[1] > errors[2] > errors[3], "MSE does not decrease with increasing k!"
     

def test_check_easy_cluster2():
    """
    This test checks to see that the KMeans algorithm properly clusters a very easily clusterable dataset.
    It does this checking that the MSE of each point with its cluster centroid is low, which would suggest tight clustering,
    which is expected with the easily clusterable data used in this test
    This test will assert that MSE is less than the distance between the two further points. This will show that, in general, points are closer to their 
    cluster centroids than two points that are relatively far apart. I will use the same k for KMeans clustering as there are clusters in the true data,
    to avoid getting a low MSE due to overfitting with a high k
    This also tests the _MSE method; if it works correctly it should give a small value here.
    """
    k = 3
    clusters, labels = make_clusters(k=k,scale=0.3)

    #fit KMeans and predictions for these samples
    km = KMeans(k=k,random_state = 0)
    km.fit(clusters)
    pred = km.predict(clusters)

    max_distance = np.max(cdist(clusters,clusters).ravel()) #max distance between any two points in the dataset
    training_mse = km.get_error()
    assert training_mse < max_distance, "Your model was not able to properly cluster an easy dataset and get a low MSE"


def test_label_assignment():
    """
    Trying to predict labels from the same data that was used to fit the model should return the same labels that were assigned during the fitting process
    If that is not the case, something has gone wrong either in the KMeans.fit or KMeans.predict methods.
    Therefore, here I assert that the cluster assignments generated during training, stored in a private attribute KMeans._training_clusters,
    are the same as the cluster assignments KMeans.predict produces on the data.
    """
    k = 4
    # Generate clustered data, train KMeans, get clustering predictions on the training data
    clusters, labels = make_clusters(scale=0.3,k=k)
    km = KMeans(k=k,random_state = 0)
    km.fit(clusters)
    pred_labels = km.predict(clusters)

    assert np.array_equal(km._training_clusters,pred_labels), "cluster assignments during fitting are different than predictions on the training data!"

