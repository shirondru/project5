# write your silhouette score unit tests here
from cluster import make_clusters, KMeans, Silhouette
import numpy as np
from scipy.spatial.distance import cdist
import pytest


def test_silhouette():
    """
    If the silhouette score was implemented properly, points near the center of a cluster should have higher scores than points that are near the edge of a cluster
    Therefore, here I implement the KMeans algorithm, calculate the silhouette score on the predictions, and find the 5 points with the highest silhouette score
    and the 5 points with the lowest silhouette scores. I then compare the average distance between these two sets of points to their closest cluster centroid.
    If the score was implemented correctly, the 5 points with the highest silhouette scores should be closer on average to their cluster centroid than the 5 points with the lowest silhouette score.
    Assert that is the case


    Here is how i do this:
    1. Generate clustered data, perform fitting, perform predictions, calculate silhouette scores
    2. Take each score in the silhouette scores array `scores` and place it inside a len(2) tuple with that scores index in `scores`. Store all of these 
    tuples in `new_scores`.
    `	3. Sort new_scores and index the first 5 and last 5 scores to get the 5 smallest and 5 largest silhouette scores. I am able to reference the original index 
    of these points in `scores` because of the second element in the tuple. This idx also corresponds to the observation's location in `clusters` (I test order preservation with a different unit test to confirm this)
    4. Use these indices to get the 5 largest and 5 smallest points from `clusters`
    5. Get the distance from all of these points to their closest cluster centroid, and find the average between the 5 points with the smallest silhouette score
    and the 5 points with the largest silhouette score.
    6. Assert that the 5 points with the largest silhouette score are closer on average to their closest centroid  than the 5 points with the smallest silhouette score
    """

    #Generate clustered data, train KMeans, cluster the data, calculate silhouette scores
    clusters, labels = make_clusters(scale=1.5,m=3)
    km = KMeans(k=3,random_state = 0)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)


    new_scores = []
    for idx,score in enumerate(scores):
        new_scores.append((score,idx)) #store each score as a tuple with its original index in a new list
        
    new_scores.sort() #sort new list by the score so I can easily index out points with high and low silhouette score
    min_scores = new_scores[:5]
    max_scores = new_scores[-5:]


    original_min_idxs = []
    for tup in min_scores:
        original_min_idxs.append(tup[1]) #get the original (pre-sorted) index of each of the 5 values
    pts_with_min_scores = clusters[original_min_idxs,] #use the pre-sorted index to extract the original observation vectors

    #get the average distance of the 5 points with the smallest silhouette scores to their closest centroid
    min_pts_avgdist_to_closest_centroid = np.mean(np.min(cdist(pts_with_min_scores,km.get_centroids()),axis=1))
        

    original_max_idxs = []
    for tup in max_scores:
        original_max_idxs.append(tup[1])
    pts_with_max_scores = clusters[original_max_idxs,]

    #get the average distance of the 5 points with the largest silhouette scores to their closest centroid
    max_pts_avgdist_to_closest_centroid = np.mean(np.min(cdist(pts_with_max_scores,km.get_centroids()),axis=1))

    assert max_pts_avgdist_to_closest_centroid < min_pts_avgdist_to_closest_centroid, "Points with large silhouette scores are farther away from their closest centroid than points with small silhouette scores!"


def test_order_preservation():
    """
    Test that the order the scores are returned from Silhouette.score are consistent with the order of observations given to Silhouette.score

    How I test for this:
    1. Generate clustered data, perform fitting, perform predictions, calculate silhouette scores
    3. recalculate the Silhouette score for every observation in the data using the private KMeans._get_silhouette_distances method
    	The order the scores are calculated in here are consistent with the order the observations appear in `pred`
    4. Check each calculated score is identical (or approximately equal to what is returned in Silhouette.score() at the same index to ensure order is preserved

    """
    #Generate clustered data, train KMeans, cluster the data, calculate silhouette scores
    clusters, labels = make_clusters(scale=1.5,m=3)
    km = KMeans(k=3,random_state = 0)
    km.fit(clusters)
    pred = km.predict(clusters)
    sil = Silhouette()
    scores = sil.score(clusters, pred)

    def approx_equal(a, b, allowed_error = 0.0001):
        return abs(a - b) < allowed_error

    centroid_mat = km.get_centroids()
    for obs in range(len(pred)):
        mean_intra_distance,nearest_centroid_distance = sil._get_silhouette_distances(clusters,pred,sorted(list(set(pred))),centroid_mat,obs)
        score = (nearest_centroid_distance - mean_intra_distance) / (max(nearest_centroid_distance,mean_intra_distance))
        assert approx_equal(scores[obs],score), f"Re-calculated Silhouette Score is not the same as the model's Silhouette score for observation {obs}"


