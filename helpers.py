import numpy as np
import random

def getInitialCentroids(dataset, k = 3):
    return random.sample(dataset, k)

def applyKMeans(dataset, initialCentroids):
    clusters = classifyElementsInClusters(dataset, initialCentroids)

    computedCentroids = []
    for cluster in clusters:
        clusterComputedCentroid = computeCentroid(cluster)
        computedCentroids.append(clusterComputedCentroid)

    if compareCentroidSets(initialCentroids, computedCentroids):
        return clusters
    else:
        return applyKMeans(dataset, computedCentroids)
    
def classifyElementsInClusters(dataset, centroids):
    clusters = [[] for _ in centroids]

    for element in dataset:
        nearestCentroidIndex = getNearestCentroidIndex(element, centroids)
        clusters[nearestCentroidIndex].append(element)

    return clusters

def getNearestCentroidIndex(element, centroids):
    distancesToCentroids = []
    for centroid in centroids:
        distance = np.sqrt(np.sum((np.array(element) - np.array(centroid)) ** 2))
        distancesToCentroids.append(distance)
    
    nearestCentroidIndex = 0
    minimumDistance = distancesToCentroids[0]
    for i in range(1, len(distancesToCentroids)):
        if distancesToCentroids[i] < minimumDistance:
            minimumDistance = distancesToCentroids[i]
            nearestCentroidIndex = i

    return nearestCentroidIndex
    
def computeCentroid(cluster):
    elementPrototype = cluster[0]
    centroid = [0 for _ in elementPrototype]

    for element in cluster:
        for i in range(0, len(element)):
            centroid[i] += element[i]

    for i in range(0, len(centroid)):
        centroid[i] /= len(cluster)
        centroid[i] = round(centroid[i], 1)

    return centroid

def compareCentroidSets(initialCentroids, finalCentroids):
    return all(initialCentroids[i] == finalCentroids[i] for i in range(0, len(initialCentroids)))