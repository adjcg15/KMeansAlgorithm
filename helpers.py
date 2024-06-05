import numpy as np
import random

# Función que regresa k centroides aleatorios de la base de datos wholesale
def getInitialCentroids(dataset, k = 3):
    return random.sample(dataset, k)

# Función recursiva que tiene toda la lógica para realizar la clasificación y obtener k clusters
def applyKMeans(dataset, initialCentroids):
    # Se clasifican los datos de la base de datos en clusters, colocándolos donde se encuentre el
    # centroide más cercano
    clusters = classifyElementsInClusters(dataset, initialCentroids)

    computedCentroids = []

    # Se recorren los clusters obtenidos en la clasificación
    for cluster in clusters:
        # Para cada cluster se calcula el nuevo centroide
        clusterComputedCentroid = computeCentroid(cluster)
        computedCentroids.append(clusterComputedCentroid)

    # Se comparan los centroides anteriores con los que se acaban de calcular, en caso de que sean
    # iguales, significa que no se reubicaron en la iteración, por lo que la calificación ya se 
    # encuentra óptima, en caso de que sean diferentes, entonces se vuelve a llamar a la función con
    # los centroides actuales (recursividad)
    if compareCentroidSets(initialCentroids, computedCentroids):
        return clusters
    else:
        return applyKMeans(dataset, computedCentroids)


# Función para clasificar los elementos en clusters
def classifyElementsInClusters(dataset, centroids):
    clusters = [[] for _ in centroids]

    # Se recorren los elementos de la base de datos
    for element in dataset:
        # Se obtiene el índice del centroide más cercano al elemento
        nearestCentroidIndex = getNearestCentroidIndex(element, centroids)
        
        # Se agrega el elemento en la posición del cluster más cercano
        clusters[nearestCentroidIndex].append(element)

    return clusters

# Función para obtener el índice del centroide más cercano
def getNearestCentroidIndex(element, centroids):
    distancesToCentroids = []

    # Se recorren los centroides actuales y se calcula la distancia entre el elemento y los 
    # centroides
    for centroid in centroids:
        distance = np.sqrt(np.sum((np.array(element) - np.array(centroid)) ** 2))
        distancesToCentroids.append(distance)
    
    nearestCentroidIndex = 0
    minimumDistance = distancesToCentroids[0]

    # Se evaluan las distancias obteniedas entre los k centroides y el elemento, y se obtiene 
    # el índice del centroide con la distancia más corta
    for i in range(1, len(distancesToCentroids)):
        if distancesToCentroids[i] < minimumDistance:
            minimumDistance = distancesToCentroids[i]
            nearestCentroidIndex = i

    return nearestCentroidIndex

# Función que calcula el centroide de un cluster, lo realiza usando la media de los puntos    
def computeCentroid(cluster):
    elementPrototype = cluster[0]
    centroid = [0 for _ in elementPrototype]

    # Se recorren los elementos del cluster
    for element in cluster:
        
        # Para cada elemento, se recorren sus posiciones (x, y, z), y se suman para posteriormente
        # calcular la media
        for i in range(0, len(element)):
            centroid[i] += element[i]

    # Se recorren las posiciones del nuevo centroide, donde se calcula la media dividiendo el valor
    # de cada posición entre el total elementos en el cluster y se redondea para tener una precisión 
    # de una décima
    for i in range(0, len(centroid)):
        centroid[i] /= len(cluster)
        centroid[i] = round(centroid[i], 1)

    return centroid

# Función que compara las posiciones de dos centroides, esto para saber si después de otra iteración
# el centroide ya no cambió
def compareCentroidSets(initialCentroids, finalCentroids):
    return all(initialCentroids[i] == finalCentroids[i] for i in range(0, len(initialCentroids)))