from ucimlrepo import fetch_ucirepo # type: ignore 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import helpers
  
# fetch dataset 
wholesale = fetch_ucirepo(id=292) 
 
dataset = wholesale.data.original.values.tolist() 
X = wholesale.data.features 
y = wholesale.data.targets

FRESH = 2
MILK = 3
GROCERY = 4
FROZEN = 5
DETERGENTS_PAPER = 6
DELICATESSEN = 7

parsedElements = [[element[FRESH], element[GROCERY], element[DETERGENTS_PAPER]] for element in dataset]

initialCentroids = helpers.getInitialCentroids(parsedElements)
clusters = helpers.applyKMeans(parsedElements, initialCentroids)

PARSED_FRESH = 0
PARSED_GROCERY = 1
PARSED_DETERGENTS_PAPER = 2

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for cluster in clusters:
    elementsColors = np.random.rand(len(clusters))
    elementsColors = np.random.rand(len(clusters))
    
    x = [element[PARSED_FRESH] for element in cluster]
    y = [element[PARSED_GROCERY] for element in cluster]
    z = [element[PARSED_DETERGENTS_PAPER] for element in cluster]

    computedCentroid = helpers.computeCentroid(cluster)

    ax.scatter(x, y, z, c=elementsColors)
    ax.scatter(
        computedCentroid[PARSED_FRESH], 
        computedCentroid[PARSED_GROCERY], 
        computedCentroid[PARSED_DETERGENTS_PAPER], 
        c = "black", marker="+",
    )
    ax.text(
        computedCentroid[PARSED_FRESH], 
        computedCentroid[PARSED_GROCERY], 
        computedCentroid[PARSED_DETERGENTS_PAPER], 
        str(len(cluster)),
        fontsize = 10
    )

ax.set_xlabel("FRESH")
ax.set_ylabel("GROCERY")
ax.set_zlabel("DETERGENTS_PAPER")

plt.legend()
plt.show()