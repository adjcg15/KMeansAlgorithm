from ucimlrepo import fetch_ucirepo # type: ignore 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import helpers
  
# fetch dataset 
wholesale = fetch_ucirepo(id=292) 
 
# Se obtienen los datos de la base de datos en forma de lista
dataset = wholesale.data.original.values.tolist()

# Se especifican las posiciones de los atributos en la base de datos
FRESH = 2
MILK = 3
GROCERY = 4
FROZEN = 5
DETERGENTS_PAPER = 6
DELICATESSEN = 7

# Se mapean los 3 atributos elegidos para cada registro en la base de datos
parsedElements = [[element[FRESH], element[GROCERY], element[DETERGENTS_PAPER]] for element in dataset]

# Se obtienen los centroides iniciales
initialCentroids = helpers.getInitialCentroids(parsedElements)

# Se llama a la función de applyKMeans para que clasifique los elementos entre los k clusters definidos
clusters = helpers.applyKMeans(parsedElements, initialCentroids)

PARSED_FRESH = 0
PARSED_GROCERY = 1
PARSED_DETERGENTS_PAPER = 2

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Se recorren los clusters obtenidos, donde para cada uno, se grafican los valores de x, y, z, configurando
# propiedades como los colores, y las formas de los centroides, así como también se muestran etiquetas de
# la cantidad de elementos que pertenecen a un cluster en particular.
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

# Se asignan leyendas a los ejes x, y, z
ax.set_xlabel("FRESH")
ax.set_ylabel("GROCERY")
ax.set_zlabel("DETERGENTS_PAPER")

# Se muestra la gráfica en 3D
plt.legend()
plt.show()