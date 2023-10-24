# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:41:05 2023

@author: wiesbrock
"""

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import math
import seaborn as sns

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

crosscorr=pd.read_excel(r'C:\Users\wiesbrock\Desktop\wiesbrock\Desktop\correlations.xlsx')

path=r'Y:\File transfer\Michelle_transfer\IMARIS\IMARIS traces\20230612\20230612_testis_ATP_3.1\ATP on testis_c_ATP3.1_0_Pos_Statistics\ATP on testis_c_ATP3.1_0_Pos_Detailed.csv'
data=pd.read_csv(path,skiprows=(0,1,2))

x_pos=data['Track Position X Start']
y_pos=data['Track Position Y Start']

x_pos=np.array(x_pos)
y_pos=np.array(y_pos)

crosscorr=np.array(crosscorr)

plt.plot(x_pos,y_pos,'k.')

distance_matrix=np.zeros((len(x_pos),len(y_pos)))

for i in range(len(x_pos)):
    for j in range(len(x_pos)):
        distance_matrix[i,j]=distance(x_pos[i],y_pos[i],x_pos[j],y_pos[j])
        
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Annahme: x_pos und y_pos sind Ihre Positionsdaten als NumPy-Arrays
# Sie sollten die Anzahl der Cluster (Zell-Cluster) festlegen.
num_clusters = 4  # Ändern Sie dies entsprechend Ihren Daten

# Kombinieren Sie x_pos und y_pos zu einer Matrix
data_matrix = np.column_stack((x_pos, y_pos))

wcss = []  # Within-Cluster-Sum-of-Squares
max_clusters = 10  # Maximale Anzahl von Clustern, die getestet werden
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_matrix)
    wcss.append(kmeans.inertia_)

# Plot der Elbow-Methode
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.title('Elbow-Methode zur Ermittlung der optimalen Anzahl von Clustern')
plt.grid()
plt.show()

# Initialisieren und trainieren Sie das K-Means-Modell
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(data_matrix)

# Die Zuordnung jeder Position zu einem Cluster
cluster_labels = kmeans.labels_

# Die Zentren der Cluster
cluster_centers = kmeans.cluster_centers_

# Erstellen Sie einen Scatterplot, der die Cluster in verschiedenen Farben darstellt
plt.figure(figsize=(20, 20))
for cluster_id in range(num_clusters):
    cluster_x = data_matrix[cluster_labels == cluster_id, 0]
    cluster_y = data_matrix[cluster_labels == cluster_id, 1]
    plt.scatter(cluster_x, cluster_y, label=f'Cluster {cluster_id + 1}')
    plt.grid(visible=False)
    plt.yticks([])
    plt.xticks([])
    plt.ylabel([])
    plt.xlabel([])
    
for i in range(len(x_pos)):
    for j in range(len(x_pos)):
        if i != j:
            if crosscorr[i,j]>0.3 or crosscorr[j,i]>0.3:# Punkte nicht mit sich selbst verbinden
                plt.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], '-',c=plt.cm.jet((crosscorr[i,j]+crosscorr[j,i])/2),lw=0.3) 

#plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, marker='x', label='Cluster Center')
plt.xlabel('X-Position')
plt.ylabel('Y-Position')
sns.despine()
#plt.title('Zell-Cluster')
#plt.legend()
plt.show()


