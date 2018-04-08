# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:50:36 2018

@author: cml
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.rand(50,2)

for row in X:
    row[0] += 25
    row[1] -= 25

kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)

print("Index of the centroid:", kmeans2.labels_)
print("Centers: ", kmeans2.cluster_centers_)

print("Algorithm: ", kmeans2.algorithm)
print("No. clusters: ", kmeans2.n_clusters)

print("Sum of squares: ", kmeans2.inertia_)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

ax1.scatter(X[:,0], X[:,1], s=40, c=kmeans2.labels_, cmap=plt.cm.prism) 
ax1.set_title('K-Means Clustering Results with K=2')
ax1.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2)

kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X)

print("Sum of squares: ", kmeans3.inertia_)

ax2.scatter(X[:,0], X[:,1], s=40, c=kmeans3.labels_, cmap=plt.cm.prism) 
ax2.set_title('K-Means Clustering Results with K=3')
ax2.scatter(kmeans3.cluster_centers_[:,0], kmeans3.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);