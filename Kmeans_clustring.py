import numpy as np
import matplotlib.pyplot as plt

def k_means_clustering(data, k, epsilon=0.0001, max_iterations=1000):
    centroids = data[:k]

    for iter in range(max_iterations):
        # Assign points to clusters
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(point)

        # Update centroids
        new_centroids = [np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)]

        # Check for convergence
        if np.linalg.norm(np.array(new_centroids) - np.array(centroids)) < epsilon:
            break

        centroids = new_centroids

    return clusters, centroids
