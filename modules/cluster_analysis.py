import numpy as np
from typing import Any


def find_cluster_centroids(embeddings, max_k=10) -> Any:
    from sklearn.cluster import KMeans # Lazy import
    # from sklearn.metrics import silhouette_score # Lazy import if needed
    inertia = []
    cluster_centroids = []
    K = range(1, max_k+1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto') # Added n_init='auto' to suppress future warning
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
        cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})

    diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
    optimal_centroids = cluster_centroids[diffs.index(max(diffs)) + 1]['centroids']

    return optimal_centroids

def find_closest_centroid(centroids: list, normed_face_embedding) -> list:
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)
        
        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None