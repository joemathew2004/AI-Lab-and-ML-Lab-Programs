'''import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Randomly initialize centroids by selecting k data points from X
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):
            # Step 1: Assign clusters
            self.labels = self.assign_clusters(X)

            # Step 2: Update centroids
            new_centroids = self.update_centroids(X)

            # Step 3: Check for convergence (if centroids don't change)
            if np.all(new_centroids == self.centroids):
                break
            
            self.centroids = new_centroids

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # Compute distances to centroids
        return np.argmin(distances, axis=1)  # Assign clusters based on closest centroid

    def update_centroids(self, X):
        new_centroids = np.zeros((self.k, X.shape[1]))  # Initialize new centroids
        for i in range(self.k):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)  # Compute mean for the new centroid
        return new_centroids

    def predict(self, X):
        return self.assign_clusters(X)

    def plot_clusters(self, X):
        plt.figure(figsize=(8, 6))
        # Plot each cluster
        for i in range(self.k):
            plt.scatter(X[self.labels == i, 0], X[self.labels == i, 1], label=f'Cluster {i + 1}')
        
        # Plot the centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate random data for demonstration
    np.random.seed(0)  # For reproducibility
    X = np.random.rand(100, 2)  # 100 samples with 2 features

    # Create KMeans instance and fit the data
    kmeans = KMeans(k=3, max_iterations=100)
    kmeans.fit(X)

    # Print the final centroids and labels
    print("Final Centroids:\n", kmeans.centroids)
    print("Labels for each point:\n", kmeans.labels)

    # Plot the clusters
    kmeans.plot_clusters(X)'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  
y = iris.target  

def kmeans(X, K, max_iters=100, tol=1e-4):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    prev_centroids = np.zeros_like(centroids)
    labels = np.zeros(X.shape[0])
    
    for i in range(max_iters):
        for j in range(X.shape[0]):                    # Step 3: Assign points to the nearest centroid
            distances = np.linalg.norm(X[j] - centroids, axis=1)
            labels[j] = np.argmin(distances)
        
        for k in range(K):        # Step 4: Update centroids by calculating the mean of points in each cluster
            centroids[k] = np.mean(X[labels == k], axis=0)
        
        if np.all(np.abs(centroids - prev_centroids) < tol):  # Check for convergence (if centroids do not change)
            break
        prev_centroids = np.copy(centroids)
    
    return centroids, labels

K = 2  # Number of clusters
centroids, labels = kmeans(X, K)

plt.figure(figsize=(10, 6))

for i in range(K):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i+1}") # Plot each cluster with different colors
    
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')
plt.title('K-means Clustering on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()






