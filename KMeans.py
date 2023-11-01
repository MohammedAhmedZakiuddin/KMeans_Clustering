# Mohammed Zakiuddin
# 1001675091 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class KMeans:
    # Initialize the class with number of clusters = 2. 
    def __init__(self, K=2, max_iters=100, plot_step=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_step = plot_step
        
        # Initialize the clusters and centroids to empty lists. 
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    # Fit the data to the KMeans algorithm.
    def fit(self, X, pdf=None):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Loops through the data and clusters the data points into the clusters.
        for i in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_step and (i == 0 or i == self.max_iters//2 or i == self.max_iters-1):
                self.plot(i+1, pdf=pdf)

            centroids_old = self.centroids

            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters) # Method returns the cluster labels for each data point based on their assigned centroid.
    
    # Method to get the cluster labels for each data point based on their assigned centroid.
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    # Method to create the clusters.
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    # Method to find the closest centroid to the data point.
    def _closest_centroid(self, sample, centroids):
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    # Method to get the centroids.
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    # Method to check if the centroids have converged.
    def _is_converged(self, centroids_old, centroids):
        distances = [self._euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    # Method to calculate the euclidean distance between two points.
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Method to plot the clusters and centroids.
    def plot(self, round_num, pdf=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.title("K={0} - Round {1}".format(self.K, round_num))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        # Save the figure to the PDF file if it is provided
        if pdf is not None:
            with PdfPages(pdf) as pdf_pages:
                pdf_pages.savefig(fig)

        plt.show()

    # Method to run the KMeans algorithm.
    def KMeans(self, datasetFile, K=2, pdf=None):
        # Load data from file
        X = np.loadtxt(datasetFile, delimiter=",")

        # Run KMeans algorithm
        kmeans = KMeans(K=K, plot_step=True)
        labels = kmeans.fit(X)

        # Save plot to PDF
        if pdf is not None:
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=labels)
            plt.title(f"KMeans Clustering with K={K}")
            pdf.savefig()

        return labels

# Create an instance of the class
# Create an instance of the class
my_clustering_instance = KMeans()

# Open the PDF file for writing
with PdfPages("output.pdf") as pdf_pages:
    # Test with default K=2
    labels = my_clustering_instance.KMeans("ClusteringData.txt", pdf=pdf_pages)
    print("Labels for K=2: ", labels)

    # Test with K=4
    labels = my_clustering_instance.KMeans("ClusteringData.txt", K=4, pdf=pdf_pages)
    print("Labels for K=4: ", labels)

    # Test with K=8
    labels = my_clustering_instance.KMeans("ClusteringData.txt", K=8, pdf=pdf_pages)
    print("Labels for K=8: ", labels)
