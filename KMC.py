import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class KMC:

    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centroids = []
        self.distances = []
        self.points = []

    def init_centroids(self):
        # Randomly choosing Centroids
        i = np.random.choice(len(self.x), self.k, replace=False)
        self.centroids = self.x[i, :]

    def distance(self, distance):
        # finding the distance between centroids and all the data points
        self.distances = cdist(self.x, self.centroids, distance)

    def point(self):
        # Centroid with the minimum Distance
        self.points = np.array([np.argmin(i) for i in self.distances])

    def k_means(self, distance, no_of_iterations):
        # distance stands for the distance to be used calculated, usually euclidean distance
        self.init_centroids()
        self.distance(distance)
        self.point()

        # Repeating the above steps for a number a iterations

        for _ in range(no_of_iterations):
            centroids = []
            for j in range(self.k):
                # Updating Centroids by taking mean of Cluster it belongs to
                temp_cent = self.x[self.points == j].mean(axis=0)
                centroids.append(temp_cent)

            self.centroids = np.vstack(centroids)  # Updated Centroids

            self.distance(distance)
            self.point()

        return self.points

    def visualize(self):
        label = self.points
        u_labels = np.unique(label)
        for i in u_labels:
            plt.scatter(self.x[label == i, 0], self.x[label == i, 1], label=i)
        plt.legend()
        plt.show()
