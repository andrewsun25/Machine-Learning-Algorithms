from scipy.io import loadmat
import numpy as np
from numpy.linalg import norm
from math import isclose
import matplotlib.pyplot as plt
from pprint import pprint
import my_utilities as my


class Point:

    points_total = 0
    points_existing = 0

    def __init__(self, position):
        self.id = Point.points_existing
        Point.points_existing += 1
        if Point.points_existing == Point.points_total:
            Point._reset_points_existing()
        self.position = position

    @classmethod
    def _reset_points_existing(cls):
        cls.points_existing = 0

    def set_centroid_id(self, centroids):
        distances = []
        for centroid in centroids:
            distances.append(norm(centroid.position - self.position))
        self.centroid_id = distances.index(min(distances))


class Centroid:

    centroids_total = 0
    centroids_existing = 0

    def __init__(self, all_points):
        self.id = Centroid.centroids_existing
        Centroid.centroids_existing += 1
        if Centroid.centroids_existing == Centroid.centroids_total:
            Centroid._reset_centroids_existing()
        m = len(all_points)
        self.position = all_points[np.random.randint(m)].position
        self.cluster = []

    @classmethod
    def _reset_centroids_existing(cls):
        cls.centroids_existing = 0

    def get_cluster_positions(self):
        if self.cluster:
            point_positions = []
            for point in self.cluster:
                point_positions.append(point.position)
            return point_positions
        else:
            print("In get_cluster_positions, self.cluster is not defined")

    def move_to_mean(self):
        if self.cluster:
            point_positions = np.array(self.get_cluster_positions())
            self.position = point_positions.mean(0)
        else:
            print("In move_to_mean, self.cluster is not defined")


def initialize_points(X):
    points = []
    for position in X:
        points.append(Point(position))
    return points

def assign_centroids_random(points, k):
    centroids = []
    for i in range(0, k):
        centroids.append(Centroid(points))
    return centroids

def compute_cost(points, centroids):
    costs = []
    for point in points:
        for centroid in centroids:
            if point.centroid_id == centroid.id:
                costs.append(norm(point.position - centroid.position) ** 2)
    if not costs:
        print("costs is empty")
    return np.mean(costs)

def run_k_means(points, centroids):
    for i, point in enumerate(points):
        points[i].set_centroid_id(centroids)
    for i, centroid in enumerate(centroids):
        for point in points:
            if centroid.id == point.centroid_id:
                centroids[i].cluster.append(point)
        if centroid.cluster:
            centroids[i].move_to_mean()
    return points, centroids

def repeat_k_means(X, k, num_iterations, rel_tol):
    points_initial = initialize_points(X)
    Centroid.centroids_total = k
    Point.points_total = len(points_initial)
    costs_ran_once_list = []
    centroids_ran_once_list = []
    points_ran_once_list = []
    # Finds N starting points with lowest cost after 1 iteration of k-means
    for iteration in range(0, num_iterations):
        centroids_initial = assign_centroids_random(points_initial, k)
        points_ran_once, centroids_ran_once = run_k_means(points_initial, centroids_initial)
        cost_ran_once = compute_cost(points_ran_once, centroids_ran_once)
        costs_ran_once_list.append(cost_ran_once)
        centroids_ran_once_list.append(centroids_ran_once)
        points_ran_once_list.append(points_ran_once)
    min_cost_index = costs_ran_once_list.index(min(costs_ran_once_list))
    ## Run K_means to convergence using starting points with lowest cost.
    points, centroids = points_ran_once_list[min_cost_index], centroids_ran_once_list[min_cost_index]
    current_cost = 1000
    previous_cost = 0
    while not isclose(current_cost, previous_cost, rel_tol=rel_tol):
        previous_cost = current_cost
        points, centroids = run_k_means(points, centroids)
        current_cost = compute_cost(points, centroids)
        print(current_cost)
    return points, centroids


### Execution
A = loadmat('data/bird_small.mat')['A']
# A_norm shape (128, 128, 3)
A_norm = A / 1.
# X shape (16384, 3)
X = np.reshape(A_norm, (A.shape[0] * A.shape[1], A.shape[2]))
points, centroids = repeat_k_means(X, 16, 5, rel_tol=0.5e-2)
print("FINISHED")
X_clustered = []
for i, pixel in enumerate(points):
    X_clustered.append(centroids[pixel.centroid_id].position)
X_clustered = np.rint(np.array(X_clustered)).astype('uint8')
A_clustered = np.reshape(X_clustered, (A.shape[0], A.shape[1], A.shape[2]))
plt.imshow(A)
plt.show()
plt.imshow(A_clustered)
plt.show()