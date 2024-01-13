import copy
import numpy as np
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data, k, seed=142):
        self._seed = seed
        np.random.seed(seed)
        n = len(data)
        self._data = copy.copy(data)  # data is of shape n x d
        self._k = k
        self._loss = float('inf')  # positive infinity
        self._clustered = False  # clustering is not yet done
        self._threshold = 1e-2
        self._cluster_ids = np.zeros(n, dtype=np.int32)  # shape is n x 1
        self._cluster_centers = np.zeros((k, self._data.shape[1]))  # shape is k x d
        self._distance_matrix = np.zeros((n, k))  # size is n x k

    def initialize(self, method="random"):
        """
        initial schem is choosing K random centers
        :param method: One of "random" or "kmeans_++"
        :return:
        """
        np.random.seed(self._seed)

        def random_center():
            return np.array([self._data[np.random.randint(0, len(self._data) - 1)] for _ in range(self._k)])

        def k_means_plus_plus():
            raise NotImplementedError

        self._cluster_centers = random_center() if method == "random" else k_means_plus_plus()

    @staticmethod
    def compute_distance(a, b):
        """
        Compute Euclidean distance between two vectors
        :param a: first vector
        :param b: second vector
        :return: Euclidean distance
        """
        return np.linalg.norm(a - b)

    def run_iteration(self):
        """
        Given K cluster centers, compute  for every point distance of K centers
        :return:
        """
        n = len(self._data)
        for i in range(n):
            point = self._data[i, :]
            for j in range(self._k):
                self._distance_matrix[i, j] = Kmeans.compute_distance(point, self._cluster_centers[j, :])

    def assign_centers(self):
        """
        Given distance matrix of shape n x k, assign cluster center for each point
        :return:
        """
        n = len(self._data)
        for i in range(n):
            self._cluster_ids[i] = np.argmin(self._distance_matrix[i, :])  # find cluster with lowest distance

    def compute_centers(self):
        """
        Given new cluster membership, compute new centers
        :return:
        """
        for j in range(self._k):
            # find all points with cluster membership j
            point_indices = np.where(self._cluster_ids == j)[0]
            self._cluster_centers[j] = np.mean(self._data[point_indices], axis=0)  # column means

    def compute_loss(self):
        """
        Computer MSE by computing sum of squared distance from cluster center to every point
        :return:
        """
        n = len(self._data)
        mse = 0
        for i in range(n):
            point = self._data[i, :]
            cluster_center = self._cluster_centers[self._cluster_ids[i]]
            # loss += np.linalg.norm(point - cluster_center) # work's well but not correct metric
            mse += np.mean(np.power(np.subtract(point, cluster_center), 2))

        return mse

    def cluster(self, enable_visualization=False):

        if self._clustered:
            print('Clustering Already done')
            return copy.copy(self._cluster_ids)

        # pick the initial K centers
        self.initialize()
        curr_loss = 2 ** 32
        # while loss has not stabilized
        count = 0
        while abs(curr_loss - self._loss) > self._threshold:

            self._loss = curr_loss
            # run iteration
            self.run_iteration()
            # assign cluster centers
            self.assign_centers()
            # compute new centers
            self.compute_centers()
            # compute loss
            curr_loss = self.compute_loss()
            count += 1
            print(f'Iteration {count}, mse:{curr_loss}, previous mse: {self._loss}')
            if enable_visualization:  # use this only for 2-dimensional data
                self.visualize_data_2d(False)

            if count > 1000:
                break

        self._clustered = True
        return copy.copy(self._cluster_ids)

    def visualize_data_2d(self, colors=True):
        n = int(len(self._data))
        if colors is False:
            cmap = plt.cm.get_cmap("tab20", self._k + 1)  # Adjust colormap and number of colors as needed
            colors = [cmap(cust_id) for cust_id in self._cluster_ids]  # Include K itself
        else:
            colors = ["red"] * (n // 2) + ["blue"] * (n // 2)

        plt.figure(figsize=(8, 6))
        plt.scatter(self._data[:, 0], self._data[:, 1], c=colors, s=50, alpha=0.8)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("Scatter Plot of Generated Data")
        plt.grid(True)
        plt.show()

    @staticmethod
    def get_test_data():
        """
        Generate data from two Gaussian distributions in 2 dimensions
        :return:
        """
        data = np.concatenate([
            np.random.normal(loc=[2, 4], scale=[1, 0.5], size=(50, 2)),  # First distribution
            np.random.normal(loc=[5, 2], scale=[1.5, 1], size=(50, 2))  # Second distribution
        ])

        return data


if __name__ == "__main__":
    K = 2
    obj = Kmeans(Kmeans.get_test_data(), K)
    obj.visualize_data_2d()
    cluster_ids = obj.cluster(True)
