
import numpy as np
from sklearn.neighbors import NearestNeighbors


class CounterfactualSMOTE:

    def __init__(self, k_neighbors=3, n_iter=100, random_state=22, opt_steps=10, strategy="majority", origin="majority"):
        self.k_neighbors = k_neighbors
        self.n_iter = n_iter
        self.random_state = random_state
        self.minority_neighbors_model = None
        self.majority_neighbors_model = None
        self.opt_steps = opt_steps
        self.strategy = strategy
        self.origin = origin

    def _fit_neighbors_models(self, X_minority, X_majority):

        self.minority_neighbors_model = NearestNeighbors(
            n_neighbors=self.k_neighbors).fit(X_minority)
        self.majority_neighbors_model = NearestNeighbors(
            n_neighbors=self.k_neighbors).fit(X_majority)

    def _binary_search(self, minority_sample, majority_neighbor):

        if self.origin == "minority":
            direction = minority_sample - majority_neighbor
        else:
            direction = majority_neighbor - minority_sample

        low = 0.0
        high = 1.0
        last_class = "majority"
        found_minority = False
        last_valid = np.array(0)

        for _ in range(self.opt_steps):

            mid = 0.5 * (low + high)

            current_point = majority_neighbor + mid * direction

            distances_majority, _ = self.majority_neighbors_model.kneighbors(
                current_point.reshape(1, -1), self.k_neighbors
            )
            distances_minority, _ = self.minority_neighbors_model.kneighbors(
                current_point.reshape(1, -1), self.k_neighbors
            )

            distances = np.zeros(self.k_neighbors * 2)

            distances[: self.k_neighbors] = distances_majority
            distances[self.k_neighbors:] = distances_minority

            boundary = self.k_neighbors // 2 + 1 if self.strategy == "majority" else 1

            if np.sum(np.isin(np.argsort(distances)[: self.k_neighbors], range(self.k_neighbors))) >= boundary:

                last = "majority"
                low = mid

            else:
                last = "minority"
                high = mid
                found_minority = True
                last_valid = current_point

        if last == "majority" and found_minority:
            return last_valid, "minority"

        else:
            return current_point, last_class

    def fit_resample(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)

        unique_labels, label_counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(label_counts)]

        X_minority = X[y == minority_label]
        X_majority = X[y != minority_label]
        self._fit_neighbors_models(X_minority, X_majority)

        X_synthetic = []
        y_synthetic = []

        for i in range(self.n_iter):

            np.random.seed(i)

            for i in range(len(X_minority)):
                if len(X_minority) + len(X_synthetic) > len(X_majority):
                    break
                majority_neighbor = X_majority[np.random.choice(
                    len(X_majority))]
                synthetic_sample, last = self._binary_search(
                    X_minority[i], majority_neighbor)

                if last == "majority":
                    continue
                else:

                    X_synthetic.append(synthetic_sample)
                    y_synthetic.append(minority_label)

        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)

        del self.minority_neighbors_model,  self.majority_neighbors_model

        if len(X_synthetic) > 0:
            X_combined = np.vstack((X, X_synthetic))
            y_combined = np.hstack((y, y_synthetic))

            return X_combined, y_combined

        else:
            return X, y
