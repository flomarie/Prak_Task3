import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.b = [DecisionTreeRegressor(criterion='squared_error',
                                        max_depth=self.max_depth,
                                        **self.trees_parameters)
                  for t in range(self.n_estimators)]
        rng = np.random.default_rng(seed=42)
        feature_subsample_size = self.feature_subsample_size
        if self.feature_subsample_size is None:
            feature_subsample_size = max(1, int(X.shape[1] / 3))
        self.feat_sub = np.zeros(shape=(self.n_estimators, feature_subsample_size), dtype=int)
        for t in range(self.n_estimators):
            ind_sub = rng.integers(0, X.shape[0], size=X.shape[0])
            feature_sub = rng.integers(0, X.shape[1], feature_subsample_size)
            X_sub = X[ind_sub][:, feature_sub]
            y_sub = y[ind_sub]
            self.b[t].fit(X_sub, y_sub)
            self.feat_sub[t] = feature_sub
		        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction = np.zeros(shape=X.shape[0])
        for t in range(self.n_estimators):
            prediction += self.b[t].predict(X[:, self.feat_sub[t]])
        prediction *= 1 / self.n_estimators
        return prediction



class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        self.b = [DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
                  for i in range(self.n_estimators)]
        self.alphas = np.ones(shape=self.n_estimators)
        rng = np.random.default_rng(seed=42)
        feature_subsample_size = self.feature_subsample_size
        if feature_subsample_size is None:
            feature_subsample_size = max(1, int(X.shape[1] / 3))
        self.feat_sub = np.zeros(shape=(self.n_estimators, feature_subsample_size), dtype=int)
        f = np.zeros_like(y)
        for t in range(0, self.n_estimators):
            s = y - f
            feature_sub = rng.integers(0, X.shape[1], size=feature_subsample_size)
            self.b[t].fit(X[:, feature_sub], s)
            self.feat_sub[t] = feature_sub
            b_t = self.b[t].predict(X[:, feature_sub])
            alpha = minimize_scalar(lambda a: np.sum((f + a * self.learning_rate * b_t - y) ** 2))
            self.alphas[t] = alpha.x
            f += self.learning_rate * alpha.x * b_t

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y = np.zeros(shape=X.shape[0])
        for t in range(self.n_estimators):
            y += self.learning_rate * self.alphas[t] * self.b[t].predict(X[:, self.feat_sub[t]])
        return y
