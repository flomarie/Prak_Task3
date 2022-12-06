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
        self.included = np.array([True] * self.estimators)
        rng = np.random.default_rng(seed=42)
        EPS1 = 1e-3
        EPS2 = 1e-3
        self.feat_sub = 
        for t in range(self.n_estimators):
            ind_sub = rng.integers(0, X.shape[0], size=X.shape[0])
            if self.feature_subsample_size is None:
                feature_sub = rng.integers(0, X.shape[1], size=max(1, int(X.shape[1] / 3)))
            else:
                feature_sub = rng.integers(0, X.shape[1], size=self.feature_subsample_size)
            X_sub = X[ind_sub, feature_sub]
            y_sub = y[ind_sub]
            self.b[t].fit(X_sub, y_sub)
            '''
            if 1 / X_sub.shape[0] * (self.b[t].predict(X_sub) - y_sub) ** 2 > EPS1:
                self.included[t] = False
            if not(X_val is None) and not(y_val is None):
                if 1 / X_val.shape[0] * (self.b[t].predict(X_val) - y_val) ** 2 > EPS2:
                    self.included[t] = False
            '''
		        
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
            prediction += self.b[t].predict(X)
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
        def mse(y_true, y_pred):
            return 1 / y_true.shape[0] * (y_true - y_pred) ** 2

        self.b = [DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
                  for i in range(self.n_estimators)]
        self.b[0].fit(X, y)
        f = self.b[0].predict(X)
        self.alphas = np.zeros(shape=self.n_estimators)
        for t in range(1, self.n_estimators):
            s = y - f
            self.b[t] = DecisionTreeRegressor(max_depth=self.max_depth,
                                              **self.trees_parameters)
            if self.feature_subsample_size is None:
                feature_sub = rng.integers(0, X.shape[1], size=max(1, int(X.shape[1] / 3)))
            else:
                feature_sub = rng.integers(0, X.shape[1], size=self.feature_subsample_size)
            self.b[t].fit(X, s)
            b_t = self.b[t].predict(X)
            alpha = minimize_scalar(lambda a: np.sum((f + a * self.learning_rate * b_t - y) ** 2))
            self.alphas[t] = alpha
            f += self.learning_rate * alpha * b_t

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
            y += self.learning_rate * self.alphas[t] * self.b[t].predict(X)
        return y

