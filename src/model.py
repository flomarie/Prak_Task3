from cmath import exp
from ensembles import *
import preprocessing


class Params_RF:
    n_estimators = 100
    max_depth = None
    feature_subsample_size = None

class Params_GB:
    n_estimators = 100
    max_depth = 3
    feature_subsample_size = 1
    learning_rate = 0.1

gradient_boosting = None
random_forest = None

def train_gb(params, path):
    global gradient_boosting
    gradient_boosting = GradientBoostingMSE(n_estimators=params.n_estimators,
                                            learning_rate=params.learning_rate,
                                            max_depth=params.max_depth,
                                            feature_subsample_size=params.feature_subsample_size)
    X, y = preprocessing.prepare_data_train(path)
    gradient_boosting.fit(X, y)

def test_gb(path):
    global gradient_boosting
    X = preprocessing.prepare_data_test(path)
    return gradient_boosting.predict(X)

def train_rf(params, path):
    global random_forest
    random_forest = RandomForestMSE(n_estimators=params.n_estimators,
                                            max_depth=params.max_depth,
                                            feature_subsample_size=params.feature_subsample_size)
    X, y = preprocessing.prepare_data_train(path)
    random_forest.fit(X, y)

def test_rf(path):
    global random_forest
    X = preprocessing.prepare_data_test(path)
    return random_forest.predict(X)
