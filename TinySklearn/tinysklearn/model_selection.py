import numpy as np
def train_test_split(X, y, test_size=0.2, random_state=None):
    assert X.shape[0] == y.shape[0], \
        "x和y的长度必须相同"
    assert 0.0 <= test_size <= 1.0, \
        "test_size必须在0-1之间"

    if random_state:
        np.random.seed(random_state)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_size)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
