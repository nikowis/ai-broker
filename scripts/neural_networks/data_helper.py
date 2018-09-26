import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler


def train_test_split(X, y, scale=True):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    if scale:
        expanded_y_train = np.expand_dims(y_train, axis=1)
        expanded_y_test = np.expand_dims(y_test, axis=1)
        scaler = MinMaxScaler()
        train_data = np.append(X_train, expanded_y_train, axis=1)
        test_data = np.append(X_test, expanded_y_test, axis=1)
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        X_size = X_train.shape[1]

        X_train = train_data[:, :X_size]
        y_train = train_data[:, X_size]
        X_test = test_data[:, :X_size]
        y_test = test_data[:, X_size]
    return X_train, X_test, y_train, y_test
