# Importing the libraries
from time import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import utils

# Load data
data = np.loadtxt("abalone.csv", delimiter=",")
a = data[:, 1:9]
b = data[:, 0:1]


def train(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the Kernel SVM model as a warm up as it seems like first run always takes time
    grid_params = {
        'C': [1, 5, 10]
    }

    gs = GridSearchCV(
        SVC(kernel='rbf', probability=True), grid_params, cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)

    # Training the Kernel SVM model
    t0 = time()
    grid_params = {
        'C': [1, 5, 10]
    }

    gs = GridSearchCV(
        SVC(kernel='rbf', probability=True), grid_params, cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print("SVM Training done in %0.4fs" % (time() - t0))

    # Training AdaBoost model
    t0 = time()
    grid_params = {
        'n_estimators': [10, 25, 50]
    }

    gs = GridSearchCV(
        AdaBoostClassifier(), grid_params, cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print("Adaboost Training done in %0.4fs" % (time() - t0))

    # Training KNN model
    t0 = time()
    grid_params = {
        'n_neighbors': [3, 5, 7],
        'metric': ['euclidean', 'manhattan']
    }

    gs = GridSearchCV(
        KNeighborsClassifier(), grid_params, cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print("KNN Training done in %0.4fs" % (time() - t0))


for i in (20, 200, 2000):
    print("Complexities with %d data points" % i)
    n = a[0:i, :]
    m = b[0:i, :].ravel()
    train(n, m)
    print()
