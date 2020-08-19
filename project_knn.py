from time import time

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import utils

# Load data
data = np.loadtxt("abalone.csv", delimiter=",")
X = data[:, 1:9]
y = data[:, 0:1].ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training
t0 = time()
grid_params = {
    'n_neighbors': [3, 5, 7, 11, 13, 15, 17, 19, 21, 23, 25],
    'metric': ['euclidean', 'manhattan']
}

gs = GridSearchCV(
    KNeighborsClassifier(), grid_params, verbose=1, cv=5, n_jobs=-1
)
gs_results = gs.fit(X_train, y_train)
print("KNN Training done in %0.3fs\n" % (time() - t0))
print("Best estimator after cross validation:")
print("Metric - %s\nK - %d\n" % (gs.best_estimator_.metric,gs.best_estimator_.n_neighbors))

# Testing
t0 = time()
y_pred = gs.predict(X_test)
print("KNN Testing done in %0.3fs\n" % (time() - t0))

# ROC Curve plot
probs = gs.predict_proba(X_test)
probs = probs[:, 1]
auc = metrics.roc_auc_score(y_test, probs)
print('AUC: %.2f\n' % auc)
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
utils.plot_roc_curve('KNN ROC', fpr, tpr)

# Confusion Matrix
print('KNN Confusion Matrix')
print('-------------------------')
print(confusion_matrix(y_test, y_pred))
