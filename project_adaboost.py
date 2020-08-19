from time import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
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
    'n_estimators': [10, 25, 50, 75, 100, 125, 150, 175, 200, 300]
}

gs = GridSearchCV(
    AdaBoostClassifier(), grid_params, verbose=1, cv=5, n_jobs=-1
)
gs_results = gs.fit(X_train, y_train)
print("Adaboost Training done in %0.3fs\n" % (time() - t0))
print("Best estimator after cross validation:")
print("Decision Stumps - %d\n" % gs.best_estimator_.n_estimators)

# Testing
t0 = time()
y_pred = gs.predict(X_test)
print("Adaboost Testing done in %0.3fs\n" % (time() - t0))

# ROC Curve plot
probs = gs.predict_proba(X_test)
probs = probs[:, 1]
auc = metrics.roc_auc_score(y_test, probs)
print('AUC: %.2f\n' % auc)
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
utils.plot_roc_curve('Adaboost ROC', fpr, tpr)

# Confusion Matrix
print('Adaboost Confusion Matrix')
print('-------------------------')
print(confusion_matrix(y_test, y_pred))
