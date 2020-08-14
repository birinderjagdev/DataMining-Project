from time import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils

# Load data
data = np.loadtxt("abalone.csv", delimiter=",")
X = data[:, 1:9]
y = data[:, 0:1].ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Training
t0 = time()
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                algorithm="SAMME",
                                n_estimators=200)

classifier.fit(X_train, y_train)
print("Adaboost Training done in %0.3fs\n" % (time() - t0))

# Testing
t0 = time()
y_pred = classifier.predict(X_test)
print("Adaboost Testing done in %0.3fs\n" % (time() - t0))

# ROC Curve plot
probs = classifier.predict_proba(X_test)
probs = probs[:, 1]
auc = metrics.roc_auc_score(y_test, probs)
print('AUC: %.2f\n' % auc)
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
utils.plot_roc_curve('Adaboost ROC',fpr, tpr)

# Confusion Matrix
print('Adaboost Confusion Matrix')
print('-------------------------')
print(confusion_matrix(y_test, y_pred))
