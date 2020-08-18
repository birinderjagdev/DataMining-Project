# Importing the libraries
from time import time

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Training the Kernel SVM model on the Training set
t0 = time()
grid_params = {
    'C': [10,1e2, 1e3, 5e3, 1e4]
}

gs = GridSearchCV(
    SVC(kernel='rbf',probability=True), grid_params, verbose=1, cv=5, n_jobs=-1
)
gs_results = gs.fit(X_train, y_train)
print("SVM Training done in %0.3fs\n" % (time() - t0))
print("Best estimator after cross validation:")
print("C-support - %d\n" % gs.best_estimator_.C)

# Testing
t0 = time()
y_pred = gs.predict(X_test)
print("SVM Testing done in %0.3fs\n" % (time() - t0))

# ROC Curve plot
probs = gs.predict_proba(X_test)
probs = probs[:, 1]
auc = metrics.roc_auc_score(y_test, probs)
print('AUC: %.2f\n' % auc)
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
utils.plot_roc_curve('SVM ROC', fpr, tpr)

# Confusion Matrix
print('SVM Confusion Matrix')
print('-------------------------')
print(confusion_matrix(y_test, y_pred))
