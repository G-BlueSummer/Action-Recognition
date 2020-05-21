import os
import numpy as np
from joblib import dump
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

X = np.load(os.path.join('features', 'X_samples.npy'))
y = np.load(os.path.join('features', 'y_samples.npy'))

le = LabelEncoder()
y = le.fit_transform(y)
dump(le, os.path.join('model', 'label.joblib'))

print(X.shape)
pca = PCA(n_components=50, whiten=True)
X = pca.fit_transform(X)
print(X.shape)
dump(pca, os.path.join('model', 'pca.joblib'))

cf = AdaBoostClassifier()
cf.fit(X, y)
dump(cf, os.path.join('model', 'classifier.joblib'))

print(confusion_matrix(y, cf.predict(X)))