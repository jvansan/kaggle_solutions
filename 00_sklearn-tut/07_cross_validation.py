# Cross-validation for parameter tuning, model selection, and feature selection
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read dataset
iris = load_iris()

# create X and y data
X, y = iris.data, iris.target

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# check classification of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(metrics.accuracy_score(y_test,knn.predict(X_test)))

# rerun different train/test split
# initial (state=4) = 97.36%
# next (state=1) = 100%
# final (state=3) = 94.74%

# Steps for k-fold Cross-validation
# choose a number for k partitions of equal size (folds)
# use fold 1 as testing and rest as training selection
# train model and calc testing accuracy_score
# report k times with different fold each time

# GOAL: select best tuning parametes for KNN on iris dataset
from sklearn.model_selection import cross_val_score
# 10-fold cross-validation with K=5 for KNN
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)

# use the average as an estimator for out-of-sample accuracy
print(scores.mean())

# search for an optimal value of k for KNN
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

# plot with matplotlib
import matplotlib.pyplot as plt

plt.plot(k_range, k_scores)

# GOAL: compare best KNN to logreg
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# 10-fold cross validation with logreg
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

# cross validation for feature selection
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# load data
df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# feature col list
feature_cols = ['TV', 'radio', 'newspaper']
# select X and y data
X = df[feature_cols]
y = df.sales

# 10-fold cross validation with all three features
ln = LinearRegression()
scores = cross_val_score(ln, X, y, cv=10, scoring='mean_squared_error')
print(scores)
# fix the signs of MSE scores
mse_scores = -scores
print(mse_scores)
# convert from mse to RMSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)
print(rmse_scores.mean())

# 10-fold cross val for two features
feature_cols = ['TV', 'radio']
X = df[feature_cols]
print(np.sqrt(-cross_val_score(ln,X,y,cv=10,scoring='mean_squared_error')).mean())
