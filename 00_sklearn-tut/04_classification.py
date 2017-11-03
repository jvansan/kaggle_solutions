import numpy as np
#import iris data set
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
# print(iris.data)
# print(iris.target_names)
# print(iris.target)
# print(iris.data.shape)
# print(iris.target.shape)

# store feature matrix as "X"
X = iris.data
# store response vector as "y"
y = iris.target

# K-nearest neighbours (KNN) classification
# Step 1: Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Instantiant the estimator
knn = KNeighborsClassifier(n_neighbors=1)
# print(knn)

# Step 3: Fit the model to training data
knn.fit(X,y)

# Step 4: Make predictions
knn.predict([[3,5,4,2]])
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)

# Using different values of K
# model tuning
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
knn.predict(X_new)

# Try using a different classification model
# import the classifier
from sklearn.linear_model import LogisticRegression

# instantiate
logreg = LogisticRegression()

# fit
logreg.fit(X,y)
# predict
logreg.predict(X_new)
