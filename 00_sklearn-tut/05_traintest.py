# PROCEDURE 1: Train and test on entire dataset
# read in iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

X, y = iris.data, iris.target

# Apply LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit
logreg.fit(X,y)
# predict response for observations of X
y_pred = logreg.predict(X)
len(y_pred)

# classification accurarcy
# compute classification accuracy from logreg
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))
# training accuracy of 96%

# KNN=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))
# slightly better than logreg 96.7%

# KNN=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))
# Here we get 100% accuracy!
# But this doesn't mean this model is the best!
# KNN has memorized the training set!
# training and testing your data on same model is BAD

# TRAIN TEST SPLIT

# step 1, split X and y, into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# X_train.shape
# X_test.shape
# y_train.shape
# y_test.shape

# step 2, train model on training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# step 3: make prediction
y_pred = logreg.predict(X_test)

# compate actual response on testing set
print(metrics.accuracy_score(y_test, y_pred))
# 95% training accuracy

# Now try with KNN = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
# Here still get 96.7%

# and now KNN = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# Can we locate an ever better value for K?
k_range = range(1,25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# import matplotlib
import matplotlib.pyplot as plt
plt.xlabel('K value')
plt.ylabel('Testing Accuracy')
plt.plot(k_range, scores)

plt.show()

# Try making a prediction on out-of-sample data
# Choose 11 b/c in the middle or range above
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X,y)
knn.predict([[3,5,4,2]])

# Should use k-fold cross validation to fix variance bias
