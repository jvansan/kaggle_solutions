from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# read and load data
iris = load_iris()
X, y = iris.data, iris.target

# more efficient parameter tuning search using GridSearchCV
from sklearn.model_selection import GridSearchCV

# define parameter values to search
k_range = range(1,31)

# create grid of parameters to be searched
param_grid = dict(n_neighbors=k_range)

# instantiate and fit grid
# using n_jobs = -1 for parallelization
knn = KNeighborsClassifier(n_neighbors=5)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid.fit(X,y)
# view scores
grid.cv_results_

grid.cv_results_['mean_test_score']

# plot results
plt.plot(k_range, grid.cv_results_['mean_test_score'])

# examine best model
grid.best_score_
grid.best_params_
grid.best_estimator_

# searching multiple parameters simultaneously
#define parameter values
k_range = range(1,31)
weight_options = ['uniform', 'distance']

# create grid
param_grid = dict(n_neighbors=k_range, weights=weight_options)

#instantiate and fit grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid.fit(X,y)

# view scores
grid.cv_results_['mean_test_score']
grid.best_score_
grid.best_params_

# Using the best parameters
# make sure you train model using all data before using for out-of-train data
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)
# make prediction on out-of-sample data
knn.predict([[3,5,4,2]])

# gridsearchcv has a shortcut
# it automatically refits to all the data
grid.predict([[3,5,4,2]])

# reducing computational expense
from sklearn.model_selection import RandomizedSearchCV

#specificy param distribution rather than grid
param_dist = dict(n_neighbors=k_range, weights=weight_options)

# IMPORTANT: specify continous distribution for continous parameters

# n_iter controls # of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, n_jobs=-1)
rand.fit(X,y)
rand.cv_results_['mean_test_score']
rand.best_score_
rand.best_params_

# run RandomizedSearchCV 20 times with n_iter=10 and record best score
best_score = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, n_jobs=-1)
    rand.fit(X,y)
    best_score.append(round(rand.best_score_, 3))

print(best_score)
