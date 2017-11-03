# model evaluation necessary to determine how well model will
# work on out-of-sample data
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)

# checkout df
pima.head()
# QUESTION: can we predict the diabetes status given health measurements

#define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X=pima[feature_cols]
y = pima.label

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train logistic regression on training sets
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make class prediciton for train sets
y_pred_class = logreg.predict(X_test)

# classification: precent accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

# Null accuracy: accuracy_score that can always be predicted by most frequent class
# examine class distribution
y_test.value_counts()
#calculate percentage of ones
y_test.mean()
#calc percentrage of zeros
1-y_test.mean()

# calculate null accuracy (for binary classification as 0/1)
max(y_test.mean(), 1-y_test.mean())

# calculate null accuracy (for multi-classification problems)
y_test.value_counts().head(1) / len(y_test)


# Compare true and predicted response values
#print first 25 values
print('True:', y_test.values[0:25])
print('Pred:', y_pred_class[0:25])
# See that usually makes mistakes for 1's (not 0's)
# Look at the Confusion matrix
# IMPORTANT: first arguement is true values, second is predicted
print(metrics.confusion_matrix(y_test, y_pred_class))

#  |     n=192  | Pred 0  | Pred 1 |
#  |------------|---------|--------|
#  | actual = 0 |  118    |   12   |
#  |------------|---------|--------|
#  | actual = 1 |   47    |   15   |

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

# classification accuracy: how often correct
print ((TP+TN) / (TN+TP+FP+FN))
print(metrics.accuracy_score(y_test, y_pred_class))

# classification error: how often incorrect
print((FP+FN)/(TP+TN+FP+FN))
print(1-metrics.accuracy_score(y_test, y_pred_class))

# sensitivity: when the actualy value is +, how often is this predicted
# True rate or RECALL
print(TP / (TP + FN))
print(metrics.recall_score(y_test, y_pred_class))

# specificity: when the actual value i -, how often is this predicted
# PRECISION
print(TN/(TN + FP))

# False positive rate: when actual -, how often is it predicted?
print(FP/(TN+FP))

# Precision (True Negative Rate): when actual +, how often is it predicted
print(TP/(TP+FP))
print(metrics.precision_score(y_test, y_pred_class))

# Adjusting classification threshold
# print first 10 pred responses
logreg.predict(X_test)[0:10]

# print fist 10 predicted probabilities of class membership
logreg.predict_proba(X_test)[0:10]

# print first 10 predicted probs for class 1
logreg.predict_proba(X_test)[0:10, 1]
# store all these values
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# plot these data
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# histogram of predicted probs
plt.hist(y_pred_prob, bins=8)

# Decrease threshold for prediciting diabetes
# This isn't working for some reason...
from sklearn.preprocessing import binarize

y_pred_class = binarize(y_pred_prob, threshold=0.3)

# print the first 10
print(y_pred_prob[0:10])
print(y_pred_class[0:10])

# ROC Curves and Area Under the Curve (AUC)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
# ROC curve can help you choose a threshold that balances sensitivity and specificity
def evaluate_threshold(threshold):
    print('sensitivity:', tpr[thresholds > threshold][-1])
    print('specificity:', 1-fpr[thresholds > threshold][-1])

evaluate_threshold(0.5)
evaluate_threshold(0.3)
# AUC want ROC that hugs top left corner
# AUC often used as single number summary of how good model is

print(metrics.roc_auc_score(y_test, y_pred_prob))

# calculate cross validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc', n_jobs=-1).mean()
