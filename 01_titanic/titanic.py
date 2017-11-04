import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

# load data
df = pd.read_csv('train.csv', skipinitialspace=True)
test = pd.read_csv('test.csv', skipinitialspace=True)

# make 'Sex' numerical in training data
df = df.replace({'male':0,'female':1})
test = test.replace({'male':0,'female':1})

# try multiplying 'SibSp' and 'Parch'
df['SibAndPar'] = (df['SibSp'] * df['Parch'])
test['SibAndPar'] = (test['SibSp'] * test['Parch'])

# select features
imp = Imputer()
feature_list = ['Pclass', 'Sex', 'Age', 'SibAndPar', 'Fare']
X = df[feature_list]
y = df.Survived

# impute missing age values
imp = imp.fit(X)
X_imp = imp.transform(X)

# fit classifier
from sklearn.ensemble import GradientBoostingClassifier
clf3 = GradientBoostingClassifier()
print(cross_val_score(clf3, X_imp, y, cv=10, scoring='accuracy', n_jobs=-1).mean())
clf3.fit(X_imp, y)
# Run model on test data
X_test = test[feature_list]
X_imp_test = imp.transform(X_test)
results = clf3.predict(X_imp_test)

# store results in new dataframe
out = pd.DataFrame(test.PassengerId)
out['Survived'] = results
out.to_csv(path_or_buf='titanic_submission.csv', index=False)
