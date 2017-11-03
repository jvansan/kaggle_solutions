# import pandas
import pandas as pd

# read CSV directly from URL and save results
df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
df.head(3)
# check shape of dataframe
df.shape

# Visualization data using seaborn
import seaborn as sns

# visualize the relationship the features and response using scatterplots
sns.pairplot(df, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')

# Linear regression
# create a python list of feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of the original DF
X = df[feature_cols]

## Equivalent is
# X = df[['TV', 'radio', 'newspaper']]
# X.shape
# type(X)
X.head()
y = df['sales']

y.head()
# type(y)
# y.shape

# Split X and y intro train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)

# Linar regression in sklearn
# import and instantiate linear regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

# fit the model to training data
linreg.fit(X_train, y_train)
# interprest model coeffiecents
print(linreg.intercept_)
print(linreg.coef_)
# pair feature names with coeffiecents
list(zip(feature_cols, linreg.coef_))
# making predictions
y_pred = linreg.predict(X_test)

# we need a new evaluation metric
# define true and predicted values
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]

# calculate the MAE by hand
print('MAE = %f' %((10+0+20+10)/4))
#or use sklean
from sklearn import metrics
print('MAE = %f' %metrics.mean_absolute_error(true,pred))

# now calc MSE
print('MSE = %f' %((10**2+0**2+20**2+10**2)/4))
print('MSE = %f' %metrics.mean_squared_error(true,pred))

# NOW RMSE!
import numpy as np
print('RMSE = %f' %(np.sqrt((10**2+0**2+20**2+10**2)/4)))
print('RMSE = %f' %np.sqrt(metrics.mean_squared_error(true,pred)))

# Compute RMSE for sales predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# What if we're over fitting?
# Choose less/features
feature_cols = ['TV', 'radio']
X = df[feature_cols]
y = df.sales
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print('RMSE = %f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
