from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
# TODO

data = datasets.load_diabetes()
#get x
# TODO 
x = data.data
y = data.target
#Total number of examples
# TODO 
estimator = LinearRegression()
estimator.fit(x, y)

from sklearn.metrics import mean_squared_error,r2_score
y_pred = estimator.predict(x)
MSE = mean_squared_error(y, y_pred)
R2 = r2_score(y, y_pred)
print('Total number of examples')
print('MSE=',MSE)
print('R-squared=',R2)
#3:1 100
from sklearn.model_selection import train_test_split
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(x,y,test_size = 0.25,random_state = 100)
lm2=LinearRegression()
lm2.fit(xTrain2,yTrain2)
#TODO 
y_pred2 = lm2.predict(xTrain2)
y_pred3 = lm2.predict(xTest2)
MSE2 = mean_squared_error(yTrain2, y_pred2)
MSE3 = mean_squared_error(yTest2, y_pred3)

print('Split 3:1')
print('train MSE=',MSE2            )
print('test MSE=',MSE3           )
print('train R-squared='               )
print('test R-squared='                )
