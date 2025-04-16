#from sklearn import datasets
#from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
# TODO
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
# df = pd.DataFrame(boston.data.T, ['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']) #有13個feature
# # TODO
# # MEDV即預測目標向量
# # TODO
# X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']]
# y = df['MEDV']
X = boston.data
y = boston.target
#分出20%的資料作為test set
# TODO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

transfer = StandardScaler()
X = transfer.fit_transform(X)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Fit linear model 配適線性模型

estimator = linear_model.LinearRegression()
estimator.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error,mean_absolute_error

y_pred = estimator.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred,squared=False)
# TODO

print('MAE:',mae)
print('MSE:',mse)
print('RMSE:',rmse)

X_new = ([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]])
X_new = transfer.transform(X_new)
prediction = estimator.predict(X_new)
print(prediction)
