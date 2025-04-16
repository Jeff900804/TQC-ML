# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:22:55 2023

@author: jeff9
"""


# #############################################################################
# 本題參數設定，請勿更改
seed = 0   # 亂數種子數  
# #############################################################################
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 

# TODO
data = pd.read_csv("Taipei_house.csv")
# 讀取台北市房價資料集
# TODO


# 對"行政區"進行 one-hot encoding
# TODO
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

data_dict = data[["行政區"]].to_dict(orient='records')
transfer = DictVectorizer(sparse = False)
new_data = transfer.fit_transform(data_dict)
data["行政區_信義區"] = new_data[:,0]
data["行政區_大安區"] = new_data[:,1]
data["行政區_文山區"] = new_data[:,2]
data["行政區_松山區"] = new_data[:,3]

# 處理"車位類別"
# TODO
# data["車位類別"] = data["車位類別"].apply(lambda x: 0 if x == '無' else 1)
label = LabelEncoder()
encoder = label.fit_transform(data["車位類別"])
data["車位類別"] = [0 if i==7 else 1 for i in encoder]
# 計算 Adjusted R-squared
def adj_R2(r2, n, k):
    """ 函式描述：計算 Adjusted R-squared
    參數：
        r2:R-squared 數值
        n: 樣本數
        k: 特徵數

    回傳：
        Adjusted R-squared
    """
    return r2-(k-1)/(n-k)*(1-r2)

# TODO

# 切分訓練集(80%)、測試集(20%)
features= ['土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '用途', 
           '房數', '廳數', '衛數', '電梯', '車位類別', 
           '行政區_信義區', '行政區_大安區', '行政區_文山區','行政區_松山區']
target = '總價'  
# TODO
x = data[features]
y = data[target]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score


models = {'複迴歸': LinearRegression(),
          '脊歸': Ridge(alpha=10),
          '多項式迴歸': PolynomialFeatures(degree=2),
          '多項式迴歸+Lasso迴歸': Lasso(alpha=10)}

for name, model in models.items():
    if name == '多項式迴歸':
        poly = PolynomialFeatures(degree=2)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        model = LinearRegression()
        model.fit(x_train_poly, y_train)
        y_pred_train = model.predict(x_train_poly)
        y_pred_test = model.predict(x_test_poly)
        
    elif name == '多項式迴歸+Lasso迴歸':
        poly = PolynomialFeatures(degree=2)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        model.fit(x_train_poly, y_train)
        y_pred_train = model.predict(x_train_poly)
        y_pred_test = model.predict(x_test_poly)

    else:
        model.fit(x_train , y_train )
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    print(f'{name} 訓練集RMSE: {rmse_train:.0f} 試集RMSE: {rmse_test:.0f}')
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train_adj = adj_R2(r2_train, x_train.shape[0] , x_train.shape[1])
    r2_test_adj = adj_R2(r2_test,x_test.shape[0] , x_test.shape[1])    
    if name == '多項式迴歸' or name == '多項式迴歸+Lasso迴歸':
        r2_train_adj = adj_R2(r2_train, x_train.shape[0] , x_train_poly.shape[1])
        r2_test_adj = adj_R2(r2_test,x_test.shape[0] , x_test_poly.shape[1])

    print(f'{name} 訓練集R^2: {r2_train_adj:.4f} 測試集R^2: {r2_test_adj:.4f}')
    
poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x)
estimator = Lasso(alpha = 10)
estimator.fit(x_poly,y)
x_test1 = [[36,99,32,4,4,0,3,2,1,0,0,0,0,0,1]]
y_pred2 = estimator.predict(poly.transform(x_test1))
print(f'預測房價: {y_pred2[0]:.0f}')