import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

# 原始資料
titanic = pd.read_csv("titanic.csv")
print('raw data')
# TODO
titanic["Age"].fillna(titanic["Age"].median(),inplace = True)
# 將年齡的空值填入年齡的中位數
# TODO

print("年齡中位數=",titanic["Age"].median())
# TODO

# 更新後資料
#print(titanic["Age"])
# TODO

# 轉換欄位值成為數值
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(titanic["PClass"])
titanic["PClass"] = encoded_class


# TODO
X = titanic[["PClass","Age","SexCode"]]
y = titanic["Survived"]

estimator = linear_model.LogisticRegression()
estimator.fit(X,y)

intercept = estimator.intercept_
coef = estimator.coef_[0][2]
# 建立模型
# TODO

print('截距=%.4f'%intercept)
print('迴歸係數=%.4f'%coef)


# 混淆矩陣(Confusion Matrix)，計算準確度
print('Confusion Matrix')
# TODO
from sklearn.metrics import  accuracy_score
y_pred = estimator.predict(X)
accuracy = accuracy_score(y, y_pred)
print(accuracy)


