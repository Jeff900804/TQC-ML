import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
from sklearn.metrics import r2_score


NBApoints_data= pd.read_csv("NBApoints.csv")
#TODO

y = NBApoints_data["3P"] 

label_encoder_conver = preprocessing.LabelEncoder()
Pos_encoder_value = label_encoder_conver.fit_transform(NBApoints_data["Pos"])
NBApoints_data["Pos"] = Pos_encoder_value

label_encoder_conver = preprocessing.LabelEncoder()
Tm_encoder_value = label_encoder_conver.fit_transform(NBApoints_data["Tm"])
NBApoints_data["Tm"] = Tm_encoder_value


#train_X = pd.DataFrame(NBApoints_data["Pos"],NBApoints_data["Age"]).T
train_X = NBApoints_data[["Pos","Age","Tm"]]
NBApoints_linear_model = LinearRegression()
NBApoints_linear_model.fit(train_X, y)

X = [[5,28,10]]
NBApoints_linear_model_predict_result=NBApoints_linear_model.predict(X)
print("三分球得球數=",NBApoints_linear_model_predict_result)

y_pred = NBApoints_linear_model.predict(train_X)

# r_squared =
# print("R_squared值=",r_squared)

# print("f_regresstion\n")
# print("P值="              )
# print("\n")
f = f_regression(train_X, y)
print(r2_score(y, y_pred))
print(f[1][0],f[1][1])