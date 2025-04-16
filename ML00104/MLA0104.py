import pandas as pd

# 載入寶可夢資料集
# TODO
data = pd.read_csv('pokemon.csv')

# 處理遺漏值
features = ['Attack', 'Defense']
# TODO

data = data.dropna(subset=features)

# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO

# 編碼 Type1

# TODO
y = data[data['Type1'].isin(["Normal","Fighting","Ghost"])]["Type1"]
X = data[data['Type1'].isin(["Normal","Fighting","Ghost"])][features]


# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
transfer = StandardScaler()
X = transfer.fit_transform(X)



# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
estimator = LinearSVC(C=0.1, dual=False, class_weight='balanced')
estimator.fit(X,y)
# 計算分類錯誤的數量
# TODO

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
y_pred = estimator.predict(X)

print("error: ",len(y)-accuracy_score(y, y_pred,normalize=False))
print('Accuracy: ',accuracy_score(y, y_pred))

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO

print('F1-score: ',f1_score(y, y_pred,average='weighted'))

# 預測未知寶可夢的 Type1
# TODO
X1 = [[100,75]]
X1 = transfer.transform(X1)
y_pred_1 = estimator.predict(X1)
print('Type1: ',y_pred_1)

