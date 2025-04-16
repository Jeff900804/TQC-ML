# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################

import pandas as pd

# 載入寶可夢資料
# TODO
data = pd.read_csv("pokemon.csv")
# 取出目標欄位
X = data[["Defense","SpecialAtk"]] #TODO     特徵欄位
y = data["Type1"] #TODO     Type1 欄位

# 編碼 Type1
from sklearn import preprocessing
# TODO

# 切分訓練集、測試集，除以下參數設定外，其餘為預設值
# #########################################################################
# X, y, test_size=0.2, random_state=seed
# #########################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)
# 訓練集
# 分別建立 RandomForest, kNN, SVC, Voting，除以下參數設定外，其餘為預設值
# #############################################################################
# RandomForest: n_estimators=10, random_state=seed
# kNN: n_neighbors=4
# SVC: gamma=.1, kernel='rbf', probability=True
# Voting: estimators=[('RF', clf1), ('kNN', clf2), ('SVC', clf3)], 
#         voting='hard', n_jobs=-1
# #############################################################################    
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# TODO
estimator_f = RandomForestClassifier(n_estimators=10, random_state=seed) 
estimator_k = KNeighborsClassifier(n_neighbors=4)
estimator_s = SVC(gamma=.1, kernel='rbf', probability=True)
estimator_v = VotingClassifier(estimators=[('RF', estimator_f), ('kNN', estimator_k), ('SVC', estimator_s)],voting='hard', n_jobs=-1)
# 建立函式 kfold_cross_validation() 執行 k 折交叉驗證，並回傳準確度的平均值
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

def kfold_cross_validation(scalar, model):
    """ 函式描述：執行 k 折交叉驗證
    參數：
        scalar (StandardScaler):標準化適配的結果
        model: 機器學習模型

    回傳：
        k 折交叉驗證的準確度(accuracy)平均值
    """
    # 建立管線，用來進行(標準化 -> 機器學習模型)
    # pipeline = make_pipeline(scalar,model)  #TODO
    # pipeline.fit(X_train,y_train)
    model.fit(X_train,y_train)
    # 產生 k 折交叉驗證，除以下參數設定外，其餘為預設值
    # #########################################################################
    # n_splits=5, shuffle=True, random_state=seed
    # #########################################################################
    kf = KFold(n_splits=5, shuffle=True, random_state=seed) #TODO
    
    # 執行 k 折交叉驗證
    # #########################################################################
    # pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1
    # #########################################################################
    cv_result =cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1) #TODO
    
    return cv_result.mean() #TODO
# 利用 kfold_cross_validation()，分別讓分類器執行 k 折交叉驗證，計算準確度(accuracy)

#TODO

# #############################################################################
result_f = kfold_cross_validation(StandardScaler(),estimator_f)
result_k = kfold_cross_validation(StandardScaler(),estimator_k)
result_s = kfold_cross_validation(StandardScaler(),estimator_s)
result_v = kfold_cross_validation(StandardScaler(),estimator_v)
print(result_f,result_k,result_s,result_v)
# 利用訓練集的標準化結果，針對測試集進行標準化
# TODO

# 上述分類器針對測試集進行預測，並計算分類錯誤的個數與準確度
from sklearn.metrics import accuracy_score
# TODO
y_predf = estimator_f.predict(X_test)
acc_f = accuracy_score(y_test, y_predf)

y_predk = estimator_k.predict(X_test)
acc_k = accuracy_score(y_test, y_predk)

y_preds = estimator_s.predict(X_test)
acc_s = accuracy_score(y_test, y_preds)

y_predv = estimator_v.predict(X_test)
acc_v = accuracy_score(y_test, y_predv)
# #############################################################################
print(acc_f,acc_k,acc_s,acc_v)
print("error: ",len(y_test)-accuracy_score(y_test, y_predk,normalize=False))
# 分別利用上述分類器預測分類
print("===== 預測分類 ======")
# TODO
non = [[100,70]]
non = transfer.transform(non)
y_pred = estimator_f.predict(non)
print(y_pred)
