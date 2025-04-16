import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt

input_file = 'cardata.txt'
data = np.genfromtxt(input_file, delimiter=',', dtype='str')
# Reading the data
X = data[:,:6]
y = data[:,6]
# TODO

# Convert string data to numerical data將字串資料轉換為數值資料
# TODO
for i in range(6):
    label_encoder = preprocessing.LabelEncoder()
    encoded_class = label_encoder.fit_transform(data[:,i])
    data[:,i] = encoded_class
    print(label_encoder.classes_)
# Build a Random Forest classifier建立隨機森林分類器
# TODO

estimator = RandomForestClassifier(max_depth=(8),random_state=(7))

# Cross validation交叉驗證
from sklearn import model_selection
# TODO

param_dict = {"n_estimators":[200]}
estimator = model_selection.GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(X,y)

print("Accuracy of the classifier=%.2f"%(estimator.best_score_*100)+"%")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = ['high', 'low', '2', 'more', 'med', 'high']
input_data = [[0,1,0,2,1,0]] 
y_predict = estimator.predict(input_data)
# TODO

print(y_predict)
# Predict and print output for a particular datapoint
# TODO
print("Output class="                     )

########################
# Validation curves 驗證曲線

# TODO

estimator = RandomForestClassifier(max_depth=(8),random_state=(7))
train_scores, validation_scores = model_selection.validation_curve(estimator, X, y, 
        param_name = "n_estimators", param_range=np.linspace(25,200,8).astype(int), cv=5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)



