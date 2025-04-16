# TODO
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# TODO
X = iris_dataset.data
y = iris_dataset.target

# create dataframe from data in X_train 根據X_train中的資料創建dataframe
# label the columns using the strings in iris_dataset.feature_names 使用iris_dataset.feature_names中的字串標記列
# TODO
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=(1),train_size=0.6)
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier()
estimator.fit(X_train,y_train)


print("Test set score: ",estimator.score(X_test, y_test))
print(iris_dataset.target_names)
# TODO
X_test1 = [[5,2.9,1,0.2]]
y_pred = estimator.predict(X_test1)
print("Predicted target name:",y_pred)

# TODO
X_test2 = [[5.7,2.8,4.5,1.2]]
y_pred = estimator.predict(X_test2)
print("Predicted target name:",y_pred)

# TODO
X_test3 = [[7.7,3.8,6.7,2.1]]
y_pred = estimator.predict(X_test3)
print("Predicted target name:",y_pred)
