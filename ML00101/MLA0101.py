import numpy as np
import pandas as pd

input_file = 'wine.csv'
data = pd.read_csv(input_file,header = None)
data = np.array(data)

X = data[:,1:]
y = data[:,0]


# TODO


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,random_state = 5)

from sklearn.tree import DecisionTreeClassifier

estimator = DecisionTreeClassifier()
estimator.fit(X_train,y_train)

# TODO


# compute accuracy of the classifier計算分類器的精確度
accuracy = estimator.score(X_test,y_test)
print("Accuracy of the classifier =",round(accuracy*100,2), "%")

X_test1 =[[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]]
X_test2 = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065]]
X_test3 = [[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]]

y_predict1 = estimator.predict(X_test1)
print(y_predict1)
y_predict2 = estimator.predict(X_test2)
print(y_predict2)
y_predict3 = estimator.predict(X_test3)
print(y_predict3)
# TODO

