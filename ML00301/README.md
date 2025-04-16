This folder contains the questions, implementation plans and answers for TQC+ Question 301.

Use sklearn's built-in **iris dataset**.
```python
from sklearn.datasets import load_iris
```

Splitting training and test sets.  
*X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=(1),train_size=0.6)*
```python
from sklearn.model_selection import train_test_split
```

Build a **KNN** machine learning model to predict iris varieties.
```python
from sklearn.neighbors import KNeighborsClassifier
```
