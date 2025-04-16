This folder contains the questions, implementation plans and answers for TQC+ Question 303.

Practice classification using 4 classifiers at the same time.  
**RandomForest、KNN、SVC、Voting Classifier**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
```
Using **K-fold cross** validation to test accuracy
```python
from sklearn.model_selection import KFold, cross_val_score
```
