

# Importing the libraries
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('./diabetes2.csv')
X = dataset.iloc[:,:6].values
y = dataset.iloc[:, 6].values
# print(X)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Aplying kernal SVC to the training set
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

filename='diab_pred_model'
pickle.dump(classifier,open(filename,'wb'))
# Predicting the test set result
y_pred = classifier.predict(X_test)  # 2d innput here

# Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
print(X_test,y_pred)
#[148.   72.   35.    0.   33.6  50. ]