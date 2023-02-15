import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import (dump,load)
from sklearn import svm


data = load_iris()



data_=pd.DataFrame(data=data['data'])
target=pd.DataFrame(data=data['target'], columns=['target'])
data=pd.concat([data_,target], axis = 1)


X=data[[0,1,2,3]]
y=data['target']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, shuffle=True)


#Logestic regression 
lin_svc= load('lin_svc.joblib')
#LR score and prediction
print("NN score and classification:")
print(lin_svc.score(X_test, y_test))
print(lin_svc.predict(X_test))





#DecisionTreeClassifier
dtc= load('dtc_model.joblib')
#LR score and prediction
print(dtc.score(X_test, y_test))
print(dtc.predict(X_test))
print("See you later")