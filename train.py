import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from sklearn import svm


data = load_iris()



data_=pd.DataFrame(data=data['data'])
target=pd.DataFrame(data=data['target'], columns=['target'])
data=pd.concat([data_,target], axis = 1)


X=data[[0,1,2,3]]
y=data['target']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, shuffle=True)


#Logestic regression 
C = 1.0 # paramètre de régularisation
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train, y_train)
#lR.score(X_test,y_test)

dump(lin_svc, 'lin_svc.joblib')




#DecisionTreeClassifier
dtc= DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
#dtc.score(X_test,y_test)


dump(dtc, 'dtc_model.joblib')