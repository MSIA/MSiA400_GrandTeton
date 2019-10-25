import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics

build = pd.read_csv('Building_Set_Balanced_10000_Generated_9_MAY_2019.csv')
build.iloc[:, :-2] = (build.iloc[:, :-2] - build.iloc[:, :-2].mean()) / build.iloc[:, :-2].std()
build.Go = build.Go.astype(int)
build = build.fillna(build.mean())

X_train, X_test, y_train, y_test = train_test_split(build.iloc[:, :-3], build.iloc[:, -2], test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))