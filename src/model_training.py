import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from dvclive import Live


df=pd.read_csv('./data/admission.csv')

X = df.drop(columns=['admit'])
y = df['admit']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


n_estimator=100
max_depth=10

rf=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)


with Live(save_dvc_exp=True) as live :
        live.log_metric(('accuracy',accuracy_score(y_test,y_pred)))
        live.log_metric(('precision',precision_score(y_test,y_pred)))
        live.log_metric(('recall',recall_score(y_test,y_pred)))
        live.log_metric(('f1_score',f1_score(y_test,y_pred)))

        live.log_param('n_estimator',n_estimator)
        live.log_param('max_depth',max_depth)
