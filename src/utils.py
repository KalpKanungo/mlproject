import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file:
            dill.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)
def eval_model(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}
        model_names = list(models.keys())
        model_values = list(models.values())
        params=list(param.values())

        for i in range(len(models)):
            model_name = model_names[i]
            model = model_values[i]
            para=params[i]

            gs =GridSearchCV(model, param_grid=para, cv=3, scoring='r2', n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
