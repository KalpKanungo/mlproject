import os 
import sys 
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,eval_model

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainerconfig()

    def initiateModeltrainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing input has started")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
            "Linear Regression":LinearRegression(),
            "Decision Tree":DecisionTreeRegressor(),
            'Random Forest':RandomForestRegressor(),
            "Ada Boost":AdaBoostRegressor(),
            "Cat Boost":CatBoostRegressor(verbose=False),
            "XGBRegressor":XGBRegressor(),
            "K-Neighbors":KNeighborsRegressor()
            }
            params = {
                "Linear Regression": {},

                "Decision Tree": {
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 5, 10],
                    "criterion": ["squared_error", "friedman_mse"]
                },
                
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                
                "Ada Boost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },

                "Cat Boost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6]
                },
                
                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6]
                },
                
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree"]
                }
            }


            model_report:dict=eval_model(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                   models=models,param=params)
            best_model_score=max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model found")
            logging.info(f"best model found on both training and testing data")

            save_obj(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)