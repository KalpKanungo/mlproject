import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os
import sys
from src.utils import save_obj
@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns=["writing score","reading score"]
            catergorical_columns=[
                "gender","race/ethnicity","parental level of education","test preparation course"
            ]
            num_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )
            logging.info('Numerical Columns scaling done')
            catergorical_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("Encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info('Catergorical Columns encoding done')

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipline,numerical_columns),
                    ("cat_pipeline",catergorical_pipline,catergorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("The train and test is read")
            logging.info("Obtaining preprocessing object")
            preprocessor=self.get_data_transformer()
            target="math score"
            numerical_columns=["writing score","reading score"]
            input_feature_train_df=train_df.drop(columns=[target],axis=1)
            target_feature_train=train_df[target]

            input_feature_test_df=test_df.drop(columns=[target],axis=1)
            target_feature_test=test_df[target]
            logging.info("Applying the preprocessing on train and test")

            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train)]
            
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test)]
            logging.info("Saved preprocessing")
            save_obj(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor)

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        