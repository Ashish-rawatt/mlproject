import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self): # this is responsible for data transformation
        try:
            numerical_columns=["reading_score", "writing_score"]
            categorical_columns= [
                "gender", 
                "race_ethnicity",
                "parental_level_of_education",
                "lunch", 
                "test_preparation_course"]
            
            num_pipline= Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )
        
            cat_pipline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical Columns: {categorical_columns}")
            logging.info(f"Categorical Columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipline",num_pipline, numerical_columns),
                ("cate_pipline",cat_pipline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)



