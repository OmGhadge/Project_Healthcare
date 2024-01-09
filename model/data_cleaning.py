import logging
from abc import ABC, abstractmethod
from typing import  Union

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


class DataStrategy(ABC):
    
    """
    Abstract class defining strategy for handling data
    
    """
    
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->pd.DataFrame:
        pass
    
    
class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which processes the data
    
    """    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills the missing values and does other preprocessing steps
        Args:
            data (pd.DataFrame): _description_

        Returns:
            DataFrame: _description_
        """
        try:
            #Handling missing values in bmi column
            DT_bmi_pipe=Pipeline(steps=[('scale',StandardScaler()),('lr',DecisionTreeRegressor(random_state=42))])
            
            #copy relevant columns for imputation
            X=data[['age','gender','bmi']].copy()
            X.gender=X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype('np.uint8')
            
            #seperate missing values and non missing values
            Missing=X[X.bmi.isna()]
            X=X[~X.bmi.isna()]
            Y=X.pop('bmi')
            
            #fit the pipeline on non-missing data
            DT_bmi_pipe.fit(X,Y)
            
            #predict the missing 'bmi' values
            predicted_bmi=pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]) ,index=Missing.index)
            
            #Fill missing 'bmi'values in the original dataframe
            data.loc[Missing.index,'bmi']=predicted_bmi
            
            
            # Encoding categorical values

            data['gender'] = data['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
            data['Residence_type'] = data['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
            data['work_type'] = data['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)
            
            
            #handling the imbalance in the data
            
            from imblearn.over_sampling import SMOTE
            smote=SMOTE()
            X  = data[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
            y = data['stroke']

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
            X_train_resh, y_train_resh = smote.fit_resample(X_train, y_train.ravel())
            
            return data
            
        except Exception as e:    
            logging.error(e)
            raise e   

class DataDivideStrategy(DataStrategy):
    """
    Data division strategy which splits the data into train and test data

    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Divides the data into train and test data
    
        """
        try:
            x=data.drop("stroke",axis=1)
            y=data["stroke"]
            X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e :
            logging.error(e)
            raise e
        
class DataCleaning:
    """
    Data cleaning class which preprocesses the data and splits it into train and test data
    
    """
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy)->None:
        """
        Initializes the DataCleaning class with specific strategy
        """
        self.df=data
        self.strategy=strategy
        
    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        """
        handles the data based on provided strategy
         
        """
        return self.strategy.handle_data(self.df)    
        