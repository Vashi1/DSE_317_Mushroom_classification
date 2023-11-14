
"""
Created on Thu October 05, 2023, 10:35:24

@author: Tanmay Basu
"""

import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from collections import Counter

path='/home/tanmay/ml_class_aug_dec_2023/code/'

# Load the file using CSV Reader          
# fl=open(self.path+'winequality_white.csv',"r")  
# reader = list(csv.reader(fl,delimiter='\n')) 
# fl.close()
# data=[]; labels=[];
# for item in reader[1:]:
#     item=''.join(item).split(';')
#     labels.append(item[-1]) 
#     data.append(item[:-1])
# # labels=[int(''.join(item)) for item in labels]
# data=np.asarray(data)
 
# Load the file using Pandas       
reader=pd.read_csv(path+'energy_consumtion_data.csv')  

# Select all rows except the ones belong to particular class'
# mask = reader['target'] == 9
# reader = reader[~mask]

data=reader.iloc[:, :-1]
labels=reader['target']

     
# Training and test split WITHOUT stratification        
training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                                test_size=0.10, random_state=42)

print('\n Training Data ')
training_cat=[x for x in training_cat]

print('\n Validation Data ')
validation_cat=[x for x in validation_cat]

  # Classification
     
rgr1 = LinearRegression() 
rgr2 = Ridge(alpha=1.0,solver='lbfgs',positive=True)
rgr3 = Lasso(alpha=1.0) 
rgr4 = DecisionTreeRegressor(random_state=0)
 

rgr2.fit(training_data,training_cat)
predicted=rgr2.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))

