#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu October 06, 2023, 11:24:03

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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter


class regression():
     def __init__(self,path='/home/xyz/ml_code/energy_consumtion_data.csv',rgr_opt='lr',no_of_selected_features=None):
        self.path = path
        self.rgr_opt=rgr_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of regression techniques  
     def regression_pipeline(self):    
    # AdaBoost 
        if self.rgr_opt=='ab':
            print('\n\t### AdaBoost Regression ### \n')
            be1 = LinearRegression()              
            be2 = DecisionTreeRegressor(random_state=0)
            be2 = Ridge(alpha=1.0,solver='lbfgs',positive=True)             
            rgr = AdaBoostRegressor(n_estimators=100)
            rgr_parameters = {
            'rgr__base_estimator':(be1,be2),
            'rgr__random_state':(0,10),
            }      
    # Decision Tree
        elif self.rgr_opt=='dt':
            print('\n\t### Decision Tree ### \n')
            rgr = DecisionTreeRegressor(random_state=40) 
            rgr_parameters = {
            'rgr__criterion':('squared_error','friedman_mse','poisson'), 
            'rgr__max_features':('auto', 'sqrt', 'log2'),
            'rgr__max_depth':(10,40,45,60),
            'rgr__ccp_alpha':(0.009,0.01,0.05,0.1),
            } 
    # Ridge Regression 
        elif self.rgr_opt=='rg':
            print('\n\t### Ridge Regression ### \n')
            rgr = Ridge(alpha=1.0,positive=True) 
            rgr_parameters = {
            'rgr__solver':('auto', 'lbfgs'),
            } 
    # Linear Regression 
        elif self.rgr_opt=='lr':   
            print('\n\t### Linear Regression ### \n')
            rgr = LinearRegression()  
            rgr_parameters = {
            'rgr__positive':(True,False),
            }         
    # Random Forest 
        elif self.rgr_opt=='rf':
            print('\n\t ### Random Forest ### \n')
            rgr = RandomForestRegressor(max_features=None)
            rgr_parameters = {
            'rgr__criterion':('squared_error','friedman_mse','poisson'),       
            'rgr__n_estimators':(30,50,100),
            'rgr__max_depth':(10,20,30),
            'rgr__max_features':('auto', 'sqrt', 'log2'),
            }          
    # Support Vector Machine  
        elif self.rgr_opt=='svr': 
            print('\n\t### SVM Regressor ### \n')
            rgr = SVR(probability=True)  
            rgr_parameters = {
            'rgr__C':(0.1,1,100),
            'rgr__kernel':('linear','rbf','poly','sigmoid'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return rgr,rgr_parameters     

# Load the data 
     def get_data(self):
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
        reader=pd.read_csv(self.path+'energy_consumtion_data.csv')  
        
    # Select all rows except the ones belong to particular class'
        # mask = reader['class'] == 9
        # reader = reader[~mask]
        
        data=reader.iloc[:, :-1]
        labels=reader['target']
 
        # Training and Test Split           
        training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                               test_size=0.5, random_state=42)   

        return training_data, validation_data, training_cat, validation_cat
    
# Regression using the Gold Statndard after creating it from the raw text    
     def regression(self):  
   # Get the data
        training_data, validation_data, training_cat, validation_cat=self.get_data()

        rgr,rgr_parameters=self.regression_pipeline()
        pipeline = Pipeline([('rgr', rgr),])
        grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10)          
        grid.fit(training_data,training_cat)     
        rgr= grid.best_estimator_  
        print('\n\n The best set of parameters of the pipiline are: ')
        print(rgr)     
        joblib.dump(rgr, self.path+'training_model.joblib')
        predicted=rgr.predict(validation_data)


    # Regression report
        mse=mean_squared_error(validation_cat,predicted,squared=True)
        print ('\n MSE:\t'+str(mse)) 
        rmse=mean_squared_error(validation_cat,predicted,squared=False)
        print ('\n RMSE:\t'+str(rmse))
        r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
        print ('\n R2-Score:\t'+str(r2))


