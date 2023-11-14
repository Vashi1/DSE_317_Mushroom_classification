#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu October 06, 2023, 11:48:27

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
from sklearn.model_selection import KFold, train_test_split
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
     def get_data(self,filename):
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
        reader=pd.read_csv(self.path+filename)  
        
    # Select all rows except the ones belong to particular class'
        # mask = reader['class'] == 9
        # reader = reader[~mask]
        
        data=reader.iloc[:, :-1]
        labels=reader['target']       

        return data, labels
    
# Regression using the Gold Statndard after creating it from the raw text    
     def regression(self):  
   # Get the data
        data,labels=self.get_data('energy_consumtion_data.csv')
        data=np.asarray(data)

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = KFold(n_splits=10)
        predicted_target=[]; actual_target=[]; 
        count=0; 
        for train_index, test_index in skf.split(data,labels):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(data[item])
                y_train.append(labels[item])
            for item in test_index:
                X_test.append(data[item])
                y_test.append(labels[item])
            count+=1                
            print('Training Phase '+str(count))
            rgr,rgr_parameters=self.regression_pipeline()
            pipeline = Pipeline([('rgr', rgr),])
            grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_micro',cv=10)          
            grid.fit(X_train,y_train)     
            rgr= grid.best_estimator_  
            # print('\n\n The best set of parameters of the pipiline are: ')
            # print(rgr)     
            predicted=rgr.predict(X_test)  
            for item in y_test:
                actual_target.append(item)
            for item in predicted:
                predicted_target.append(item)           

    # Regression report
        print('\n\n Performance on Training Data \n')   
        mse=mean_squared_error(actual_target,predicted_target,squared=True)
        print ('\n MSE:\t'+str(mse)) 
        rmse=mean_squared_error(actual_target,predicted_target,squared=False)
        print ('\n RMSE:\t'+str(rmse))
        r2=r2_score(actual_target,predicted_target,multioutput='variance_weighted') 
        print ('\n R2-Score:\t'+str(r2))
        
        # Experiments on Given Test Data during Test Phase
        
        tst_data,tst_target=self.get_data('energy_consumtion_data.csv')
        tst_data=np.asarray(data)
        predicted=rgr.predict(tst_data)
        
        print('\n\n Performance on Test Data \n')
        mse=mean_squared_error(tst_target,predicted_target,squared=True)
        print ('\n MSE:\t'+str(mse)) 
        rmse=mean_squared_error(tst_target,predicted_target,squared=False)
        print ('\n RMSE:\t'+str(rmse))
        r2=r2_score(tst_target,predicted_target,multioutput='variance_weighted') 
        print ('\n R2-Score:\t'+str(r2)) 
