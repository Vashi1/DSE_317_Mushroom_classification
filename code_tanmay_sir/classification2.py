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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter


class classification():
     def __init__(self,path='/home/xyz/ml_code/winequality_white.csv',clf_opt='lr',no_of_selected_features=None):
        self.path = path
        self.clf_opt=clf_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of classifiers  
     def classification_pipeline(self):    
    # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True)              
            be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            be3 = DecisionTreeClassifier(max_depth=50)
#            clf = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)            
            clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=100)
            clf_parameters = {
            'clf__base_estimator':(be1,be2,be3),
            'clf__random_state':(0,10),
            }      
    # Decision Tree
        elif self.clf_opt=='dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40) 
            clf_parameters = {
            'clf__criterion':('gini', 'entropy'), 
            'clf__max_features':('auto', 'sqrt', 'log2'),
            'clf__max_depth':(10,40,45,60),
            'clf__ccp_alpha':(0.009,0.01,0.05,0.1),
            } 
    # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
    # Linear SVC 
        elif self.clf_opt=='ls':   
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,100),
            }         
    # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }            
    # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__criterion':('entropy','gini'),       
            'clf__n_estimators':(30,50,100),
            'clf__max_depth':(10,20,30,50,100,200),
            }          
    # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced',probability=True)  
            clf_parameters = {
            'clf__C':(0.1,1,100),
            'clf__kernel':('linear','rbf','poly','sigmoid'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters     

# Statistics of individual classes
     def get_class_statistics(self,labels):
        class_statistics=Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))
    
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
        reader=pd.read_csv(self.path+'winequality_white.csv')  
        
    # Select all rows except the ones belong to particular class'
        # mask = reader['class'] == 9
        # reader = reader[~mask]
        
        data=reader.iloc[:, :-1]
        labels=reader['target']

        self.get_class_statistics(labels)  
        # Training and Test Split           
        training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                               test_size=0.05, random_state=42,stratify=labels)   

        return training_data, validation_data, training_cat, validation_cat
    
# Classification using the Gold Statndard after creating it from the raw text    
     def classification(self):  
   # Get the data
        training_data, validation_data, training_cat, validation_cat=self.get_data()

        clf,clf_parameters=self.classification_pipeline()
        pipeline = Pipeline([
                    ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
            #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                    ('clf', clf),])
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_macro',cv=5)          
        grid.fit(training_data,training_cat)     
        clf= grid.best_estimator_  
        print('\n\n The best set of parameters of the pipiline are: ')
        print(clf)     
        joblib.dump(clf, self.path+'training_model.joblib')
        predicted=clf.predict(validation_data)

    # Evaluation
        class_names=list(Counter(validation_cat).keys())
        class_names = [str(x) for x in class_names] 
        # print('\n The classes are: ')
        # print(class_names) 
    # Evaluation
        print('\n *************** Confusion Matrix ***************  \n')
        print (confusion_matrix(validation_cat, predicted)) 

        class_names=list(Counter(validation_cat).keys())
        class_names = [str(x) for x in list(Counter(validation_cat).keys())]        

        # Classification report
        print('\n ##### Classification Report ##### \n')
        print(classification_report(validation_cat, predicted, target_names=class_names))        
        
        pr=precision_score(validation_cat, predicted, average='macro') 
        print ('\n Precision:\t'+str(pr)) 
        
        rl=recall_score(validation_cat, predicted, average='macro') 
        print ('\n Recall:\t'+str(rl))
        
        fm=f1_score(validation_cat, predicted, average='macro') 
        print ('\n F1-Score:\t'+str(fm))
