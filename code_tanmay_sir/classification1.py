
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


# Statistics of individual classes
def get_class_statistics(labels):
    class_statistics=Counter(labels)
    print('\n Class \t\t Number of Instances \n')
    for item in list(class_statistics.keys()):
        print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))
    
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
reader=pd.read_csv(path+'winequality_white.csv')  

# Select all rows except the ones belong to particular class'
# mask = reader['target'] == 9
# reader = reader[~mask]

data=reader.iloc[:, :-1]
labels=reader['target']

get_class_statistics(labels)  
input('\n \t Press enter to move forward   \n')
     
# Training and test split WITHOUT stratification        
# training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
#                                                test_size=0.10, random_state=42)
# Training and test split WITH stratification   
training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                               test_size=0.5, random_state=42,stratify=labels)
print('\n Training Data ')
training_cat=[str(x) for x in training_cat]
get_class_statistics(labels) 
input('\n \t Press enter to move forward   \n')

print('\n Validation Data ')
validation_cat=[str(x) for x in validation_cat]
get_class_statistics(labels) 
input('\n \t Press enter to move forward   \n')

  # Classification
     
clf1 = LogisticRegression(solver='liblinear',class_weight='balanced') 
clf2 = RandomForestClassifier(max_features=None,class_weight='balanced')
clf3 = svm.SVC(class_weight='balanced',kernel='linear',C=1,probability=True)
clf3 = MultinomialNB(fit_prior=True, class_prior=None)
clf4 = DecisionTreeClassifier(random_state=40) 
clf5 = svm.LinearSVC(class_weight='balanced') 

clf1.fit(training_data,training_cat)
predicted=clf1.predict(validation_data)
class_names=[str(item) for item in list(Counter(validation_cat).keys())]

# Classification report
print('\n ##### Classification Report ##### \n')
print(classification_report(validation_cat, predicted, target_names=class_names))


pr=precision_score(validation_cat, predicted, average='macro') 
print ('\n Precision:\t'+str(pr)) 

rl=recall_score(validation_cat, predicted, average='macro') 
print ('\n Recall:\t'+str(rl))

fm=f1_score(validation_cat, predicted, average='macro') 
print ('\n F1-Score:\t'+str(fm))

