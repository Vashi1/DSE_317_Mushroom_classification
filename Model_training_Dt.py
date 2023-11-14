import pandas as pd
from Data_loader import dummy_enc
from Data_imputation import knn_imputer, mean_imputer, median_imputer, mf_imputer
from GridSearch_params import split_data
from FeatureSelection import PCA_data, Kernal_PCA, TrSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold  

# Importing the Data
data = pd.read_csv('train.csv')
y = data.outcome
x = data.drop(['outcome'], axis=1)

# Applying Dummy Encoding
x_dum = dummy_enc(x)

# Applying KNN-Imputation on the data
x_kn = knn_imputer(x_dum)
# Applying Mean-Imputation on the data
x_me = mean_imputer(x_dum)
# Applying Median-Imputation on the data
x_med = median_imputer(x_dum)
# Applying MostFrequent-Imputation on the data
x_mf = mf_imputer(x_dum)

#Applying PCA for Feature Selection
x_kn_pca = PCA_data(x_kn)
x_me_pca = PCA_data(x_me)
x_med_pca = PCA_data(x_med)
x_mf_pca = PCA_data(x_mf)

#Applying Kernal PCA for Feature Selection
x_kn_kpca = Kernal_PCA(x_kn)
x_me_kpca = Kernal_PCA(x_me)
x_med_kpca = Kernal_PCA(x_med)
x_mf_kpca = Kernal_PCA(x_mf)

#Applying Truncated SVD for Feature Selection
x_kn_tsvd = TrSVD(x_kn)
x_me_tsvd = TrSVD(x_me)
x_med_tsvd = TrSVD(x_med)
x_mf_tsvd = TrSVD(x_mf)

print("Implementing Decision Tree Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection")
dt_dc_kn_PCA = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None, splitter='best')
x_tr, x_te, y_tr, y_te = split_data(x_kn_pca, y)
print("Decision Tree Classifier without using any Cross-Validation Technique")
dt_dc_kn_PCA.fit(x_tr, y_tr)
ypr = dt_dc_kn_PCA.predict(x_te)
print("Classification Report")
print(classification_report(y_te, ypr))
print("Plotting the Confusion Matrix")
cormat = confusion_matrix(y_te, ypr)
sns.heatmap(cormat, annot = True)
plt.show()

print("Implementing Decision Tree Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection with KFolds Cross Validation Technique")

print("Implementing Decision Tree Classifier on Data with Dummy Encoding with Mean-Imputation and PCA Feature Selection")
dt_dc_me_PCA = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None, splitter='best')
x_tr, x_te, y_tr, y_te = split_data(x_me_pca, y)
print("Decision Tree Classifier without using any Cross-Validation Technique")
dt_dc_me_PCA.fit(x_tr, y_tr)
ypr = dt_dc_kn_PCA.predict(x_te)
print("Classification Report")
print(classification_report(y_te, ypr))
print("Plotting the Confusion Matrix")
cormat = confusion_matrix(y_te, ypr)
sns.heatmap(cormat, annot = True)
plt.show()

print("Implementing Decision Tree Classifier on Data with Dummy Encoding with Median-Imputation and PCA Feature Selection")
dt_dc_med_PCA = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None, splitter='best')
x_tr, x_te, y_tr, y_te = split_data(x_med_pca, y)
print("Decision Tree Classifier without using any Cross-Validation Technique")
dt_dc_kn_PCA.fit(x_tr, y_tr)
ypr = dt_dc_kn_PCA.predict(x_te)
print("Classification Report")
print(classification_report(y_te, ypr))
print("Plotting the Confusion Matrix")
cormat = confusion_matrix(y_te, ypr)
sns.heatmap(cormat, annot = True)
plt.show()