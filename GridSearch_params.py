import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import GradientBoostingClassifier

def split_data(data_red, labels):
    X_Train, X_Test, y_train, y_test = train_test_split(data_red, labels, test_size=0.2, random_state=42)
    return X_Train, X_Test, y_train, y_test


# Searching the Best set of parameters for Decision Tree
def grid_search_dt(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    param_grid_dt = [
        {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
            'max_features': [None, 'sqrt', 'log2'],
            'ccp_alpha': [0.0, 0.1, 0.01, 0.001, 0.2, 0.002]
        }
    ]
    gs_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, n_jobs=-1)
    gs_dt.fit(X_train, y_train)
    print("Optimal Decision Parameters: \n\n")
    print(gs_dt.best_params_)
    score = cross_val_score(gs_dt, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))


def grid_search_rft(X_train, y_train):
    rft = RandomForestClassifier(random_state=42)
    param_grid_rft = [
        {
            'n_estimators': [100, 10, 200, 150, 250, 300],
            'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 19, 20],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'ccp_alpha': [0, 0.1, 0.01, 0.001, 0.001, 0.0015]

        }
    ]
    gs_rft = GridSearchCV(rft, param_grid_rft, scoring='accuracy', cv=5, n_jobs=-1)
    gs_rft.fit(X_train, y_train)
    print("Optimal Random Forest Tree Parameters: \n\n")
    print(gs_rft.best_params_)
    score = cross_val_score(gs_rft, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))


def grid_search_lr(X_train, y_train):
    lr = LogisticRegression(random_state=42)
    param_grid_lr = [
        {
            'penalty': [None, 'l2'],
            'C': [1, 1.1],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag'],
        }
    ]
    gs_lr = GridSearchCV(lr, param_grid_lr, scoring='accuracy', cv=5, n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    print("Optimal Logistic Regreesion Parameters: \n\n")
    print(gs_lr.best_params_)
    score = cross_val_score(gs_lr, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))


def grid_search_svm(X_train, y_train):
    svc = SVC(random_state=42)
    param_grid_svc = [
        {
            'C': [1, 1.2,],
            'kernel': ['linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'shrinking': [True, False],
            'decision_function_shape': ['ovo', 'ovr']
        }
    ]
    gr_svm = GridSearchCV(svc, param_grid_svc, scoring='accuracy', cv=5, n_jobs=-1)
    gr_svm.fit(X_train, y_train)
    print("Optimal Support Vector Machine Parameters: \n\n")
    print(gr_svm.best_params_)
    score = cross_val_score(gr_svm, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))


def grid_search_nb(X_train, y_train):
    nb = CategoricalNB()
    param_grid_nb = [
        {
            'alpha': [1, 1.1, 2.5],
            'fit_prior': [True, False],
        }
    ]
    gr_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1)
    gr_nb.fit(X_train, y_train)
    print("Optimal Naive Bayes Classifier Parameters: \n\n")
    print(gr_nb.best_params_)
    score = cross_val_score(gr_nb, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))

def grid_search_gb(X_train, y_train):
    gb = GradientBoostingClassifier(random_state=42)
    param_grid_gb = [
        {
            'loss' : ['log_loss', 'exponential'],
            'learning_rate' : [0.1, 0.01, 0.02]
        }
    ]
    gr_nb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
    gr_nb.fit(X_train, y_train)
    print("Optimal Gradient Boosting Parameters: \n\n")
    print(gr_nb.best_params_)
    score = cross_val_score(gr_nb, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))