from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_val_predict

def adb_kn_classifier_pca(x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                                 splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2', n_estimators=250)
    svm_clf = SVC(C = 1.2, kernel = 'linear',  random_state=42)
    lr_clf = LogisticRegression(C = 1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost Classifier without using any Cross-Validation Technique and PCA Feature Selection")
    clf.fit(x_tr, y_tr)
    ypr = clf.predict(x_te)
    print("Classification Report")
    print(classification_report(y_te, ypr))
    print("Plotting the Confusion Matrix")
    cormat = confusion_matrix(np.array(y_te).argmax(axis=1), np.array(ypr).argmax(axis=1))
    sns.heatmap(cormat, annot= True, fmt = 'g')
    print(clf.feature_importances_)
    plt.savefig("Implementing XGboost Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection.png")
    plt.show()

def adb_kn_classifier_pca_kfold(x_data, y_data,x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost Classifier with K-Fold Cross-Validation Technique(K = 10) and PCA Feature Extraction")
    outer_cv = KFold(n_splits=10,  random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average = 'macro')
    pre_scorer_weighted = make_scorer(precision_score, average = 'weighted')
    nested_score_precision_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average = 'macro')
    rec_score_weighted = make_scorer(recall_score, average = 'weighted')
    nested_score_recall_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average = 'micro')
    f1_scorer_weighted = make_scorer(f1_score, average = 'weighted')
    nested_score_f1_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(clf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(np.array(y_te).argmax(axis = 1), np.array(y_pr).argmax(axis = 1))
    sns.heatmap(cormat, annot=True, fmt = 'g')
    print(clf.feature_importances_)
    plt.savefig("Implementing AdaBoost Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection with K-Fold Cross Validation.png")
    plt.show()

def adb_kn_classifier_pca_stratfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost with Stratified K-Fold Cross-Validation Technique and PCA Feature Extraction Technique")
    outer_cv = StratifiedKFold(n_splits=10,  random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    y_data = np.argmax(y_data, axis = 1)
    nested_score_precision_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print(clf.feature_importances_)
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(clf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(np.array(y_te).argmax(axis = 1), np.array(y_pr).argmax(axis = 1))
    sns.heatmap(cormat, annot=True,  fmt = 'g')
    plt.savefig("Implementing AdaBoost Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection with Stratified K-Fold Cross Validation.png")
    plt.show()



def adb_kn_classifier_kpca(x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost Classifier without using any Cross-Validation Technique and Kernel PCA Feature Selection")
    clf.fit(x_tr, y_tr)
    ypr = clf.predict(x_te)
    print("Classification Report")
    print(classification_report(y_te, ypr))
    print("Plotting the Confusion Matrix")
    print(clf.feature_importances_)
    cormat = confusion_matrix(np.array(y_te).argmax(axis=1), np.array(ypr).argmax(axis=1))
    sns.heatmap(cormat, annot fmt = 'g')
    plt.savefig("Implementing XGboost Classifier on Data with Dummy Encoding with KNN-Imputation and Kernel PCA Feature Selection.png")
    plt.show()


def adb_kn_classifier_kpca_kfold(x_data, y_data,x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost Classifier with K-Fold Cross-Validation Technique(K = 10) and Kernel PCA Feature Extraction")
    outer_cv = KFold(n_splits=10,  random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average = 'macro')
    pre_scorer_weighted = make_scorer(precision_score, average = 'weighted')
    nested_score_precision_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average = 'macro')
    rec_score_weighted = make_scorer(recall_score, average = 'weighted')
    print(clf.feature_importances_)
    nested_score_recall_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average = 'micro')
    f1_scorer_weighted = make_scorer(f1_score, average = 'weighted')
    nested_score_f1_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(clf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(np.array(y_te).argmax(axis = 1), np.array(y_pr).argmax(axis = 1))
    sns.heatmap(cormat, annot fmt = 'g')
    print(clf.feature_importances_)
    plt.savefig("Implementing AdaBoost Classifier on Data with Dummy Encoding with KNN-Imputation and Kernel PCA Feature Selection with K-Fold Cross Validation.png")
    plt.show()


def adb_kn_classifier_kpca_stratfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost with Stratified K-Fold Cross-Validation Technique and Kernel PCA Feature Extraction Technique")
    outer_cv = StratifiedKFold(n_splits=10,  random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    y_data = np.argmax(y_data, axis=1)
    nested_score_precision_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                            scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(clf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(np.array(y_te).argmax(axis=1), np.array(y_pr).argmax(axis=1))
    sns.heatmap(cormat, annot fmt='g')
    print(clf.feature_importances_)

    plt.savefig(
        "Implementing AdaBoost Classifier on Data with Dummy Encoding with KNN-Imputation and Kernel PCA Feature Selection with Stratified K-Fold Cross Validation.png")
    plt.show()


def adb_kn_classifier_trsvd(x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    clf.fit(x_tr, y_tr)
    ypr = clf.predict(x_te)
    print("Classification Report")
    print(classification_report(y_te, ypr))
    print("Plotting the Confusion Matrix")
    cormat = confusion_matrix(np.array(y_te).argmax(axis=1), np.array(ypr).argmax(axis=1))
    sns.heatmap(cormat, annot fmt = 'g')
    print(clf.feature_importances_)

    plt.savefig("Implementing XGboost Classifier on Data with Dummy Encoding with KNN-Imputation and Truncated SVD Feature Selection.png")
    plt.show()


def adb_kn_classifier_trsvd_kfold(x_data, y_data,x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost Classifier with K-Fold Cross-Validation Technique(K = 10) and Truncated SVD Feature Extraction")
    outer_cv = KFold(n_splits=10,  random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average = 'macro')
    pre_scorer_weighted = make_scorer(precision_score, average = 'weighted')
    nested_score_precision_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average = 'macro')
    rec_score_weighted = make_scorer(recall_score, average = 'weighted')
    nested_score_recall_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average = 'micro')
    f1_scorer_weighted = make_scorer(f1_score, average = 'weighted')
    nested_score_f1_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(clf, x_te, y_te, cv =  outer_cv)
    cormat = confusion_matrix(np.array(y_te).argmax(axis = 1), np.array(y_pr).argmax(axis = 1))
    sns.heatmap(cormat, annot fmt = 'g')
    print(clf.feature_importances_)

    plt.savefig("Implementing AdaBoost Classifier on Data with Dummy Encoding with KNN-Imputation and Truncated SVD Feature Selection with K-Fold Cross Validation.png")
    print(clf.feature_importances_)
    plt.show()


def adb_kn_classifier_trsvd_stratfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=None, max_features=None,
                                    splitter='best')
    rf_clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced', max_depth=19, max_features='log2',
                                    n_estimators=250)
    svm_clf = SVC(C=1.2, kernel='linear',  random_state=42)
    lr_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
    base_estimators = [dt_clf, rf_clf, svm_clf, lr_clf]
    clf = AdaBoostClassifier(base_estimators, random_state=42)
    print("AdaBoost with Stratified K-Fold Cross-Validation Technique and Truncated SVD Feature Extraction Technique")
    outer_cv = StratifiedKFold(n_splits=10,  random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    y_data = np.argmax(y_data, axis = 1)
    nested_score_precision_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print(clf.feature_importances_)

    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(clf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(clf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(np.array(y_te).argmax(axis = 1), np.array(y_pr).argmax(axis = 1))
    sns.heatmap(cormat, annot fmt = 'g')
    plt.savefig("Implementing AdaBoost Classifier on Data with Dummy Encoding with KNN-Imputation and Truncated SVD Feature Selection with Stratified K-Fold Cross Validation.png")
    plt.show()