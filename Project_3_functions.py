import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import numpy as np
import datetime
import seaborn as sns

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, LabelEncoder
from visualize import generate_moons_df, preprocess, plot_boundaries

from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, plot_roc_curve, \
classification_report, mean_squared_error, r2_score, auc, \
accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve,f1_score, fbeta_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelBinarizer

from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb

"""
Summary:
Function 1: Prep data with binarization. Take in df and desired target and perform
            train test split + preprocessing for feat + binarization for target.
            Binarized data used for classification reports, scores, ROC curve
Function 2: Prep data WITHOUT binarization. Take in df and desired target and perform
            train test split + preprocessing for feat + binarization for target.
            NON-binarized data used for confusion matrix
Function 3:
Function 4:
Function 5:
Function 6:

"""



### FUNCTION 1: PREP DATA with binarization.
# Take in df and desired target and perform train test split + preprocessing for feat + binarization for target.
def data_prep_binarized_with_names(df, features, target):
    """
    Function binarizes target variable, splits feature set into train/test split,
    Inputs: dataframe, feature columns, target column.
    Outputs: Preprocessed X train and test sets, y train and test sets.
    """
    ### 1. Features, target, T/T split ###

    # a) separate features and target.
    X = df.loc[:, features]
    y = df.loc[:, target]


    # b) binarize y
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    # c) split X ------ this will be replaced by kf cv
    X_train, X_test, y_train_b, y_test_b = train_test_split(X, y_bin)

    train_names = pd.DataFrame(X_train['name'], index = X_train.index)
    test_names = pd.DataFrame(X_test['name'], index = X_test.index)

    ### 2. OHE ###
    # a) Identify categorical variables to apply OHE to
    categoricals = []
    continuous = []
    for col in X.columns:
        if col != 'name':
            if df.loc[:,col].dtypes != 'float64':
                categoricals.append(col)
            else:
                continuous.append(col)





    # b) initiate one hot encoder and fit transform to X
    ohe = OneHotEncoder(sparse=False)
    cat_matrix_train = ohe.fit_transform(X_train.loc[:, categoricals])

    # c) Turn it into a df
    X_train_ohe = pd.DataFrame(cat_matrix_train,
                        columns=ohe.get_feature_names(categoricals), #create meaningful column names
                        index=X_train.index) #keep the same index values

    # d) combine continuous and categorical data
    X_train_preprocessed = pd.concat([X_train.loc[:, continuous], X_train_ohe], axis=1)

    # e) scale train data
    ss = StandardScaler()
    X_train_preprocessed = pd.DataFrame(ss.fit_transform(X_train_preprocessed),columns = X_train_preprocessed.columns, index=X_train.index)


    # f) repeat a-d for test set
    cat_matrix_test = ohe.transform(X_test.loc[:, categoricals]) #remember to only transform on the test set!

    X_test_ohe = pd.DataFrame(cat_matrix_test,
                           columns=ohe.get_feature_names(categoricals), #create meaningful column names
                           index=X_test.index) #keep the same index values

    X_test_preprocessed = pd.concat([X_test.loc[:, continuous], X_test_ohe], axis=1)
    X_test_preprocessed = pd.DataFrame(ss.transform(X_test_preprocessed),columns = X_test_preprocessed.columns, index=X_test.index)

    print('X_train_preprocessed, X_test_preprocessed, y_train_b, y_test_b, train_names, test_names variables created')

    return X_train_preprocessed, X_test_preprocessed, y_train_b, y_test_b, train_names, test_names

### FUNCTION 2: PREP DATA without binarizing

def data_prep_NOT_binarized_with_names(df, features_with_name, target):
    """
    Function binarizes target variable, splits feature set into train/test split,
    Inputs: dataframe, feature columns, target column.
    Outputs: Preprocessed X train and test sets, y train and test sets.
    """
    ### 1. Features, target, T/T split ###

    # a) separate features and target.
    X = df.loc[:, features_with_name]
    y = df.loc[:, target]

    # need to save names here
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_names = pd.DataFrame(X_train['name'], index = X_train.index)
    test_names = pd.DataFrame(X_test['name'], index = X_test.index)

    ### 2. OHE ###
    # a) Identify categorical variables to apply OHE to
    categoricals = []
    continuous = []
    for col in X.columns:
        if col != 'name':
            if df.loc[:,col].dtypes != 'float64':
                categoricals.append(col)
            else:
                continuous.append(col)

    # b) initiate one hot encoder and fit transform to X
    ohe = OneHotEncoder(sparse=False)
    cat_matrix_train = ohe.fit_transform(X_train.loc[:, categoricals])

    # c) Turn it into a df
    X_train_ohe = pd.DataFrame(cat_matrix_train,
                        columns=ohe.get_feature_names(categoricals), #create meaningful column names
                        index=X_train.index) #keep the same index values

    # d) combine continuous and categorical data
    X_train_preprocessed = pd.concat([X_train.loc[:, continuous], X_train_ohe], axis=1)

    # e) scale train data
    ss = StandardScaler()
    X_train_preprocessed = pd.DataFrame(ss.fit_transform(X_train_preprocessed),columns = X_train_preprocessed.columns, index=X_train.index) #keep the same index values


    # f) repeat a-d for test set
    cat_matrix_test = ohe.transform(X_test.loc[:, categoricals]) #remember to only transform on the test set!

    X_test_ohe = pd.DataFrame(cat_matrix_test,
                           columns=ohe.get_feature_names(categoricals), #create meaningful column names
                           index=X_test.index) #keep the same index values

    X_test_preprocessed = pd.concat([X_test.loc[:, continuous], X_test_ohe], axis=1)
    X_test_preprocessed = pd.DataFrame(ss.transform(X_test_preprocessed),columns = X_test_preprocessed.columns, index=X_test.index) #keep the same index values




    print('X_train_preprocessed, X_test_preprocessed, y_train, y_test, train_names, test_names variables created')

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, train_names, test_names

### FUNCTION 3: get_classification_reports
    # cannot do KF CV on this because outputs are long strings and cannot be averaged across folds
def get_classification_reports(models, model_names, X_train, X_test, y_train, y_test):
    """
    Inputs: list of fit models, list of model names, training data, and testing data. Note: data should be pre-processed (scaled etc).
    Outputs a classification report for specified model. Models are fit to training set and scored on testing set.
    Note: requires binarized data.
    """

    for name, model in list(zip(model_names, models)):
        print(name)
        print('\n')

        # Accounting for one vs rest target data
        if y_train.shape[1] != 1:
            model = OneVsRestClassifier(model)

        # fit model
        if model == xgb:
            xgb_param = model.get_xgb_params()
            xgb_param['num_class'] = 3
        model.fit(X_train, y_train)

        # Predicting on test set
        y_predict = model.predict(X_test)

        # Classification report for model: Precision, recall, F1-score, support for each class,
        # as well as averages for these metrics.
        print(classification_report(y_test, y_predict))

        print('-'*60)
        print('\n')

### FUNCTION 4: get model scores
def get_scores(models, model_names, X_train, X_test, y_train, y_test):
    ## run model to get class reports
    # requires binarized data
    for model, name in list(zip(models, model_names)):
        print(name)
        print('\n')

        # Accounting for one vs rest target data
        if y_train.shape[1] != 1:
            model = OneVsRestClassifier(model)

        # fit model
        if model == xgb:
            xgb_param = model.get_xgb_params()
            xgb_param['num_class'] = 3
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)


        accuracies_across_models, precisions_across_models, recalls_across_models, f1s_across_models = [], [], [], []

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        accuracies_across_models.append(acc)
        precisions_across_models.append(pre)
        recalls_across_models.append(rec)
        f1s_across_models.append(f1)

        print(f'Accuracy:  {np.mean(acc):.3f} +- {np.std(acc):3f}')
        print(f'Precision:  {np.mean(pre):.3f} +- {np.std(pre):3f}')
        print(f'Recall:  {np.mean(rec):.3f} +- {np.std(rec):3f}')
        print(f'f1-score:  {np.mean(f1):.3f} +- {np.std(f1):3f}')

        print('-'*60)
        print('\n')

### FUNCTION 5: get model scores with KF cv
# generates non-binarized data
# data cleaning is included in this function because Kf CV takes the place of train-test split, and T/T is part of other data cleaning functions.
def process_and_get_scores_CV(models, model_names, X, y):
    """
    Inputs: unprocessed X, y

    Outputs: Average accuracy, precision, recall and f1 scores across 5 folds for each model.
    """
    accuracy_dict = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    dics = [accuracy_dict, precision_dict, recall_dict, f1_dict]

    ### Splitting into K folds
    kf = KFold(n_splits=5, shuffle=True, random_state = 71)

    fold_iteration = 1

    for train_ind, val_ind in kf.split(X,y): # 5 folds
        X_train, y_train = X.loc[train_ind], y.loc[train_ind]
        X_val, y_val = X.loc[val_ind], y.loc[val_ind]

        print('\n')
        print('-'*36)
        print('\tBeginning of fold: {}'.format(fold_iteration))
        print('-'*36)
        print('\n')

        ### PREPROCESSING ###
        # a) Identify categorical variables to apply OHE to

        categoricals = []
        continuous = []
        for col in X_train.columns:
            if X_train.loc[:,col].dtypes != 'float64':
                categoricals.append(col)
            else:
                continuous.append(col)

        # b) initiate one hot encoder and fit transform to X
        ohe = OneHotEncoder(sparse=False)
        cat_matrix_train = ohe.fit_transform(X_train.loc[:, categoricals])

        # c) Turn it into a df
        X_train_ohe = pd.DataFrame(cat_matrix_train,
                            columns=ohe.get_feature_names(categoricals), #create meaningful column names
                            index=X_train.index) #keep the same index values

        # d) combine continuous and categorical data
        X_train_preprocessed = pd.concat([X_train.loc[:, continuous], X_train_ohe], axis=1)

        # e) scale train data
        ss = StandardScaler()

        X_train_preprocessed = ss.fit_transform(X_train_preprocessed)


        # f) repeat a-d for test set
        cat_matrix_test = ohe.transform(X_val.loc[:, categoricals]) #remember to only transform on the test set!

        X_val_ohe = pd.DataFrame(cat_matrix_test,
                               columns=ohe.get_feature_names(categoricals), #create meaningful column names
                               index=X_val.index) #keep the same index values

        X_val_preprocessed = pd.concat([X_val.loc[:, continuous], X_val_ohe], axis=1)
        X_val_preprocessed = ss.transform(X_val_preprocessed)


        ### Looping through models
        for model, name in list(zip(models, model_names)):
            for dic in dics:
                if name not in dic:
                    dic[name] = []

            print('model: {}'.format(name))

            # fit model
            if model == xgb:
                xgb_param = model.get_xgb_params()
                xgb_param['num_class'] = 3
            model.fit(X_train_preprocessed, y_train)

            y_pred = model.predict(X_val_preprocessed)

            acc = accuracy_score(y_val, y_pred) # one model, one fold
            pre = precision_score(y_val, y_pred, average="weighted")
            rec = recall_score(y_val, y_pred, average="weighted")
            f1 = f1_score(y_val, y_pred, average="weighted")

            # Add scores to respective model keys in dicts
            accuracy_dict[name].append(acc)
            precision_dict[name].append(pre)
            recall_dict[name].append(rec)
            f1_dict[name].append(f1)

        fold_iteration+=1

    print('\n')
    for model, name in list(zip(models, model_names)):

        print(name)
        print(f'accuracy across folds: {np.mean(accuracy_dict[name]):.3f} +- {np.std(accuracy_dict[name]):.3f}')
        print(f'precision across folds: {np.mean(precision_dict[name]):.3f} +- {np.std(precision_dict[name]):.3f}')
        print(f'recall across folds: {np.mean(recall_dict[name]):.3f} +- {np.std(recall_dict[name]):.3f}')
        print(f'f1 across folds: {np.mean(f1_dict[name]):.3f} +- {np.std(f1_dict[name]):.3f}')
        print('\n')

### FUNCTION 5: get confusion matrix
def get_confusion_matrix(models, model_names, X_train, X_test, y_train, y_test):
    #requires non-binarized data

    for model, name in list(zip(models, model_names)):

        # Multiple classes?
        if y_train.shape[1] != 1:
            model = OneVsRestClassifier(model)

        model.fit(X_train, y_train)

        fig, ax = plt.subplots(figsize=(7, 7))
        plot_confusion_matrix(model, X_test, y_test, ax=ax)
        plt.title(name, size = 15)

### FUNCTION 6: generate ROC curve
def get_ROC_curve(models, model_names, X_train, X_test, y_train, y_test):
    """
    Inputs: list of fit models, list of model names, training data, and testing data. Note: data should be pre-processed (scaled etc).
    Outputs a graph of n ROC curves for each model, where n=num classes. Models are fit to training set and scored on testing set.
    Note: requires binarized data.
    """
    for model, name in list(zip(models, model_names)):
        # print(name)
        # print('\n')

        # Multiple classes?
        if y_train.shape[1] != 1:
            model = OneVsRestClassifier(model)


        # fit model
        if model == xgb:
            xgb_param = model.get_xgb_params()
            xgb_param['num_class'] = 3
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Get class probabilities
        probs = model.predict_proba(X_test)

        # getting rates by class
        n_classes = y_train.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            classes = ['Bad', 'Good', 'Neutral']
            plt.plot(fpr[i], tpr[i], label="{}, area: {}".format(classes[i], round(roc_auc[i],2)))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by class for {}'.format(name))
        plt.legend()
        plt.show()
