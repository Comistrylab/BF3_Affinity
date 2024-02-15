"""
This module aims to train data with a number of Machine Learning algorithms

"""

# IMPPORT DATA-PROCESSING LIBs
import math
from models import get_models
from plot import plot_feature_importance, plot_metric_boxplot, plot_prediction, plot_accuracy

import pandas as pd
import numpy as np

# IMPORT EVALUATION METHODS
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
# IMPORT METRICS
from sklearn.metrics import mean_absolute_error, make_scorer
from scipy.stats import pearsonr

# IMPORT UTIL MODULES
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

def pearsoncv(y_true, y_pred):
    pearsoncv = pearsonr(y_true, y_pred)
    return pearsoncv[0]

# Custom splitting algorithm
class CustomSplitting:
    @classmethod
    def split(cls,
              X: pd.DataFrame,
              y: np.ndarray,
              n_splits: int=5, 
              n_repeats: int=6):

            skf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=22)
            labels = pd.cut(y, n_splits, labels=False)
        
            for i, (train_index, test_index) in enumerate(skf.split(X, labels)):
                yield np.asarray(train_index), np.asarray(test_index)

def cross_validation(X_train, y_train, feature_names, result_path):
    """
    This function aims to evaluate metrics of Machine Learning algorithms using
    Cross Validation.
    For each models, it will compute 2 metrics: Pearson R and MAE.

    The results are exported as CSV files and saved in result directory.
    Some figures are also generated in result directory.

    Inputs: (1) X_train, y_train, feature_names: data
            (2) result_path: path leading to result directory to save results.
    """
    models = get_models()
    names = list()
    feature_importances = pd.DataFrame()
    mae_cv_results, r_cv_results = list(), list()

    # SETUP CROSS VALIDATION
    scoring = {'mae':'neg_mean_absolute_error',
               'pr_score': make_scorer(pearsoncv)}

    for name, model in models:  # for each model in the list
        # apply cross validation method
        cv = CustomSplitting.split(X=X_train, y=y_train)
        this_cv_scores = cross_validate (model, X_train, y_train, cv=cv,
                                        scoring=scoring, n_jobs=-1,
                                        error_score='raise',
                                        return_estimator=True)

        # save results to lists
        mae_cv_results.append((this_cv_scores['test_mae']))
        r_cv_results.append(this_cv_scores['test_pr_score'])

        names.append(name)  # get the name of models

        # GET FEATURE IMPORTANCES
        for idx, estimator in enumerate(this_cv_scores['estimator']):
            if name in ['RF', 'GB']:
                this_fi = pd.DataFrame([estimator.feature_importances_], index=[idx],
                                       columns=feature_names)

                this_fi['Model'] = [name]
                feature_importances = pd.concat([feature_importances, this_fi])

    # EXPORT CV METRICS AS CSV FILES
    result_path=os.path.join(result_path,"Cross_validation")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    mae_cv_results = np.transpose(mae_cv_results)
    mae_cv_results = pd.DataFrame(mae_cv_results, columns=names)
    mae_cv_results.to_csv(os.path.join(result_path, "MAE_cv_result.csv"),
                           index=False)

    r_cv_results = np.transpose(r_cv_results)
    r_cv_results = pd.DataFrame(r_cv_results, columns=names)
    r_cv_results.to_csv(os.path.join(result_path, "R_cv_result.csv"),
                        index=False)

    # construct graphs
    plot_metric_boxplot('MAE',mae_cv_results,result_path)
    plot_metric_boxplot('R',r_cv_results,result_path)

    # EXPORT FEATURE IMPORTANCES OF CV
    for model in ['RF', 'GB']:
        # select rows and columns corresponding with current model
        this_model_fi = feature_importances[feature_importances['Model'] == model]
        this_model_fi = this_model_fi.loc[:, this_model_fi.columns != 'Model']

        # setup csv file path and export
        file_path =  "FeatureImportance_" + model + ".csv"
        this_model_fi.to_csv(os.path.join(result_path, file_path),
                             index=False)    

        # construct graphs
        plot_feature_importance(model, this_model_fi, result_path, 10)

# Metric functions
def get_boltzmann(values):
    values=np.array(values)
    prob=np.exp(values/2.479)
    prob=prob/np.sum(prob)
    energy=np.sum(np.multiply(values,prob))
    return energy

def get_accuracy(y_test,y_pred):
    affinity_order=0
    best_atom=0
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
    for i in range (0,len(y_test)):
        if y_test[i]==y_pred[i]:
            affinity_order+=1
            if y_test[i]==1:
                best_atom+=1
    affinity_order=affinity_order/len(y_test)
    best_atom=best_atom/np.count_nonzero(y_test == 1)
    return [affinity_order,best_atom]

def get_metrics(name, y_test, y_pred):
    MAE = mean_absolute_error(y_test, y_pred)
    Pearson_r, p_value = pearsonr(y_test, y_pred)
    metrics = [name, MAE, Pearson_r]
    return metrics

def train_test_validate(test, X_train, y_train, X_test, y_test, exp_table, result_path):
    """
    This function aims to evaluate Machine Learning model's prediction
    power. The evaluation is metrics computed by using get_metrics and 
    get_accuracy helper function. The metrics and the predicted values 
    of y are exported as CSV files and saved in result directory.
    Some figures are also generated in result directory.

    Inputs: (1) test, X_train, y_train, X_test, y_test, y_exp: data
            (2) result_path: path leading to result directory to save results.
    """
    models = get_models()

    ML_DFT_metrics = list()
    ML_Exp_metrics = list()
    test_table = test.iloc[:, :5]
    exp_table['DFT_BF3_Affinity']=[get_boltzmann(test_table[test_table['LB']==i]['DFT_BF3_Affinity']) for i in exp_table['LB'].unique()]

    for name, model in models:
        if name in ['GB','RF']:
            fitting = model.fit(X_train, y_train)  # train and fit the model
            y_pred = fitting.predict(X_test)
            test_table[name+'_BF3_Affinity']=y_pred
            exp_table[name+'_BF3_Affinity']=[get_boltzmann(test_table[test_table['LB']==i][name+'_BF3_Affinity']) for i in exp_table['LB'].unique()]

            ML_DFT_metrics.append(get_metrics(name, y_test, y_pred))
            ML_Exp_metrics.append(get_metrics(name, exp_table['Exp_BF3_Affinity'], exp_table[name+'_BF3_Affinity']))

    #EXPERIMENTAL VALUES AND BING ATOM VALIDATION
    for model in ['DFT','GB','RF']:
        test_table[model+'_Rank'] = test_table.groupby('LB')[model+'_BF3_Affinity'].rank(ascending=False,method='first')
    
    LB_biniding_atom_accuracy = list()

    for model in ['GB','RF']:
        polybase = [i for i in exp_table['LB'].unique() if len(test_table[test_table['LB']==i]['DFT_Rank'].values)>1]
        accuracy=get_accuracy(test_table[test_table['LB'].isin(polybase)]['DFT_Rank'],test_table[test_table['LB'].isin(polybase)][model+'_Rank'])
        LB_biniding_atom_accuracy.append([model]+accuracy)

    # SAVE AND EXPORT FINAL RESULT
    result_path=os.path.join(result_path,"Train_test_validation")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for name, metrics in [['ML_DFT_metrics', ML_DFT_metrics],
                          ['ML_Exp_metrics', ML_Exp_metrics],
                          ['LB_biniding_atom_accuracy', LB_biniding_atom_accuracy]]:
        if name == 'LB_biniding_atom_accuracy':
            metric_df = pd.DataFrame(metrics, columns=['', 'Affinity Order Accuracy', 'Highest-Affinity Atom Accuracy'])
        else:
            metric_df = pd.DataFrame(metrics, columns=['', 'MAE', 'Pearson_r'])

        metric_path = os.path.join(result_path, name+".csv")
        metric_df.to_csv(metric_path, index=False)  # save as .csv file

        metric_txt = metric_df.round(decimals=2)  # save as .txt file
        with open(os.path.join(result_path, name+".txt"), 'w') as f:
            f.write(metric_txt.to_string(index=False))

    # SAVE ALL PREDICTED DATA	(TEST SET)
    test_table.iloc[:,:-3].to_csv(os.path.join(result_path, "All_DFT_BF3_Affinity.csv"), index=False)
    exp_table.to_csv(os.path.join(result_path, "All_Exp_BF3_Affinity.csv"), index=False)
    
    #PLOTTING
    for model in ['GB','RF']:
        plot_prediction('DFT',test_table['DFT_BF3_Affinity'],model,test_table[model+'_BF3_Affinity'],result_path)
        plot_prediction('Experiment',exp_table['Exp_BF3_Affinity'],model,exp_table[model+'_BF3_Affinity'],result_path)
        plot_accuracy(model,test_table[test_table['LB'].isin(polybase)]['DFT_Rank'].values.tolist(),
                      test_table[test_table['LB'].isin(polybase)][model+'_Rank'].values.tolist(),result_path)