"""
An ultility function helps create graphs.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction(true_name,true,pred_name,pred,result_path):
    """
    This function aims to create scatterplots for prediction of ML models 

    Parameters:
        (1) true_name: DFT or experiment
        (2) pred: a list of true values for ML validation
        (3) pred_name: ML models (GB/RF)
        (4) pred: a list of ML prediction
        (5) result_path: path leading to result directory to save results
    """

    ### PLOTTING
    fig, ax = plt.subplots(tight_layout=True, figsize=(6,5),dpi=300)

    ax.scatter(pred,true)
    ax.axline([0, 0], [180, 180], color="black", linestyle="--")
    ax.set_xlabel(pred_name,fontsize=12)
    ax.set_ylabel(true_name,fontsize=12)
    ax.set_title(pred_name + ' versus ' + true_name,fontsize=15)

    if true_name == 'Experiment':
        ax.set_xlim(20,180)
        ax.set_ylim(20,180)
    elif true_name == 'DFT':
        ax.set_xlim(-20,180)
        ax.set_ylim(-20,180)

    # setup location and save figures
    figure_dir = os.path.join(result_path, "Figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(os.path.join(figure_dir,pred_name+'_'+true_name+'_plot.png'), format='png', dpi=300)

def plot_accuracy(name,true_rank,pred_rank,result_path):
    """
    This function aims to create boxplots for evluation metrics from ML models 

    Parameters:
        (1) name: a string of the current model's name
        (2) true_rank: a list of true ranks for ML validation
        (3) pred_rank: a list of ML-predicted ranks
        (3) result_path: path leading to result directory to save results
    """
    ### PLOTTING
    size=int(max(true_rank))
    data=np.zeros([size,size])
    for i in range (size-1,-1,-1):
        for j in range (0,size):
            data[i][j]=len([x for x in range (0,len(true_rank)) if true_rank[x]==size-i and pred_rank[x]==j+1])

    fig, ax = plt.subplots(tight_layout=True, figsize=(6,5),dpi=300)
    ax.set_xlabel(name+'-predicted Descending Rank',fontsize=12)
    ax.set_ylabel('DFT-calculated Descending Rank',fontsize=12)
    ax.set_title(name+' Prediction Accuracy of Affinity Order',fontsize=15)
    pos=ax.imshow(data,cmap='Blues')

    labels = ['',1,'',2,'',3,'',4]
    ax.set_xticklabels(labels)
    labels = ['',4,'',3,'',2,'',1]
    ax.set_yticklabels(labels)
    fig.colorbar(pos,ax=ax)

    # setup location and save figures
    figure_dir = os.path.join(result_path, "Figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(os.path.join(figure_dir,name+ '_LB_Atom_Prediction.png'), format='png', dpi=300)

def plot_metric_boxplot(name,metric_df,result_path):
    """
    This function aims to create boxplots for evluation metrics from ML models 

    Parameters:
        (1) name: a string of the metric
        (2) metric_df: a dataframe of metrics extrected from the models
        (3) result_path: path leading to result directory to save results
    """

    ### PLOTTING
    fig, ax = plt.subplots(tight_layout=True, figsize=(6,5),dpi=300)

    ax.boxplot(metric_df,labels=metric_df.columns)
    ax.set_ylabel(name,fontsize=12)

    if name == 'R':
        ax.set_title('Pearson Correlation Coefficient', fontsize=15)
    elif name == 'MAE':
        ax.set_title('Mean Absolute Error', fontsize=15)

    # setup location and save figures
    figure_dir = os.path.join(result_path, "Figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(os.path.join(figure_dir,name+ '_boxplot.png'), format='png', dpi=300)

def plot_feature_importance(model, fi_df, result_path, threshold):
    """
    This function aims to create graphs for feature importances extracted 
    from ML models (Random Forest, Gradient Boosting)

    Parameters:
        (1) model: a string of the current model's name
        (2) fi_df: a dataframe of feature importance extrected from the current model
        (3) result_path: path leading to result directory to save results
        (4) threshold: threshold of minimum value to consider to plot
    """
	### COMPUTE MEAN AND STD
    fi = np.array([fi_df.columns, fi_df.mean(), fi_df.std().astype(float)]).transpose()
    fi = pd.DataFrame(fi,columns=['Descriptor','fi_mean','fi_std'])
    fi['fi_mean'] = pd.to_numeric(fi['fi_mean'])

    # select 10 most important features
    to_plot = fi.nlargest(n=threshold, columns=['fi_mean'])
    
	### PLOTTING
    fig, ax = plt.subplots(tight_layout=True, figsize=(6,5),dpi=300) 
    ax.barh(to_plot['Descriptor'],to_plot['fi_mean'],
         xerr=to_plot['fi_std'],ecolor='black',capsize=4)


    ax.set_xlabel('Feature Importance',fontsize=12)
    ax.set_title('10 most important descriptors of '+model,fontsize=15)
    
    # setup location and save figures
    figure_dir = os.path.join(result_path, "Figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(os.path.join(figure_dir,model + '_FI.png'), format='png', dpi=300)
    