import os
import matplotlib.pyplot as plt
import torch
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from plot import plot_prediction,plot_accuracy

# Setting random seeds and Ignoring warnings
warnings.simplefilter(action="ignore", category=UserWarning)
tf.random.set_seed(22)
torch.manual_seed(22)
np.random.seed(22)
random.seed(22)

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

# Loading datasets
path=os.getcwd()
train=dc.data.DiskDataset(os.path.join(path,'Data','InSilico_Dataset'))
test=dc.data.DiskDataset(os.path.join(path,'Data','Experimental_Dataset'))
exp=pd.read_csv(os.path.join(path,'Data','Experimental_Dataset','Experimental_Data.csv'))

# Training the GNN model
model = dc.models.GraphConvModel(1,mode='regression')
model.fit(train, nb_epoch=500)

# Generating predictions
train_table=train.to_dataframe()[['ids','y']]
train_table['LB']=[int(x[:x.index('_')]) for x in train_table['ids'].values.tolist()]
train_table['DFT_Rank']=train_table.groupby('LB')['y'].rank(ascending=False,method='first')
predictions=model.predict(train)
train_table['Pred']=predictions
train_table['GNN_Rank']=train_table.groupby('LB')['Pred'].rank(ascending=False,method='first')

test_table=test.to_dataframe()[['ids','y']]
test_table['LB']=[int(x[:x.index('_')]) for x in test_table['ids'].values.tolist()]
test_table['DFT_Rank']=test_table.groupby('LB')['y'].rank(ascending=False,method='first')
predictions=model.predict(test)
test_table['Pred']=predictions
test_table['GNN_Rank']=test_table.groupby('LB')['Pred'].rank(ascending=False,method='first')
exp['DFT']=[get_boltzmann(test_table[test_table['LB']==i]['y']) for i in exp['LB'].unique()]
exp['Pred']=[get_boltzmann(test_table[test_table['LB']==i]['Pred']) for i in exp['LB'].unique()]

# Getting results
result_path=os.path.join(path,'Result')

test_table.to_csv(os.path.join(result_path, "Predicted_DFT_BF3_Affinity.csv"),index=False)
exp.to_csv(os.path.join(result_path, "Predicted_Exp_BF3_Affinity.csv"),index=False)

metric=[]
metric.append(get_metrics('GNN-DFT',test_table['y'],test_table['Pred']))
metric.append(get_metrics('GNN-Exp',exp['Exp'],exp['Pred']))
metric_table=pd.DataFrame(metric,columns=['', 'MAE', 'Pearson_r'])
metric_table.to_csv(os.path.join(result_path,'GNN_Performance_Metrics.csv'), index=False)
metric_txt = metric_table.round(decimals=2)
with open(os.path.join(result_path, 'GNN_Performance_Metrics.txt'), 'w') as f:
    f.write(metric_txt.to_string(index=False))
f.close()

accuracy=[]
for name,table in [['InSilico Dataset',train_table],['Experimental Dataset',test_table]]:
    polybase = [i for i in table['LB'].unique() if len(table[table['LB']==i]['DFT_Rank'].values)>1]
    accuracy_metrics=get_accuracy(table[table['LB'].isin(polybase)]['DFT_Rank'],table[table['LB'].isin(polybase)]['GNN_Rank'])
    accuracy.append([name]+accuracy_metrics)

accuracy_table=pd.DataFrame(accuracy,columns=['', 'Affinity Order Accuracy', 'Highest-Affinity Atom Accuracy'])
accuracy_table.to_csv(os.path.join(result_path,'LB_biniding_atom_accuracy.csv'), index=False)
accuracy_txt = accuracy_table.round(decimals=2)
with open(os.path.join(result_path, 'LB_biniding_atom_accuracy.txt'), 'w') as f:
    f.write(accuracy_txt.to_string(index=False))
f.close()

# Plotting
plot_prediction('DFT',test_table['y'],'GNN',test_table['Pred'],result_path)
plot_prediction('Experiment',exp['Exp'],'GNN',exp['Pred'],result_path)
plot_accuracy('GNN',test_table[test_table['LB'].isin(polybase)]['DFT_Rank'].values.tolist(),
            test_table[test_table['LB'].isin(polybase)]['GNN_Rank'].values.tolist(),result_path)

# Cheking overfitting
# epoch=501
# mae_list,r_list=[[],[]],[[],[]]
# for i in range (10,epoch,10):
#     print('\nEpoch: ',i,'/ 500')
#     model.fit(train, nb_epoch=i)
#     mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
#     r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score)

#     print('Train MAE: ',model.evaluate(train, r2))
#     print('Test MAE: ',model.evaluate(test, r2))
#     mae_list[0].append(model.evaluate(train, mae)['mean_absolute_error'])
#     mae_list[1].append(model.evaluate(test, mae)['mean_absolute_error'])
#     r_list[0].append(np.sqrt(model.evaluate(train, r2)['pearson_r2_score']))
#     r_list[1].append(np.sqrt(model.evaluate(test, r2)['pearson_r2_score']))

# plt.plot(range(10,epoch,10),mae_list[0])
# plt.plot(range(10,epoch,10),mae_list[1])
# plt.title('MAE')
# plt.show()

# plt.plot(range(10,epoch,10),r_list[0])
# plt.plot(range(10,epoch,10),r_list[1])
# plt.title('Pearson R')
# plt.show()