'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

colors_mpi = [
    '#33A5C3',
    '#87878D',
    '#007675',
    '#78004B',
    '#383C3C',
    '#ECE9D4'
    ]

colors_mpi = [
    '#33A5C3',
    '#87878D',
    '#007675',
    ]

dataset_label = {'MN':'MN', 'MW':'MW', 'PDI':'MN/MW'}

def error_density_plot(dataset, mode, rep):
    
    methods = ['RF', 'GHGNN', 'GHGNN_pss']
    methods_name = ['Random Forest', 'GH-GNN', 'GH-GNN (pss)']
    colors = colors_mpi
    
    fig = plt.figure(figsize=(7.5, 5.625))
    error_lst = []
    model_col = []
    for jj, method in enumerate(methods):
        keys_all = []
        y_true_all = []
        y_pred_all = [] 
        
        for i in range(10):
            path = '../../models/'+method+'/'+mode+'/split_'+str(i)+'/'
            df = pd.read_csv(path + dataset+'_test_pred_'+method+'.csv')
            
            y_true = df['ln-omega'].tolist()
            y_pred = df[method+'_'+rep].tolist()
            if dataset == 'PDI':
                keys = df['Solute'] + df['Solvent'] + df['T'].astype(str) + df['ln-MW'].astype(str) + df['ln-MN'].astype(str)
                keys = keys.tolist()
            else:
                keys = df['Solute'] + df['Solvent'] + df['T'].astype(str) + df['ln-'+dataset].astype(str)
                keys = keys.tolist()
            
            keys_all.extend(keys)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
        
        df_pred = pd.DataFrame({
            'key':keys_all,
            'y_true':y_true_all,
            'y_pred':y_pred_all
            })
        df_avg = df_pred.groupby('key').mean().reset_index()
        
        y_true = df_avg['y_true'].to_numpy()
        y_pred = df_avg['y_pred'].to_numpy()
        error_method = y_true-y_pred
        
        error_lst.extend(error_method)
        model_col.extend([methods_name[jj]]*len(error_method))
        
    
    df_error = pd.DataFrame({
        'Model':model_col,
        'Error':error_lst
        })
    
    ax = plt.gca()
    
    sns.histplot(data=df_error, x='Error', hue='Model', ax=ax, 
                 stat='density', element='step', kde=True, common_norm=False,
                 palette=colors, alpha=0.3, linewidth=1)
    ax.set_xlim(-1.5,1.5)
    ax.set_xlabel('$\Delta ln(\Omega_{ij}^\infty)$', fontsize=15)
    ax.set_ylabel('Error density', fontsize=15)
    ax.axvline(color='#878d91', ls='--')
    plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
    plt.title(dataset_label[dataset], fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout() 
    plt.close(fig)
    
    # Save plot
    path = '../../reports/figures/error_density_plots/'+mode
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig.savefig(path+'/'+dataset+'_'+rep+'.png', dpi=350) 
    

for dataset in ['MN', 'MW', 'PDI']:
    for mode in ['interpolation', 'extrapolation']:
        for rep in ['monomer', 'ru_wo', 'ru_w', 'oligomer_10']:
            error_density_plot(dataset, mode, rep)




