'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import os

colors_mpi = [
    '#33A5C3',
    '#87878D',
    '#007675',
    ]

def parity_plot(dataset, mode, rep):
    
    methods = ['RF', 'GHGNN', 'GHGNN_pss']
    methods_name = ['Random Forest', 'GH-GNN', 'GH-GNN (pss)']
    colors = colors_mpi
    fig = plt.figure(figsize=(7.5, 5.625))
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
        
        # plot
        plt.plot([np.min(y_true), np.max(y_true)],
                  [np.min(y_true), np.max(y_true)], '--', c='0.5')
        plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
        plt.scatter(y_true, y_pred, s=40, edgecolors='k', alpha=0.5, 
                    label=methods_name[jj], marker='o', color=colors[jj], linewidth=0.5)
           
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        text_content = f"MAE: {mae:.2f}\nR2: {r2:.2f}"
        plt.text(3 + 1.2*jj, np.min(y_true), text_content, 
                  bbox=dict(facecolor='white', alpha=1), fontsize=12, color=colors[jj])
            
    plt.legend(loc=2, fontsize=12)
    plt.xlabel('Experimental $ln(\Omega_{ij}^{\infty})$', fontsize=15)
    plt.ylabel('Predicted $ln(\Omega_{ij}^{\infty})$', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.close(fig)
    
    # Save plot
    path = '../../reports/figures/parity_plots/'+mode
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig.savefig(path+'/'+dataset+'_'+rep+'_parity.png', dpi=350)
    

for dataset in ['MN', 'MW', 'PDI']:
    for mode in ['interpolation', 'extrapolation']:
        for rep in ['monomer', 'ru_wo', 'ru_w', 'oligomer_10']:
            parity_plot(dataset, mode, rep)









