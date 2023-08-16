'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd

datasets = ['MN', 'MW', 'PDI']

for dataset in datasets:
    print('====================================')
    print(dataset)
    df = pd.read_csv('../../data/interim/'+dataset+'.csv')
    n_points = df.shape[0]
    
    # Compute keys
    if dataset in ['MN', 'MW']:
        key = df['Solute'] + '_' + df['Solvent'] + '_' + df['T'].astype(str) + '_' + df['ln-'+dataset].astype(str) #+ '_' + df['Physical_state']
    else:
        key = df['Solute'] + '_' + df['Solvent'] + '_' + df['T'].astype(str) + '_' + df['ln-MW'].astype(str) + '_' + df['ln-MN'].astype(str) #+ '_' + df['Physical_state']
    
    # Check if keys are duplicated
    if dataset == 'MN':
        exps = key.value_counts()[key.value_counts() > 1].index.tolist()
        exps_clean = []
        for e in exps:
            e = e.split("_")[:3]
            e = "_".join(e)
            exps_clean.append(e)
    
    df_repeated = key.value_counts()[key.value_counts() > 1]
    indx_repeated = df_repeated.index.tolist()
    
    index_repeated_clean = []
    for i in indx_repeated:
        ix = i.split("_")[:3]
        ix = "_".join(ix)
        if ix not in exps_clean:
            index_repeated_clean.append(i)
            
    df_repeated = df_repeated.loc[index_repeated_clean]
    
    print(df_repeated)
    

