'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

poly_names = {
    'LDPE':'Polyethylene, low-density',
    'PE':'Polyethylene',
    'PIB':'Polyisobutylene',
    'PVAc':'Poly(vinyl acetate)',
    'PBD':'Polybutadiene',
    'PMMA':'Poly(methyl methacrylate)',
    'PEMA':'Poly(ethyl methacrylate)',
    'PBMA':'Poly(n-butyl methacrylate)',
    'PS':'Polystyrene',
    'PMA':'Poly(methyl acrylate)',
    'PEO':'Poly(ethylene oxide)' 
    }

solv_names = {
    'c-C6':'Cyclohexane',
      'C6':'Hexane',
      'C7':'Heptane',
      'C8':'Octane',
      'C4':'Butane',
      '3-C6':'Hexane, 3-methyl',
      '2-C7':'Heptane, 2-methyl',
      '3-C7':'Heptane, 3-methyl',
      '2,4-C6':'Hexane, 2,4-dimethyl',
      '2,5-C6':'Hexane, 2,5-dimethyl',
      '3,4-C6':'Hexane, 3,4-dimethyl',
      'C9':'Nonane',
      'C10':'Decane',
      'C12':'Dodecane',
      '2,2,4-C5':'Pentane, 2,2,4-trimethyl',
      'C5':'Pentane',
      'ethyl acetate':'Acetic acid, ethyl ester',
      'methyl acetate':'Acetic acid, methyl ester',
      'butyl acetate':'Acetic acid, butyl ester',
      'MEK':'2-Butanone',
      'MiBK':'2-Pentanone, 4-methyl',
      'acetone':'2-Propanone',
      'chlorobutane':'Butane, 1-chloro',
      'dichloromethane':'Methane, dichloro',
      'chlorobenzene':'Benzene, chloro',
      'carbontetrachloride':'Methane, tetrachloro',
      'benzene':'Benzene',
      'toluene':'Toluene',
      'butanol':'1-Butanol',
      'propanol':'1-Propanol',
      'ethanol':'Ethanol',
      '2-propanol':'2-Propanol',
      '1-propanol':'1-Propanol',
      'methanol':'Methanol',
      'acetic acid':'Acetic acid'
    }

def rounding(x):
    return np.round(x,5)

# Get all systems in test averaging duplicates

def get_predictions(mode, rep, model, dataset):
    
    keys_all = []
    y_true_all = []
    y_pred_all = [] 
    
    for i in range(10):
        path = '../../models/'+model+'/'+mode+'/split_'+str(i)+'/'
        df = pd.read_csv(path + dataset + '_test_pred_' + model + '.csv')
        
        y_true = df['ln-omega'].tolist()
        y_pred = df[model+'_'+rep].tolist()
        
        if dataset in ['MN', 'MW']:
            keys = df['Solute'] + '_' + df['Solvent'] + '_' + df['T'].astype(str) + '_' + df['ln-'+dataset].astype(str)
        elif dataset == 'PDI':
            keys = df['Solute'] + '_' + df['Solvent'] + '_' + df['T'].astype(str) + '_' + df['ln-MN'].astype(str) + \
                '_' + df['ln-MW'].astype(str)
        
        keys_all.extend(keys.tolist())
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    
    y_true_all = np.exp(np.array(y_true_all))
    y_pred_all = np.exp(np.array(y_pred_all))
    
    df_pred = pd.DataFrame({
        'key':keys_all,
        'y_true':y_true_all,
        'y_pred':y_pred_all
        })
    # Get ensemble model predictions
    df_avg = df_pred.groupby('key').mean().reset_index()
    df_avg['mape'] = 100 * np.abs(df_avg['y_true'].to_numpy() - df_avg['y_pred'].to_numpy())/df_avg['y_true'].to_numpy()
    
    if dataset in ['MN', 'MW']:
        df_avg[['Solute', 'Solvent', 'T', 'ln-'+dataset]] = df_avg['key'].str.split('_', expand=True)
        df_avg['new_key'] = df_avg['Solute'] + '_' + df_avg['Solvent'] + '_' + df_avg['ln-'+dataset].astype(float).apply(rounding).astype(str)
    elif dataset == 'PDI':
        df_avg[['Solute', 'Solvent', 'T', 'ln-MN', 'ln-MW']] = df_avg['key'].str.split('_', expand=True)
        df_avg['new_key'] = df_avg['Solute'] + '_' + df_avg['Solvent'] + '_' + df_avg['ln-MN'].astype(float).apply(rounding).astype(str) + \
            '_' + df_avg['ln-MW'].astype(float).apply(rounding).astype(str)
    
    return df_avg

def get_pappa_df(system):
    
    df = pd.read_csv('../../data/external/'+system+'_pappa1999.csv')
    
    df['Solute'] = df['Solute'].map(poly_names)
    df['Solvent'] = df['Solvent'].map(solv_names)
    
    df['ln-MN'] = np.log(df['MN'].to_numpy())
    df['new_key_MN'] = df['Solute'] + '_' + df['Solvent'] + '_' + df['ln-MN'].apply(rounding).astype(str)
    
    df['ln-MW'] = np.log(df['MW'].to_numpy())
    df['new_key_MW'] = df['Solute'] + '_' + df['Solvent'] + '_' + df['ln-MW'].apply(rounding).astype(str)
    
    df['new_key_PDI'] = df['Solute'] + '_' + df['Solvent'] + '_' + df['ln-MN'].apply(rounding).astype(str) + \
        '_' + df['ln-MW'].apply(rounding).astype(str)
    
    return df

def get_intersection_dfs(mode, rep, model):

    dfs_dict = {}
    for dataset in ['MN', 'MW', 'PDI']:
        df_gnn = get_predictions(mode, rep, model, dataset)
    
        all_dfs = []
        for sys in ['athermal', 'polar', 'associated']:
            df_pappa = get_pappa_df(sys)
            sub_df = df_gnn[df_gnn['new_key'].isin(df_pappa['new_key_'+dataset])]
        
            # Check that I have the exact points within the T range
            not_consider_idxs = []
            for row in range(sub_df.shape[0]):
                key = sub_df['new_key'].iloc[row]
                
                df_pappa_spec = df_pappa[df_pappa['new_key_'+dataset] == key]
                
                if df_pappa_spec.shape[0] != 1:
                    raise Exception('Multiple Ts in Pappa')
                
                T1 = df_pappa_spec['T1'].iloc[0]
                T2 = df_pappa_spec['T2'].iloc[0]
                
                if np.isnan(T2):
                    T2 = np.inf
                
                # Check if within temperature
                T = float(sub_df['T'].iloc[row])
                
                if T >= T1 and T <= T2:
                    pass
                else:
                    not_consider_idxs.append(row)
                
            # Eliminate systems that are not considered
            sub_df = sub_df.reset_index().drop(not_consider_idxs)
            sub_df['sys'] = [sys]*sub_df.shape[0]
            all_dfs.append(sub_df)
                        
        df_combined = pd.concat(all_dfs)
        dfs_dict[dataset] = df_combined
    return dfs_dict

def get_non_empty(row):
    if row['MAPE_MN']:
        return row['MAPE_MN']
    elif row['MAPE_PDI']:
        return row['MAPE_PDI']
    elif row['MAPE_MW']:
        return row['MAPE_MW']
    else:
        return None

def get_performance_vs_mechanistic(mode, rep, model):
    
    path = '../../models/'+model+'/mechanistic_vs_model/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    dfs_dict = get_intersection_dfs(mode, rep, model)
    
    # Write prediction performance next to Pappa's ones
    for sys in ['athermal', 'polar', 'associated']:
        df_pappa = get_pappa_df(sys)
        for dataset in ['MN', 'MW', 'PDI']:
            df_gnn = dfs_dict[dataset]
            df_gnn = df_gnn[df_gnn['sys'] == sys]
            
            # Fill each row with MAPE from model
            mapes_gnn = []
            for row in range(df_pappa.shape[0]):
                key = df_pappa['new_key_'+dataset].iloc[row]
                
                df_gnn_spec = df_gnn[df_gnn['new_key'] == key]
                
                if df_gnn_spec.shape[0] > 0:
                    mapes_gnn.append(np.mean(df_gnn_spec['mape'].to_numpy()))
                else:
                    mapes_gnn.append('')
                
            df_pappa['MAPE_'+dataset] = mapes_gnn
        df_pappa['MAPE_PDI_MW_MN'] = df_pappa.apply(get_non_empty, axis=1)
        
        # Eliminate unnecesary columns
        columns_to_eliminate = ['ln-MN', 'ln-MW', 'new_key_MN', 'new_key_MW', 'new_key_PDI']
        df_pappa = df_pappa.drop(columns=columns_to_eliminate)
        
        # Save csv
        df_pappa.to_csv(path + mode +'_'+ rep + '_' + sys + '_' + 'mechanistic_vs_model.csv', index=False)
            
                
                
    # for dataset in ['MN', 'MW', 'PDI']:
        
    #     df_combined = dfs_dict[dataset]
    
    #     perform_dict = {}
    #     for sys in ['athermal', 'polar', 'associated']:
    #         perform_dict[sys] = {}
    #         df_spec = df_combined[df_combined['sys'] == sys]
    #         df_pappa = get_pappa_df(sys)
            
    #         df_spec.to_csv(path + '/ghgnn_'+mode+'_'+sys+'.csv', index=False)
    #         df_pappa.to_csv(path + '/pappa_'+mode+'_'+sys+'.csv',index=False)
            
    #         mapes_pappa = []
    #         mapes_ours = []
    #         points_lst = []
            
    #         for row in range(df_pappa.shape[0]):
    #             n_points_pappa = df_pappa['n_points'].iloc[row]
    #             key = df_pappa['new_key'].iloc[row]
                
    #             df_spec_spec = df_spec[df_spec['new_key'] == key]
    #             n_points_ours = df_spec_spec.shape[0]
                
    #             if n_points_pappa == n_points_ours:
    #                 mapes_pappa.append(df_pappa[sys_mapes_pappa[sys]].iloc[row])
    #                 mapes_ours.append(np.mean(df_spec_spec['mape'].to_numpy()))
    #                 points_lst.append(n_points_ours)
    #         print(f'\n   {sys}')
    #         print('      Points: ', np.sum(points_lst))
    #         print('    MAPE (ours): ', np.round(np.mean(mapes_ours), 1))
    #         print('    MAPE (pappa): ', np.round(np.mean(mapes_pappa), 1))  
            
    #         perform_dict[sys]['ghgnn'] = np.mean(mapes_ours)
    #         perform_dict[sys]['pappa'] = np.mean(mapes_pappa)
    # return perform_dict
                    

def plot_mechanistic_vs_ghgnn(results_dict, mode):
    
    path = '../../reports/figures/mechanistic_vs_ghgnn'
    if not os.path.exists(path):
        os.makedirs(path)
    
    
    systems = ['athermal', 'polar', 'associated']
    systems_names = ['Athermal', 'Polar', 'Associated']
    method_names = ['UNIFAC-ZM', 'Entropic-FV', 'GH-GNN (pss)']
    
    fig, axes = plt.subplots(nrows=1, ncols=len(systems), 
                              sharey=True, 
                              squeeze=False,
                              figsize=(12,4))
    
    fig.subplots_adjust(wspace=0, hspace=0)  
    
    for i, sys in enumerate(systems):
        
        ax = axes.flat[i]
        if i == 0:
            ax.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=11)
        
        mapes = results_dict[sys]
        
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        ax.bar(method_names, mapes, color='#36454F', zorder=2)
        ax.set_xlabel(systems_names[i], fontsize=11)
        
        # Rotate x-tick labels by 45 degrees
        ax.set_xticklabels(method_names, rotation=0, fontsize=8)
    
    plt.ylim(0,35)
    
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(path + '/mape_mechanistic_ghgnn_'+mode+'.png', dpi=350)

# Count how many points in Pappa et al. 1999 are in our datasets (MN, MW and PDI)

print('='*100)
print('Count how many points I have from the points of Pappa et al. 1999')

df_MN = pd.read_csv('../../data/interim/MN.csv')
df_MW = pd.read_csv('../../data/interim/MW.csv')
df_PDI = pd.read_csv('../../data/interim/PDI.csv')


df_MN['new_key'] = df_MN['Solute'] + '_' + df_MN['Solvent'] + '_' + df_MN['ln-MN'].apply(rounding).astype(str)
df_MW['new_key'] = df_MW['Solute'] + '_' + df_MW['Solvent'] + '_' + df_MW['ln-MW'].apply(rounding).astype(str)
df_PDI['new_key'] = df_PDI['Solute'] + '_' + df_PDI['Solvent'] + '_' + \
    df_PDI['ln-MN'].apply(rounding).astype(str) + '_' + df_PDI['ln-MW'].apply(rounding).astype(str)


dfs_dict = {'MN':df_MN,
            'MW':df_MW,
            'PDI':df_PDI}


for sys in ['athermal', 'polar', 'associated']:
    counting_points = 0
    df_pappa = get_pappa_df(sys)
    
    row_lst = []
    for row in range(df_pappa.shape[0]):
        # Get number of points
        n_points_pappa = df_pappa['n_points'].iloc[row]
        
        for dataset in ['MN', 'MW', 'PDI']:
            key = df_pappa['new_key_'+dataset].iloc[row]
            df_spec = dfs_dict[dataset]
        
            df_spec_spec = df_spec[df_spec['new_key'] == key]
        
            # Check temperature
            T1 = df_pappa['T1'].iloc[row]
            T2 = df_pappa['T2'].iloc[row]
            if np.isnan(T2):
                T2 = np.inf
        
            df_spec_spec = df_spec_spec[(df_spec_spec['T'] >= T1) & (df_spec_spec['T'] <= T2)]
        
            n_points_ours = df_spec_spec.shape[0]
        
            if n_points_pappa == n_points_ours:
                if row not in row_lst:
                    row_lst.append(row)
                    counting_points += n_points_ours
               
    missing_rows = [row for row in list(range(df_pappa.shape[0])) if row not in row_lst]
    print(f'\n   {sys}')
    print('      Points: ', counting_points)
    print('      Missing rows:', np.array(missing_rows)+2)
    
print('='*100)    

# ===========================================

modes = [
    'interpolation', 
    'extrapolation'
    ]
rep = 'ru_w'
model = 'GHGNN_pss'
for mode in modes:
    print(mode)
    print('*'*70)
    get_performance_vs_mechanistic(mode, rep, model)
    
    results_dict = {}
    for sys in ['athermal', 'polar', 'associated']:
        print(sys)
        print('-'*40)
        path = '../../models/'+model+'/mechanistic_vs_model/' + mode + \
            '_'+ rep + '_' + sys + '_' + 'mechanistic_vs_model.csv'
        df = pd.read_csv(path)
        
        if sys in ['polar', 'associated']:
            extra_label = '_dependent'
        else:
            extra_label = ''
        
        for dataset in ['MN', 'MW', 'PDI', 'PDI_MW_MN']:
            print('  Dataset: ', dataset)
            df_spec = df[df['MAPE_'+dataset].notna()]
            
            n_points = df_spec['n_points'].sum()
            mape_unifac_zm = df_spec['MAPE_UNIFAC-ZM'+extra_label].mean()
            mape_entropic_fv = df_spec['MAPE_Entropic-FV'+extra_label].mean()
            mape_ours = df_spec['MAPE_'+dataset].mean()
            
            print('    n_points   : ', n_points)
            print('    UNIFAC-ZM  : ', np.round(mape_unifac_zm,1))
            print('    Entropic-FV: ',  np.round(mape_entropic_fv,1))
            print('    GHGNN_pss  : ',  np.round(mape_ours,1))
            print(' ')
            
        results_dict[sys] = [mape_unifac_zm, mape_entropic_fv, mape_ours]
    plot_mechanistic_vs_ghgnn(results_dict, mode)
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    