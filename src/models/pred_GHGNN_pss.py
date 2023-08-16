'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
from rdkit import Chem
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from GHGNN_architecture import GHGNN
import torch
import numpy as np
from utilities.performance import get_perfromance_report
import time

def pred_GHGNN_pss(df, split, i, dataset, rep, method_name, hyperparameters):
    
    path = '../../models/' + method_name+'/'+ split + '/split_' + str(i)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    batch_size  = hyperparameters['batch_size']
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = 'ln-omega'
    
    for rep in reps:
        # Build molecule from SMILES
        mol_column_solvent     = 'Molecule_Solvent'
        df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)
    
        mol_column_solute      = 'Molecule_Solute'
        df[mol_column_solute]  = df['Solute_SMILES_'+rep].apply(Chem.MolFromSmiles)
        
        # Compute graphs
        graphs_solv, graphs_solu = 'g_solv', 'g_solu'
        if dataset == 'PDI':
            df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, 
                                                         mol_column_solute, 
                                                         target, extra1='ln-MW',
                                                         extra2='ln-MN')
        else:
            df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, 
                                                         mol_column_solute, 
                                                         target, extra1='ln-'+dataset)
            
            
        # Data loaders
        indices = df.index.tolist()
        predict_loader = get_dataloader_pairs_T(df, 
                                              indices, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size, 
                                              shuffle=False, 
                                              drop_last=False)
            
    
    
        ######################
        # --- Prediction --- #
        ######################
        # Model
        v_in = n_atom_features()
        e_in = n_bond_features()
        u_in = 3 # ap, bp, topopsa
        ext_in = [2 if dataset == 'PDI' else 1][0]
        model    = GHGNN(v_in, e_in, u_in, hidden_dim, ext_in)
    
        model.load_state_dict(torch.load(path + '/' + method_name +'_'+
                                         dataset+'_'+rep+'.pth', 
                                     map_location=torch.device(available_device)))
        device   = torch.device(available_device)
        model    = model.to(device)
    
        y_pred_final = np.array([])
        model.eval()
        with torch.no_grad():
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(device)
                batch_solute  = batch_solute.to(device)
                batch_T = batch_T.to(device)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        y_pred  = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda()).cpu()
                        y_pred  = y_pred.numpy().reshape(-1,)
                    else:
                        y_pred  = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1,)
                    y_pred_final = np.concatenate((y_pred_final, y_pred))
                
        df[method_name+'_'+rep] = y_pred_final
        
        df = df.drop(columns=[mol_column_solvent, 
                              mol_column_solute,
                              graphs_solv,
                              graphs_solu
                              ])
    return df

hyperparameters_dict = {'hidden_dim'  : 113,
                        'batch_size'  : 32
                        }

method_name = 'GHGNN_pss'    
reps = ['monomer', 'ru_w', 'ru_wo', 'oligomer_10']  
for split in ['interpolation', 'extrapolation']:
    print('-'*70)
    print(split)
    time.sleep(2)
    for i in range(10):
        print('-'*50)
        print(i)
        time.sleep(2)
        for dataset in ['MN', 'MW', 'PDI']:
            print('-'*40)
            print(dataset)
            time.sleep(2)
            for spl in ['train', 'test']:
                df = pd.read_csv('../../data/processed/'+split+'/split_'+str(i)+'/'+
                                 dataset+'_'+spl+'.csv')
                df_pred = pred_GHGNN_pss(df, split, i, dataset, reps, method_name, 
                                     hyperparameters_dict)
                df_pred.to_csv('../../models/'+method_name+'/'+ split + '/split_' + str(i)+'/'
                               + dataset + '_' + spl + '_pred_'+method_name+'.csv', index=False)
                    
get_perfromance_report(method_name)

