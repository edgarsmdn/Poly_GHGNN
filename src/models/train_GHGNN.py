'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from GHGNN_architecture import GHGNN, count_parameters
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from utilities.Train_eval import train, eval, MAE
from utilities.save_info import save_train_traj

# External utilities
from tqdm import tqdm
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch.cuda.amp import GradScaler

   
def train_GHGNN(df, split, i, dataset, rep, method_name, hyperparameters):
    
    path = '../../models/' + method_name+'/'+ split + '/split_' + str(i)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES_'+rep].apply(Chem.MolFromSmiles)
    
    target = 'ln-omega'
    
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
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    
    start       = time.time()
    
    # Data loaders
    train_index = df.index.tolist()
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True)
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    ext_in = [2 if dataset == 'PDI' else 1][0]
    model    = GHGNN(v_in, e_in, u_in, hidden_dim, ext_in)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)     
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = tqdm(range(n_epochs))
        scaler = GradScaler()
    else:
        pbar = tqdm(range(n_epochs))
        scaler=None
    
    # To save trajectory
    mae_train = []; train_loss = []
    
    for epoch in pbar:
        stats = OrderedDict()
        
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats, scaler))
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        # Scheduler
        scheduler.step(stats['MAE_Train'])
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        pbar.set_postfix(stats) # include stats in the progress bar
    
    print('-'*30)
    print('Training MAE   : '+ str(mae_train[-1]))
    
    # # Save training trajectory
    # df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    # df_model_training['MAE_Train']  = mae_train
    # save_train_traj(path, df_model_training, method_name, valid=False)
    
    # Save best model
    final_model = copy.deepcopy(model.state_dict())
    model_name = method_name+'_' + dataset + '_' + rep + '.pth'
    torch.save(final_model, path + '/' + model_name)
    
    end       = time.time()
    
    print('\nTraining time (min): ' + str((end-start)/60))
   
n_epochs = 200
hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'n_epochs'    : n_epochs,
                        'batch_size'  : 32
                        }

method_name = 'GHGNN'
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
            df = pd.read_csv('../../data/processed/'+split+'/split_'+str(i)+'/'+
                             dataset+'_train.csv')
            for rep in ['monomer', 'ru_w', 'ru_wo', 'oligomer_10']:
                print('-'*30)
                print(rep)
                time.sleep(2)
                train_GHGNN(df, split, i, dataset, rep, method_name, 
                            hyperparameters_dict)

