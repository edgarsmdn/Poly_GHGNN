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

def pretrain_GHGNN(df, dataset, method_name, hyperparameters, df_valid, n_epochs):
    
    path = '../../models/' + method_name
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)
    df_valid[mol_column_solvent] = df_valid['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    df_valid[mol_column_solute]  = df_valid['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    target = 'ln-omega'
    
    # Compute graphs
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    if dataset == 'PDI':
        df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, 
                                                     mol_column_solute, 
                                                     target, extra1='ln-MW',
                                                     extra2='ln-MN')
        df_valid[graphs_solv], df_valid[graphs_solu] = sys2graph(df_valid, 
                                                                 mol_column_solvent, 
                                                     mol_column_solute, 
                                                     target, extra1='ln-MW',
                                                     extra2='ln-MN')
    else:
        df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, 
                                                     mol_column_solute, 
                                                     target, extra1='ln-'+dataset)
        df_valid[graphs_solv], df_valid[graphs_solu] = sys2graph(df_valid, 
                                                                 mol_column_solvent, 
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
    
    valid_index = df_valid.index.tolist()
    valid_loader = get_dataloader_pairs_T(df_valid, 
                                          valid_index, 
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
    mae_valid = []; valid_loss = []
    
    for epoch in pbar:
        stats = OrderedDict()
        
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats, scaler))
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        stats.update(eval(model, device, valid_loader, MAE, stats, 'Valid', task_type))
        # Scheduler
        scheduler.step(stats['MAE_Train'])
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        valid_loss.append(stats['Valid_loss'])
        mae_valid.append(stats['MAE_Valid'])
        pbar.set_postfix(stats) # include stats in the progress bar
    
    print('-'*30)
    print('Training MAE   : '+ str(mae_train[-1]))
    print('Valid    MAE   : '+ str(mae_valid[-1]))
    
    # # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['Valid_loss']  = valid_loss
    df_model_training['MAE_Valid']  = mae_valid
    df_model_training['MAE_Train']  = mae_train
    
    save_train_traj(path, df_model_training, 
                    method_name+'_' + dataset + '_' + str(n_epochs), 
                    valid=True)
    
    # Save best model
    final_model = copy.deepcopy(model.state_dict())
    model_name = method_name+'_' + dataset + '_' + str(n_epochs) + '.pth'
    torch.save(final_model, path + '/' + model_name)
    
    end       = time.time()
    
    print('\nTraining time (min): ' + str((end-start)/60))


n_epochs = 200
hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'n_epochs'    : n_epochs,
                        'batch_size'  : 32
                        }

method_name = 'GHGNN_pretrained_small_systems'
df = pd.read_csv('../../data/processed/small_systems_train.csv')
df_valid = pd.read_csv('../../data/processed/small_systems_test.csv')

pretrain_GHGNN(df, 'MN', method_name, hyperparameters_dict, df_valid, n_epochs)
pretrain_GHGNN(df, 'PDI', method_name, hyperparameters_dict, df_valid, n_epochs)