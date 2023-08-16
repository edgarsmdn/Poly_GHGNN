'''
Project: PolyGNN
                    Random forest training
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from rdkit import Chem
import numpy as np
from utilities.fingerprints import get_fp_ECFP_bitvector
import joblib
import os
from tqdm import tqdm

def train_RF(df, split, i, dataset, rep, method_name):
    
    path = '../../models/' + method_name+'/'+ split + '/split_' + str(i)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES_'+rep].apply(Chem.MolFromSmiles)
    
    target = 'ln-omega'
    
    # Compute fingerprints
    fp_function = get_fp_ECFP_bitvector
    
    solv_mols = df[mol_column_solvent].tolist()
    solu_mols = df[mol_column_solute].tolist()
    
    # Get model inputs
    X_train_solv = np.vstack([fp_function(mol) for mol in solv_mols])
    X_train_solu = np.vstack([fp_function(mol) for mol in solu_mols])
    
    if dataset == 'PDI':
        X_train_spec = df[['ln-MW','ln-MN']].to_numpy().reshape(-1,2)
    else:
        X_train_spec = df['ln-'+dataset].to_numpy().reshape(-1,1)
    
    X_train_T = df['T'].to_numpy().reshape(-1,1)
    
    X_train = np.hstack([X_train_solv, X_train_solu, X_train_spec, X_train_T])
    y_train = df[target].to_numpy()
    
    # Train model
    model = RF(criterion="squared_error")
    model.fit(X_train, y_train)
    
    # Save model
    model_name = method_name+'_' + dataset + '_' + rep + '.joblib'
    joblib.dump(model, path+'/'+model_name)
   
method_name = 'RF'
for split in ['interpolation', 'extrapolation']:
    print('-'*70)
    print(split)
    for i in tqdm(range(10)):
        for dataset in ['MN', 'MW', 'PDI']:
            df = pd.read_csv('../../data/processed/'+split+'/split_'+str(i)+'/'+
                             dataset+'_train.csv')
            for rep in ['monomer', 'ru_w', 'ru_wo', 'oligomer_10']:
                train_RF(df, split, i, dataset, rep, method_name)

