'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
from rdkit import Chem
from utilities.fingerprints import get_fp_ECFP_bitvector
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from utilities.performance import get_perfromance_report

def pred_RF(df, split, i, dataset, reps, method_name):
    
    path = '../../models/'+ method_name+ '/' + split + '/split_' + str(i)
    
    for rep in reps:
    
        # Build molecule from SMILES
        mol_column_solvent     = 'Molecule_Solvent'
        df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

        mol_column_solute      = 'Molecule_Solute'
        df[mol_column_solute]  = df['Solute_SMILES_'+rep].apply(Chem.MolFromSmiles)
    
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
    
        # Load RF model
        model = joblib.load(path+'/'+method_name+'_'+dataset+'_'+rep+'.joblib')
        y_pred = model.predict(X_train)
    
        df[method_name+'_'+rep] = y_pred
    
        df = df.drop(columns=[mol_column_solvent, 
                              mol_column_solute,
                              ])
    return df

method_name = 'RF'    
reps = ['monomer', 'ru_w', 'ru_wo', 'oligomer_10']  
for split in ['interpolation', 'extrapolation']:
    print('-'*70)
    print(split)
    for i in tqdm(range(10)):
        for dataset in ['MN', 'MW', 'PDI']:
            for spl in ['train', 'test']:
                df = pd.read_csv('../../data/processed/'+split+'/split_'+str(i)+'/'+
                                 dataset+'_'+spl+'.csv')
                df_pred = pred_RF(df, split, i, dataset, reps, method_name)
                df_pred.to_csv('../../models/'+method_name+'/'+ split + '/split_' + str(i)+'/'
                               + dataset + '_' + spl + '_pred_'+method_name+'.csv', index=False)
                    
get_perfromance_report(method_name)
