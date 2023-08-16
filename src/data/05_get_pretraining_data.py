'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df_train = pd.read_csv('../../data/external/small_systems_train.csv')
df_test = pd.read_csv('../../data/external/small_systems_test.csv')


for df, split in zip([df_train, df_test], ['train', 'test']):
    solu_mols = [Chem.MolFromSmiles(smiles) for smiles in df['Solute_SMILES'].tolist()]
    solv_mols = [Chem.MolFromSmiles(smiles) for smiles in df['Solvent_SMILES'].tolist()]
    
    solu_mw = np.array([Descriptors.ExactMolWt(mol) for mol in solu_mols])
    solv_mw = np.array([Descriptors.ExactMolWt(mol) for mol in solv_mols])

    log_omegas = np.log(np.exp(df['log-gamma'].to_numpy()) * solu_mw/solv_mw)
    T_in_K = df['T'].to_numpy() + 273.15
    ln_MN = np.log(solu_mw)
    ln_MW = np.log(solu_mw**2/solu_mw)
    
    df['T'] = T_in_K
    df['ln-omega'] = log_omegas
    df['ln-MN'] = ln_MN
    df['ln-MW'] = ln_MW
    df['Physical_state'] = ['']*len(ln_MW)
    
    df.to_csv('../../data/processed/small_systems_'+split+'.csv', index=False)

