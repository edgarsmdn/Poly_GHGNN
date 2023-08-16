'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
import numpy as np
import sys
from contextlib import redirect_stdout

df = pd.read_csv('../../data/raw/IDACs_polymers.csv')

# Include physical state information from Notes
df_notes = pd.read_csv('../../data/raw/Notes.csv')
df_notes = df_notes[df_notes['Physical_state'].notnull()]
notes_dict = df_notes.set_index('Note')['Physical_state'].to_dict()

phs_st = []
for note in df['Note'].tolist():
    if note in notes_dict:
        phs_st.append(notes_dict[note])
    else:
        phs_st.append('')

df['Physical_state'] = phs_st

# Get only homopolymers
df = df[df['Solute_type'] == 'Homopolymer'].copy()

# Split data into MN, MW and PDI datasets
df_MN = df[~df['MN'].isnull()].copy()
df_MW = df[~df['MW'].isnull()].copy()
df_PDI = df[(~df['MN'].isnull() & ~df['MW'].isnull())].copy()

# Combine omegas - if not omega_calc get the omega
df_MN['omega'] = df_MN['calc_omega_inf'].fillna(df_MN['omega_inf'])
df_MW['omega'] = df_MW['calc_omega_inf'].fillna(df_MW['omega_inf'])
df_PDI['omega'] = df_PDI['calc_omega_inf'].fillna(df_PDI['omega_inf'])

# Neglect the rows without omega value
df_MN_cl = df_MN[~df_MN['omega'].isnull()]
df_MW_cl = df_MW[~df_MW['omega'].isnull()]
df_PDI_cl = df_PDI[~df_PDI['omega'].isnull()]

# Get just the useful columns
df_MN_final = pd.DataFrame({
    'Solute':df_MN_cl['Solute'].tolist(),
    'Solvent':df_MN_cl['Solvent'].tolist(),
    'ln-MN':np.log(df_MN_cl['MN'].astype(float).tolist()),
    'T':df_MN_cl['T (K)'].astype(float).tolist(),
    'ln-omega':np.log(df_MN_cl['omega'].astype(float).to_numpy()),
    'Physical_state':df_MN_cl['Physical_state'].tolist()
    })

df_MW_final = pd.DataFrame({
    'Solute':df_MW_cl['Solute'].tolist(),
    'Solvent':df_MW_cl['Solvent'].tolist(),
    'ln-MW':np.log(df_MW_cl['MW'].astype(float).tolist()),
    'T':df_MW_cl['T (K)'].astype(float).tolist(),
    'ln-omega':np.log(df_MW_cl['omega'].astype(float).to_numpy()),
    'Physical_state':df_MW_cl['Physical_state'].tolist()
    })

df_PDI_final = pd.DataFrame({
    'Solute':df_PDI_cl['Solute'].tolist(),
    'Solvent':df_PDI_cl['Solvent'].tolist(),
    'ln-MW':np.log(df_PDI_cl['MW'].astype(float).to_numpy()),
    'ln-MN':np.log(df_PDI_cl['MN'].astype(float).to_numpy()),
    'T':df_PDI_cl['T (K)'].astype(float).tolist(),
    'ln-omega':np.log(df_PDI_cl['omega'].astype(float).to_numpy()),
    'Physical_state':df_PDI_cl['Physical_state'].tolist()
    })


# Collect solvent SMILES
df_solvents = pd.read_csv('../../data/raw/Solvents.csv')

for df in [df_MN_final, df_MW_final, df_PDI_final]:
    print('============ ****** =================')
    solvs = df['Solvent'].tolist()
    solvs_smiles = []
    for solv in solvs:
        smiles = df_solvents[df_solvents['Solvent'] == solv]['Solvent_SMILES']
        if smiles.shape[0] != 1:
            raise Exception(f'Not unique SMILES for {solv}')
        else:
            solvs_smiles.append(smiles.iloc[0])
    df.insert(2, 'Solvent_SMILES', solvs_smiles)

# Averaged duplicates
df_MN_final['key'] = df_MN_final['Solute'] + '_' + df_MN_final['Solvent'] + \
                    '_' + df_MN_final['T'].astype(str) + '_' + df_MN_final['ln-MN'].astype(str) + \
                        '_' + df_MN_final['Physical_state']
df_MN_final = df_MN_final.groupby('key').agg(
    {'ln-omega': 'mean', 
      'Solute': 'first', 
      'Solvent': 'first',
      'Solvent_SMILES': 'first',
      'ln-MN': 'first',
      'T': 'first',
      'Physical_state':'first'
      }).reset_index()
df_MN_final = df_MN_final.drop('key', axis=1)


df_MW_final['key'] = df_MW_final['Solute'] + '_' + df_MW_final['Solvent'] + \
                    '_' + df_MW_final['T'].astype(str) + '_' + df_MW_final['ln-MW'].astype(str) + \
                        '_' + df_MW_final['Physical_state']
df_MW_final = df_MW_final.groupby('key').agg(
    {'ln-omega': 'mean', 
      'Solute': 'first', 
      'Solvent': 'first',
      'Solvent_SMILES': 'first',
      'ln-MW': 'first',
      'T': 'first',
      'Physical_state':'first'
      }).reset_index()
df_MW_final = df_MW_final.drop('key', axis=1) 


            
df_PDI_final['key'] = df_PDI_final['Solute'] + '_' + df_PDI_final['Solvent'] + \
                    '_' + df_PDI_final['T'].astype(str) + '_' + df_PDI_final['ln-MW'].astype(str) + \
                        '_' + df_PDI_final['ln-MN'].astype(str) + '_' + df_PDI_final['Physical_state'] 
df_PDI_final = df_PDI_final.groupby('key').agg(
    {'ln-omega': 'mean', 
      'Solute': 'first', 
      'Solvent': 'first',
      'Solvent_SMILES': 'first',
      'ln-MW': 'first',
      'ln-MN': 'first',
      'T': 'first',
      'Physical_state':'first'
      }).reset_index()
df_PDI_final = df_PDI_final.drop('key', axis=1) 


with open('../../reports/01_n_points.txt', 'w') as f:
    with redirect_stdout(f):
        for name, df in zip(['MN','MW','PDI'] ,[df_MN_final, df_MW_final, df_PDI_final]):
            print('============ ****** =================')
            print(name)
            print('n points  : ', df.shape[0])
            print('Polymers  : ', df['Solute'].nunique())
            print('Solvents  : ', df['Solvent'].nunique()) 
            
            obs_systems = df['Solute'] + '_' + df['Solvent']
            n_systems = obs_systems.nunique()
            perct_obs = 100*n_systems/(df['Solute'].nunique() * df['Solvent'].nunique())
            print('% observed: ', np.round(perct_obs, 2)) 

# Reset standard output to the console
sys.stdout = sys.__stdout__
     
# Save dataframes
df_MN_final.to_csv('../../data/interim/MN.csv', index=False)
df_MW_final.to_csv('../../data/interim/MW.csv', index=False)
df_PDI_final.to_csv('../../data/interim/PDI.csv', index=False)


