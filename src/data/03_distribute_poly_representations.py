'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
import numpy as np

df_rep = pd.read_excel('../../data/raw/Solutes.xlsx')
df_rep = df_rep[df_rep['in_homo'] == 'yes']

oligomer = 'oligomer_10'

reps = ['monomer', 'ru_w', 'ru_wo', 'ru_ws', oligomer]
files = ['MN', 'MW', 'PDI']

def create_oligomer_rep(ru_ws, dp):
    assert dp >= 1
    
    separators = [i for i, char in enumerate(ru_ws) if char == '*']
    
    if len(separators) == 2:
        start, end = separators
    elif len(separators) == 1:
        ru = ''
        for i, x in enumerate(ru_ws):
            if i not in separators:
                ru += x
        oligomer = ru*dp
        return oligomer
        
    
    # For detecting start point
    idxs = [x for x in range(start+1)]
    
    # For detecting end point
    ring_end_bool = False
    if ru_ws[end-1].islower():
        idxs.extend([end-2, end-1, end])
    elif ru_ws[end-1] == ']':
        if ru_ws[end-3] == '[':
            idxs.extend([end-3, end-2, end-1, end])
        elif ru_ws[end-4] == '[':
            idxs.extend([end-4, end-3, end-2, end-1, end])
    elif ru_ws[end-1] in ['1','2','3','4','5','6','7','8','9']:
        ring_id = ru_ws[end-1]
        start_ending_ring = ru_ws.find(ring_id) - 1
        end_ring_idxs = [j for j in np.arange(start_ending_ring,len(ru_ws))]
        idxs.extend(end_ring_idxs)
        ring_end_bool = True
    elif ru_ws[end-1] == ')':
        if ru_ws[end-2] in ['1','2','3','4','5','6','7','8','9']:
            ring_id = ru_ws[end-2]
            start_ending_ring = ru_ws.find(ring_id) - 1
            end_ring_idxs = [j for j in np.arange(start_ending_ring,len(ru_ws))]
            idxs.extend(end_ring_idxs)
            ring_end_bool = True
    else:
        idxs.extend([end-1, end])
    
    # In case a branch in the ending group
    branch_end_bool = False
    if end < len(ru_ws)-1:
        branch_end_bool = True
        if ru_ws[end+1] == '(':
            branch_end = []
            for j in range(len(ru_ws) - (end+1)):
                idxs.append(end+1+j)
                branch_end.append(end+1+j)
    
    branch = ''
    for i, x in enumerate(ru_ws):
        if i not in idxs:
            branch += x
            
    start_point = ru_ws[:start]
    end_point = ru_ws[end-1:end]
    if branch_end_bool:
        end_point += ru_ws[branch_end[0]:branch_end[-1]+1]
    if ring_end_bool:
        end_point = ru_ws[end_ring_idxs[0]:end_ring_idxs[-1]]
    
    
    oligomer = ''
    for n in range(dp):
        oligomer += start_point
        oligomer += branch
        oligomer += end_point
        
    return oligomer

# Create oligomer representation
oligomers = []
for ru_ws in df_rep['ru_ws'].tolist():
    oligomers.append(create_oligomer_rep(ru_ws, int(oligomer[-2:])))
df_rep[oligomer] = oligomers
    

for file in files:
    df = pd.read_csv('../../data/interim/'+file+'.csv')
    polymers = df['Solute'].tolist()
    for rep in reps:
        smiles = []
        for polymer in polymers:
            rep_smi = df_rep[df_rep['Solute'] == polymer][rep].iloc[0]
            smiles.append(rep_smi)
        df['Solute_SMILES_'+rep] = smiles
        
    df.to_csv('../../data/interim/'+file+'_reps.csv', index=False)




