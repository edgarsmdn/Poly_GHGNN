'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
from rdkit.Chem import AllChem
import os
from matplotlib.colors import LinearSegmentedColormap



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

###################################
# --- Check oligomer function --- #
###################################

test_oligomer_function = False
if test_oligomer_function:
    from rdkit.Chem import Draw
    
    deg_poly = 50
    
    ru_ws = 'C*(CCCCCCC)C*'
    oligomer = create_oligomer_rep(ru_ws, 5)
    mol_oligomer = Chem.MolFromSmiles(oligomer)
    Draw.MolToImage(mol_oligomer)
    
    similarities_all_bit = np.zeros((1, deg_poly-2))
    similarities_all_count = np.zeros((1, deg_poly-2))
    for i in range(2, deg_poly):
        
        similarities = []
        smiles_1 = [create_oligomer_rep(ru_ws, i)]
        smiles_2 = [create_oligomer_rep(ru_ws, i+1)]
        
        mols_1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_1]
        mols_2 = [Chem.MolFromSmiles(smiles) for smiles in smiles_2]
        
        fp_1_bit = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048) for mol in mols_1]
        fp_2_bit = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048) for mol in mols_2]
        
        fp_1_count = [AllChem.GetMorganFingerprint(mol, radius=4) for mol in mols_1]
        fp_2_count = [AllChem.GetMorganFingerprint(mol, radius=4) for mol in mols_2]
        
        j = 0
        for f1t, f2t, f1d, f2d in zip(fp_1_bit, fp_2_bit, fp_1_count, fp_2_count):
            similarities_all_bit[j,i-2] = DataStructs.TanimotoSimilarity(f1t, f2t)
            similarities_all_count[j,i-2] = DataStructs.DiceSimilarity(f1d, f2d)
            j += 1
            
    fig = plt.figure(figsize=(8,5))
    for i in range(similarities_all_bit.shape[0]):
        plt.plot(range(2,deg_poly), similarities_all_bit[i, :], label='polymer paper')#, color=colors[i])
    plt.axvline(x=16, color='red', ls='--')
    plt.axhline(y=1, color='k', ls='--', lw=0.5)
    plt.xlabel('Degree of polimerization', fontsize=12)
    plt.ylabel('Tanimoto similarity', fontsize=12)
    #plt.yscale('log')
    plt.xlim(2,deg_poly-2)
    ax= plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncols=2, fontsize=5)
    plt.tight_layout()
    


    # Dice convergence plot
    fig = plt.figure(figsize=(8,5))
    for i in range(similarities_all_count.shape[0]):
        plt.plot(range(2,deg_poly), similarities_all_count[i, :], label='polymer paper')#, color=colors[i])
    plt.axvline(x=16, color='red', ls='--')
    plt.axhline(y=1, color='k', ls='--', lw=0.5)
    plt.xlabel('Degree of polimerization', fontsize=12)
    plt.ylabel('Dice similarity', fontsize=12)
    #plt.yscale('log')
    plt.xlim(2,deg_poly-2)
    ax= plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncols=2, fontsize=5)
    plt.tight_layout()
        
    
        
poly_names = [
   #   'Alkyd resin',
   # 'Epoxy resin (Eponol 55)',
   'Poly(butyl acrylate)',
   # 'Poly(butylene adipate)',
   # 'Poly(di(oxyethylene)oxyadipyl), alpha,omega-dihydroxy',
   # 'Poly(di(oxyethylene)oxysuccinyl), alpha,omega-dihydroxy',
  # 'Poly(diethylsiloxane)',
  # 'Poly(dimethylsiloxane)',
  # 'Poly(DL-lactide)',
  # 'Poly(epi-chlorohydrin)',
  # 'Poly(epsilon-caprolactone)',
  # 'Poly(delta-valerolactone)',
  #'Poly(ethyl acrylate)',
  'Poly(ethyl methacrylate)',
  #'Poly(ethylene adipate)',
  'Poly(ethylene oxide)',
  'Poly(ethylene succinate)',
  # 'Poly(hexamethylene sebacate)',
  # 'Poly(hexamethylene carbonate), alpha,omega-dihydroxy',
  'Poly(methyl acrylate)',
  'Poly(methyl methacrylate)',
  # 'Poly(N-(n-octadecyl)maleimide)',
  'Poly(n-butyl acrylate)',
  'Poly(n-butyl methacrylate)',
  #'Poly(n-hexyl acrylate)',
  'Poly(n-hexyl methacrylate)',
  #'Poly(n-pentyl methacrylate)',
  'Poly(n-propyl acrylate)',
  #'Poly(n-propyl methacrylate)',
  # 'Poly(oxy-3-(2-methoxyethoxy) propyleneoxysuccinyl), alpha,omega-dihydroxy',
  # 'Poly(tetramethylenecarbonate), alpha,omega-dihydroxy',
  # 'Poly(tri(oxyethylene)oxysuccinyl), alpha,omega-dihydroxy',
  'Poly(vinyl acetate)',
  'Poly(vinyl chloride)',
  'Poly(vinyl methyl ether)',
  # 'Poly(vinyl trimethylsilane)',
  # 'Poly(vinylidene fluoride)',
  #'Poly(vinyl isobutyl ether)',
  #'Polyacrylonitrile',
  #'Polyarylate',
  'Polybutadiene',
  'Polyethylene',
  # 'Polyethylene, low-density',
  'Polyisobutylene',
  'Polyisoprene',
  # 'Polyoxyethylene, alpha,omega-dihydroxy',
  # 'Polyoxypropylene, alpha,omega-dihydroxy',
  'Polystyrene',
  'Polysulfone'
 ]

df_reps = pd.read_excel('../../data/raw/Solutes.xlsx')
df_reps = df_reps[df_reps['Solute'].isin(poly_names)]

deg_poly = 20
deg_poly = deg_poly + 2

similarities_all_bit = np.zeros((df_reps.shape[0] ,deg_poly-2))
similarities_all_count = np.zeros((df_reps.shape[0] ,deg_poly-2))

for i in range(2,deg_poly):
    
    similarities = []
    smiles_1 = [create_oligomer_rep(ru_ws, i) for ru_ws in df_reps['ru_ws'].tolist()]
    smiles_2 = [create_oligomer_rep(ru_ws, i+1) for ru_ws in df_reps['ru_ws'].tolist()]
    
    mols_1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_1]
    mols_2 = [Chem.MolFromSmiles(smiles) for smiles in smiles_2]
    
    fp_1_bit = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048) for mol in mols_1]
    fp_2_bit = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048) for mol in mols_2]
    
    fp_1_count = [AllChem.GetMorganFingerprint(mol, radius=4) for mol in mols_1]
    fp_2_count = [AllChem.GetMorganFingerprint(mol, radius=4) for mol in mols_2]
    
    j = 0
    for f1t, f2t, f1d, f2d in zip(fp_1_bit, fp_2_bit, fp_1_count, fp_2_count):
        similarities_all_bit[j,i-2] = DataStructs.TanimotoSimilarity(f1t, f2t)
        similarities_all_count[j,i-2] = DataStructs.DiceSimilarity(f1d, f2d)
        j += 1
    
# Plot
path = '../../reports/figures/convergence_degpoly'
if not os.path.exists(path):
    os.makedirs(path)

colors_mpi_extended = ["#33A5C3", "#87878D", "#007675", "#78004B", "#383C3C", 
                       "#ECE9D4", "#056e12", "#4cf185", "#1945c5", "#b69cfd", 
                       "#a335c8", "#add51f", "#ff0087", "#a2e59a", "#a33e12", 
                       "#3eeaef", "#0a60a8", "#f67afe", "#2524f9", "#e9c338", 
                       "#d6061a", "#f48e9b", "#fb9046", "#866609", "#fa1bfc"]


fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharex=True)

# First subplot for Tanimoto convergence
for i in range(similarities_all_bit.shape[0]):
    axes[0].plot(range(2, deg_poly), similarities_all_bit[i, :], 
                 label=poly_names[i], color=colors_mpi_extended[i])
axes[0].axvline(x=10, color='red', ls='--')
axes[0].axhline(y=1, color='k', ls='--', lw=0.5)
#axes[0].set_xlabel('Degree of polymerization', fontsize=12)
axes[0].set_ylabel('Tanimoto similarity', fontsize=18)
axes[0].set_xlim(2, deg_poly-1)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(ncols=2, fontsize=10)

axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
#axes[0].set_yscale("log")

# Dice convergence plot
for i in range(similarities_all_count.shape[0]):
    axes[1].plot(range(2, deg_poly), similarities_all_count[i, :], 
                 label=poly_names[i], color=colors_mpi_extended[i])
axes[1].axvline(x=10, color='red', ls='--')
axes[1].axhline(y=1, color='k', ls='--', lw=0.5)
axes[1].set_xlabel('Degree of polymerization', fontsize=18)
axes[1].set_ylabel('Dice similarity', fontsize=18)
axes[1].set_xlim(2, deg_poly-1)
axes[1].grid(True, linestyle='--', alpha=0.5)
#axes[1].legend(ncols=2, fontsize=5)
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
#axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig(path + '/oligomer_convergence.png', dpi=350)
plt.close(fig)



# MPI colors cmap
colors = ['#33A5C3', '#ECE9D4']
mpi_cmap = LinearSegmentedColormap.from_list('mpi_cmap', colors, N=100)

# Select the column for which you want to annotate values
selected_column = 8

fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

# First subplot for Tanimoto convergence
im = axes[0].imshow(similarities_all_bit, cmap=mpi_cmap, vmin=0.35, vmax=1)
axes[0].set_xlabel('Degree of polymerization', fontsize=22)
axes[0].set_yticks(np.arange(len(poly_names))) 
axes[0].set_yticklabels(poly_names)  
axes[0].set_xticks(np.arange(deg_poly-2))
axes[0].set_xticklabels(list(range(2, deg_poly)))  
axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].set_title('Tanimoto similarity', fontsize=24)
axes[0].axvline(x=7.5, color='red', ls='--', lw=0.4)
axes[0].axvline(x=8.5, color='red', ls='--', lw=0.4)

# Annotate the heat map with values from the selected column
for i in range(similarities_all_bit.shape[0]):
    value = similarities_all_bit[i, selected_column]
    axes[0].text(selected_column, i, f'{value:.2f}',
            ha='center', va='center', color='k', fontsize=10)

# Minor ticks
axes[0].set_xticks(np.arange(-.5, deg_poly-2, 1), minor=True)
axes[0].set_yticks(np.arange(-.5, 19, 1), minor=True)
axes[0].grid(which='minor', color='k', linestyle='-', linewidth=0.2)
axes[0].tick_params(which='minor', bottom=False, left=False)

# Dice convergence plot
im2 = axes[1].imshow(similarities_all_count, cmap=mpi_cmap, vmin=0.35, vmax=1)
axes[1].set_xlabel('Degree of polymerization', fontsize=22)
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].set_xticks(np.arange(deg_poly-2))
axes[1].set_xticklabels(list(range(2, deg_poly))) 
axes[1].set_title('Dice similarity', fontsize=24)
axes[1].axvline(x=7.5, color='red', ls='--', lw=0.4)
axes[1].axvline(x=8.5, color='red', ls='--', lw=0.4)

# Annotate the heat map with values from the selected column
for i in range(similarities_all_count.shape[0]):
    value = similarities_all_count[i, selected_column]
    axes[1].text(selected_column, i, f'{value:.2f}',
            ha='center', va='center', color='k', fontsize=10)

# Minor ticks
axes[1].set_xticks(np.arange(-.5, deg_poly-2, 1), minor=True)
axes[1].set_yticks(np.arange(-.5, 19, 1), minor=True)
axes[1].grid(which='minor', color='k', linestyle='-', linewidth=0.2)
axes[1].tick_params(which='minor', bottom=False, left=False)

# Create a color bar axis
cax = fig.add_axes([0.35, 0.07, 0.5, 0.03])

cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.subplots_adjust(top=1.1)
plt.savefig(path + '/oligomer_convergence_heatmap.png', dpi=350)
plt.close(fig)