'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
import random
import numpy as np

def split_interpolation(df, solute='Solute', solvent='Solvent', train_ratio=0.9,
                        base='Solute'):
    
    """
    Split the given DataFrame into training and test sets for interpolation.

    The function splits the DataFrame based on the specified solute, solvent, 
    and train_ratio parameters. It generates a training set for interpolation 
    by randomly selecting mixtures involving the specified base component from 
    the DataFrame. The remaining mixtures form the test set. Additionally, it 
    identifies and adds extrapolation cases to the training set based on solute 
    and solvent to ensure that only interpolation is tested.

    Args:
        df (pandas.DataFrame): The DataFrame to split.
        solute (str, optional): The column name representing the solute 
            component. Defaults to 'Solute'.
        solvent (str, optional): The column name representing the solvent 
            component. Defaults to 'Solvent'.
        train_ratio (float, optional): The ratio of mixtures to include in the 
            training set. Defaults to 0.9.
        base (str, optional): The component to be considered as the base for 
            splitting. Can be either 'Solute' or
            'Solvent'. Defaults to 'Solute'.

    Returns:
        tuple: A tuple containing two DataFrames - the training set (df_train) 
            and the test set (df_test).

    """
    
    if base == 'Solute':
        pos = 0
    else:
        pos = 1
    
    df = df.copy()
    df['key'] = df[solute] + '_' + df[solvent]
    keys = df['key'].unique()
    
    bases = df[base].unique()
    
    mixtures_dict = {}
    for b in bases:
        # Get all mixtures involving this base
        for key in keys:
            b_in_key = key.split('_')[pos]
            if b == b_in_key:
                if b not in mixtures_dict:
                    mixtures_dict[b] = [key]
                else:
                    mixtures_dict[b].append(key)
    
    train_lst, test_lst = [], []
    for b in bases:
        mixtures = mixtures_dict[b]
        num_train = int(len(mixtures) * train_ratio)
        train_lst.extend(random.sample(mixtures, num_train))
        test_lst.extend([x for x in mixtures if x not in train_lst])
        
    
    train_mask = df['key'].isin(train_lst)
    df_train = df[train_mask]
    df_train = df_train.drop('key', axis=1).reset_index(drop=True)
    
    
    test_mask = df['key'].isin(test_lst)
    df_test = df[test_mask]
    df_test = df_test.drop('key', axis=1).reset_index(drop=True)
    
    # Add extrapolation cases to training
    solutes_train = df_train[solute].unique()
    solutes_test = df_test[solute].unique()
    solvents_train = df_train[solvent].unique()
    solvents_test = df_test[solvent].unique()
    
    idx_to_drop = []
    for s in solutes_test:
        if s not in solutes_train:
            for i in range(df_test.shape[0]):
                if s == df_test[solute].iloc[i]:
                    if i not in idx_to_drop:
                        idx_to_drop.append(i)
                        df_train = pd.concat([df_train, df_test.iloc[i].to_frame().T], axis=0)
                              
    
    
    for s in solvents_test:
        if s not in solvents_train:
            for i in range(df_test.shape[0]):
                if s == df_test[solvent].iloc[i]:
                    if i not in idx_to_drop:
                        idx_to_drop.append(i)
                        df_train = pd.concat([df_train, df_test.iloc[i].to_frame().T], axis=0)
    
    # Delete row from test
    df_test = df_test.drop(idx_to_drop, axis=0)
    
    # Reset index
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    return df_train, df_test


def split_extrapolation(df, solute='Solute', solvent='Solvent', train_ratio=0.9,
               base='Solvent'):
    """
    Split a DataFrame into two DataFrames for extrapolation based on a 
    specified column.

    The function splits the DataFrame into two parts: one for training and one 
    for testing, with the testing part containing rows corresponding to the 
    specified column values that are not present in the training part.

    Args:
        df (pandas.DataFrame): The DataFrame to be split.
        solute (str): The column name for the solute variable. 
            Default is 'Solute'.
        solvent (str): The column name for the solvent variable. 
            Default is 'Solvent'.
        train_ratio (float): The ratio of the data to use for training. 
            Default is 0.9 (90% for training).
        base (str): The column name used for splitting the data into training 
            and testing parts. Default is 'Solvent'.

    Returns:
        tuple: A tuple containing two DataFrames - one for training and one 
            for testing.

    """
    
    df = df.copy()
    
    # Split the bases into train and test
    bases = df[base].unique().tolist()
    num_train = int(len(bases) * train_ratio)
    train_bases = random.sample(bases, num_train)
    test_bases = [x for x in bases if x not in train_bases]
    
    # Get the splits based on the splitted bases
    idx_test = []
    for b in bases:
        if b in test_bases:
            for i in range(df.shape[0]):
                if b == df[base].iloc[i]:
                    if i not in idx_test:
                        idx_test.append(i)
    
    df_test  = df.loc[idx_test].reset_index(drop=True)
    df_train = df.drop(idx_test).reset_index(drop=True)
    
    return df_train, df_test
    
      
import os
from contextlib import redirect_stdout
import sys

with open('../../reports/02_splits.txt', 'w') as f:
    with redirect_stdout(f):
        for split, split_fun in zip(['interpolation', 'extrapolation'],
                                    [split_interpolation, split_extrapolation]):
            print('\n\n\n')
            print(split)
            print('='*80)
            
            for i in range(10):
                print('\n\n')
                print('Split '+str(i))
                print('-'*60)
                
                # Create folder
                folder = '../../data/processed/' + split + '/split_' + str(i)
                if not os.path.exists(folder):
                    os.makedirs(folder)   
                    
                for file in ['MN', 'MW', 'PDI']:
                    print('\n')
                    print(file)
                    print('-'*40)
                    df = pd.read_csv('../../data/interim/'+file+'_reps.csv')
                    df_train, df_test = split_fun(df)
                    
                    # Save splits
                    df_train.to_csv(folder+'/'+file+'_train.csv', index=False)
                    df_test.to_csv(folder+'/'+file+'_test.csv', index=False)
                    
                    # Save statistics for each
                    train_percentage = np.round(df_train.shape[0]/df.shape[0]*100,2)
                    test_percentage = np.round(df_test.shape[0]/df.shape[0]*100,2)
                    print('Train % ', train_percentage)
                    print('Test % ', test_percentage)
                    print('-'*30)
                    print('Solutes: ', df['Solute'].nunique())
                    print('Solvents: ', df['Solvent'].nunique())
                    print('-'*30)
                    solutes_train = df_train['Solute'].unique()
                    solutes_test = df_test['Solute'].unique()

                    ext = 0
                    for s in solutes_test:
                        if s not in solutes_train:
                            ext += 1

                    print('Solute interpolation: ', len(solutes_test) - ext)
                    print('Solute extrapolation: ', ext)
                    print('-'*30)

                    solvents_train = df_train['Solvent'].unique()
                    solvents_test = df_test['Solvent'].unique()

                    ext = 0
                    for s in solvents_test:
                        if s not in solvents_train:
                            ext += 1
                            
                    print('Solvent interpolation: ', len(solvents_test) - ext)
                    print('Solvent extrapolation: ', ext)
                    print('-'*30)
                    
                    
    
# Reset standard output to the console
sys.stdout = sys.__stdout__    
     
# -------------------------------------------------------------------------
### Test code for checking that the split_interpolation function works

# df = pd.read_csv('../../data/interim/MN.csv')
# df_train, df_test = split_interpolation(df, base='Solute')

# print('Train %', np.round(df_train.shape[0]/df.shape[0]*100,2))
# print('Test %', np.round(df_test.shape[0]/df.shape[0]*100,2))
# print('All %', df_train.shape[0]/df.shape[0]*100 + df_test.shape[0]/df.shape[0]*100)

# print('====================')

# print('Solutes: ', df['Solute'].nunique())
# print('Solvents: ', df['Solvent'].nunique())

# print('====================')

# solutes_train = df_train['Solute'].unique()
# solutes_test = df_test['Solute'].unique()

# ext = 0
# for s in solutes_test:
#     if s not in solutes_train:
#         ext += 1

# print('Solute interpolation: ', len(solutes_test) - ext)
# print('Solute extrapolation: ', ext)
        
# print('====================')

# solvents_extrapolation, solvents_interpolation = 0,0
# solvents_train = df_train['Solvent'].unique()
# solvents_test = df_test['Solvent'].unique()

# ext = 0
# for s in solvents_test:
#     if s not in solvents_train:
#         ext += 1
        
# print('Solvent interpolation: ', len(solvents_test) - ext)
# print('Solvent extrapolation: ', ext)

# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
### Test code for checking that the split_extrapolation function works

# df = pd.read_csv('../../data/interim/MN.csv')
# df_train, df_test = split_extrapolation(df, base='Solvent')

# print('Train %', np.round(df_train.shape[0]/df.shape[0]*100,2))
# print('Test %', np.round(df_test.shape[0]/df.shape[0]*100,2))
# print('All %', df_train.shape[0]/df.shape[0]*100 + df_test.shape[0]/df.shape[0]*100)

# print('====================')

# print('Solutes: ', df['Solute'].nunique())
# print('Solvents: ', df['Solvent'].nunique())

# print('====================')

# solutes_train = df_train['Solute'].unique()
# solutes_test = df_test['Solute'].unique()

# ext = 0
# for s in solutes_test:
#     if s not in solutes_train:
#         ext += 1

# print('Solute interpolation: ', len(solutes_test) - ext)
# print('Solute extrapolation: ', ext)
        
# print('====================')

# solvents_extrapolation, solvents_interpolation = 0,0
# solvents_train = df_train['Solvent'].unique()
# solvents_test = df_test['Solvent'].unique()

# ext = 0
# for s in solvents_test:
#     if s not in solvents_train:
#         ext += 1
        
# print('Solvent interpolation: ', len(solvents_test) - ext)
# print('Solvent extrapolation: ', ext)

# -------------------------------------------------------------------------