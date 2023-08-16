'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
import matplotlib.pyplot as plt
import os


colors_mpi_extended = ["#33A5C3", "#87878D", "#007675", "#78004B", "#383C3C", 
                       "#ECE9D4", "#056e12", "#4cf185", "#1945c5", "#b69cfd", 
                       "#a335c8", "#add51f", "#ff0087", "#a2e59a", "#a33e12", 
                       "#3eeaef", "#0a60a8", "#f67afe", "#2524f9", "#e9c338", 
                       "#d6061a", "#f48e9b", "#fb9046", "#866609", "#fa1bfc"]

poly = []
solvents = []
datasets = ['MN', 'MW', 'PDI']
dataset_label = {'MN':'MN', 'MW':'MW', 'PDI':'MN/MW'}

path = '../../reports/figures/polymer_solvent_percentages/'
if not os.path.exists(path):
    os.makedirs(path)

def pie_plot(df, column, top_n):
    # plot polymer percentages
    value_counts = df[column].value_counts()
    top_values = value_counts.head(top_n)
    other_count = value_counts[top_n:].sum()
    combined_counts = pd.concat([top_values, pd.Series([other_count], index=['Others'])])
    
    fig = plt.figure(figsize=(8, 8))
    plt.pie(combined_counts, labels=combined_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors_mpi_extended, textprops={'fontsize': 10},
            pctdistance=0.8)
    plt.title(dataset_label[dataset], fontsize=18)
    plt.tight_layout()
    fig.savefig(path + dataset + '_' + column  + '.png', dpi=350)
    plt.close(fig)


for dataset in datasets:
    df = pd.read_csv('../../data/interim/'+dataset+'.csv')
    
    pie_plot(df, 'Solute', 8)
    pie_plot(df, 'Solvent', 15)
    
    poly.extend(df['Solute'].unique().tolist())
    solvents.extend(df['Solvent'].unique().tolist())
    
    
poly = set(poly)
solvents = set(solvents)

print(len(poly))
print(len(solvents))

