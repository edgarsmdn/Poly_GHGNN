'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

# Visualize the performance of different polymer representations in data-driven 
# models using different metrics (MAE, R2, RMSE, MAPE) and in the context of 
# interpolation and extrapolation

import matplotlib.pyplot as plt
import json
import numpy as np
import os
from collections import Counter
import plotly.express as px
import pandas as pd

colors_mpi = [
    '#33A5C3',
    '#87878D',
    '#007675',
    '#78004B',
    '#383C3C',
    '#ECE9D4'
    ]

colors_mpi_extended = ["#33A5C3", "#87878D", "#007675", "#78004B", "#383C3C", 
                       "#ECE9D4", "#056e12", "#4cf185", "#1945c5", "#b69cfd", 
                       "#a335c8", "#add51f", "#ff0087", "#a2e59a", "#a33e12", 
                       "#3eeaef", "#0a60a8", "#f67afe", "#2524f9", "#e9c338", 
                       "#d6061a", "#f48e9b", "#fb9046", "#866609", "#fa1bfc"]

def extract_metrics_info(file_path, mode, split, metric):
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
    mode_data = json_data[mode]
    
    metrics = {}
    for key1 in mode_data.keys():
        metrics[key1] = {}
        
        split_data = mode_data[key1][split]
        for key2 in split_data.keys():
            metrics[key1][key2] = {}
            
            metrics[key1][key2]['mean'] = np.mean(split_data[key2][metric])
            metrics[key1][key2]['std'] = np.std(split_data[key2][metric])
    return metrics

def plot_error_bars(data, metric):
    x_names = ['monomer', 'ru_wo', 'ru_w','oligomer_10']
    x_labels = ['Monomer','Rep. unit','Per. unit','Oligomer']
    x_ticks = range(len(x_labels))
    
    fig, ax = plt.subplots(figsize=(6,5))
    colors = colors_mpi
    markers = ['o', 's', '^']
    
    key_labels = {'MN':'MN',
                  'MW': 'MW',
                  'PDI': 'MN/MW'
                  }
    
    for key, color, marker in zip(data.keys(), colors, markers):
        means = [data[key][name]['mean'] for name in x_names]
        stds = [data[key][name]['std'] for name in x_names]
        
        ax.errorbar(x_ticks, means, yerr=stds, fmt=marker, label=key_labels[key], 
                    linewidth=0.5, capsize=5, color=color, alpha=0.7)
        
    metric_labels = {'MAE':'Mean absolute error (MAE)',
                     'RMSE': 'Root mean squarred error (RMSE)',
                     'R2': 'R2',
                     'MAPE': 'Mean absolute percentage error (MAPE)'
                     }

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel(metric_labels[metric], fontsize=16)
    if metric == 'MAE' or metric == 'RMSE':
        ax.set_ylim(0, 0.25)
    elif metric == 'R2':
        ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3)
    ax.legend(fontsize=14)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.close(fig)
    
    return fig

def plot_error_bars_plotly(data, metric):
    
    metric_info = data
    data = []
    for dataset, values in metric_info.items():
        for representation, metrics in values.items():
            mean = metrics['mean']
            std = metrics['std']
            data.append([mean, std, dataset, representation])
    df = pd.DataFrame(data, columns=['Mean', 'Std', 'Dataset', 'Representation'])
    
    fig = px.scatter(df, x='Representation', y="Mean", color="Dataset",
                  error_y="Std")
    
    #fig.update_xaxes(tickvals=x_ticks, ticktext=x_labels)
    tick_fontsize = 14 
    fig.update_layout(yaxis_title=metric,
                      xaxis=dict(tickfont=dict(size=tick_fontsize)),
                      yaxis=dict(tickfont=dict(size=tick_fontsize)))
    point_size = 14  # Adjust the size as needed
    fig.update_traces(marker=dict(size=point_size))
    
    if metric == 'MAE' or metric == 'RMSE':
        fig.update_yaxes(range=[0, 0.6])
    elif metric == 'R2':
        fig.update_yaxes(range=[0, 1])
    
    return fig

split = 'test'
methods = ['RF', 'GHGNN', 'GHGNN_pss']
modes = ['interpolation', 'extrapolation']
metrics = ['MAE', 'R2', 'RMSE', 'MAPE']

for method in methods:
    filename = '../../models/'+method+'/'+method+'_performance.json'
    for mode in modes:
        for metric in metrics:
            path = '../../reports/figures/polymer_representations/'+mode+'/'+metric
            if not os.path.exists(path):
                os.makedirs(path)
            
            metric_info = extract_metrics_info(filename, mode, split, metric)
            fig = plot_error_bars(metric_info, metric)
            
            fig.savefig(path + '/' + split+'_'+
                        method +
                        '.png', dpi=350)
            
            fig_plotly = plot_error_bars_plotly(metric_info, metric)
            
            fig_plotly.write_html(path + '/' + split+'_'+
                        method +
                        '.html')
            
# Summary of performance

def count_frequency_best(methods, modes, datasets, metric='MAE', mean_std='mean'):
    poly_reps = ['monomer', 'ru_wo', 'ru_w','oligomer_10']
    best_reps = []
    for method in methods:
        filename = '../../models/'+method+'/'+method+'_performance.json'
        with open(filename, 'r') as file:
            json_data = json.load(file)
        for mode in modes:
            for dataset in datasets:
                best_std = np.inf
                if metric in ['MAE', 'RMSE', 'MAPE']:
                    best_mean = np.inf
                elif metric == 'R2':
                    best_mean = -np.inf
                for rep in poly_reps:
                    if mean_std == 'mean':
                        mean = np.mean(json_data[mode][dataset]['test'][rep][metric])
                        if metric in ['MAE', 'RMSE', 'MAPE']:
                            if mean < best_mean:
                                best_mean = mean
                                best_rep = rep
                        elif metric == 'R2':
                            if mean > best_mean:
                                best_mean = mean
                                best_rep = rep
                    elif mean_std == 'std':
                        std = np.std(json_data[mode][dataset]['test'][rep][metric])
                        if std < best_std:
                            best_std = std
                            best_rep = rep
                best_reps.append(best_rep)
                    
    frequency = Counter(best_reps)
    for rep in poly_reps:
        if rep not in frequency:
            frequency[rep] = 0
    sorted_frequency = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    for element, count in sorted_frequency.items():
        print(f"{element}: {count}")
    return sorted_frequency


def create_histogram(dictionaries, metric, mean_std):
    cases = ['All', 'Interpolation', 'Extrapolation', 'Random Forest', 'GH-GNNs']
    fig, axes = plt.subplots(nrows=1, ncols=len(dictionaries), 
                             sharey=True, 
                             squeeze=False,
                             figsize=(12,4))
    
    fig.subplots_adjust(wspace=0, hspace=0)  

    for i, dictionary in enumerate(dictionaries):
        ax = axes.flat[i]
        if i == 0:
            ax.set_ylabel('Frequency of best performance', fontsize=11)
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        
        sorted_keys, sorted_values = zip(*sorted(zip(keys, values)))
        
        sorted_keys = ['Monomer', 'Oligomer', 'Per. unit', 'Rep. unit']

        
        ax.bar(sorted_keys, sorted_values, color='#36454F')
        ax.set_xlabel(cases[i], fontsize=11)
        
        # Rotate x-tick labels by 45 degrees
        ax.set_xticklabels(sorted_keys, rotation=45)
        

    plt.tight_layout()
    plt.close(fig)
    path = '../../reports/figures/polymer_representations/'
    fig.savefig(path + 'summary_'+ metric + '_' + mean_std + '.png', dpi=350)

for metric in metrics:
    for mean_std in ['mean', 'std']:

        print(f'\nBest polymer representations according to {metric} and {mean_std}\n')
                            
        print('All')  
        print('-'*50)  
        methods = ['RF', 'GHGNN', 'GHGNN_pss']
        modes = ['interpolation', 'extrapolation']   
        datasets = ['MN', 'MW', 'PDI']  
        case_1 = count_frequency_best(methods, modes, datasets, metric, mean_std)
        print('='*50)
        
        print('Interpolation')  
        print('-'*50)  
        methods = ['RF', 'GHGNN', 'GHGNN_pss']
        modes = ['interpolation']   
        datasets = ['MN', 'MW', 'PDI']  
        case_2 = count_frequency_best(methods, modes, datasets, metric, mean_std)
        print('='*50)
        
        print('Extrapolation')  
        print('-'*50)  
        methods = ['RF', 'GHGNN', 'GHGNN_pss']
        modes = ['extrapolation']   
        datasets = ['MN', 'MW', 'PDI']  
        case_3 = count_frequency_best(methods, modes, datasets, metric, mean_std)
        print('='*50)
        
        print('RF')  
        print('-'*50)  
        methods = ['RF']
        modes = ['interpolation', 'extrapolation']  
        datasets = ['MN', 'MW', 'PDI']  
        case_4 = count_frequency_best(methods, modes, datasets, metric, mean_std)
        print('='*50)
        
        print('GNNs')  
        print('-'*50)  
        methods = ['GHGNN', 'GHGNN_pss']
        modes = ['interpolation', 'extrapolation']   
        datasets = ['MN', 'MW', 'PDI']  
        case_5 = count_frequency_best(methods, modes, datasets, metric, mean_std)
        print('='*50)
        
        dictionaries = [case_1, case_2, case_3, case_4, case_5]
        create_histogram(dictionaries, metric, mean_std)






