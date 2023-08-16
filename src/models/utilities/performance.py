'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error
                             )
import json
from contextlib import redirect_stdout
import numpy as np
import sys

def get_perfromance_report(method_name):
    
    results = {}
    for split in ['interpolation', 'extrapolation']:
        results[split] = {}
        for dataset in ['MN', 'MW', 'PDI']:
            results[split][dataset] = {}
            for spl in ['train', 'test']:
                results[split][dataset][spl] = {}
                for rep in ['monomer', 'ru_w', 'ru_wo', 'oligomer_10']:
                    results[split][dataset][spl][rep] = {}
                    maes, r2s, rmses, mapes = [],[],[],[] 
                    for i in range(10):
                        df = pd.read_csv('../../models/'+method_name+'/'+ split 
                                         + '/split_' + str(i)+'/'
                                       + dataset + '_' + spl + '_pred_'+method_name+'.csv')
                        
                        y_true = df['ln-omega'].to_numpy()
                        y_pred = df[method_name+'_'+rep].to_numpy()
                        
                        maes.append(mean_absolute_error(y_true, y_pred))
                        r2s.append(r2_score(y_true, y_pred))
                        rmses.append(mean_squared_error(y_true, y_pred)**0.5)
                        mapes.append(mean_absolute_percentage_error(y_true, y_pred)*100)
                    
                    # Save results
                    results[split][dataset][spl][rep]['MAE'] = maes
                    results[split][dataset][spl][rep]['R2'] = r2s
                    results[split][dataset][spl][rep]['RMSE'] = rmses
                    results[split][dataset][spl][rep]['MAPE'] = mapes
    
    # Save json
    path = '../../models/'+method_name+'/'
    with open(path+method_name+'_performance.json', 'w') as fp:
        json.dump(results, fp, indent=4)
        
    # Write report
    with open('../../reports/03_'+method_name+'_performance.txt', 'w') as f:
        with redirect_stdout(f):
            print('Performance report for '+ method_name)
            print('='*80)
            print('\n\n')
            for key1 in results.keys():
                print(key1)
                print('-'*70)
                for key2 in results[key1]:
                    print(key2)
                    print('-'*60)
                    for key3 in results[key1][key2]:
                        print('\n')
                        print(key3)
                        print('*'*50)
                        for key4 in results[key1][key2][key3]:
                            print('\n')
                            print(key4)
                            print('-'*40)
                            spec_dict = results[key1][key2][key3][key4]
                            rounding = 2
                            for metric in ['MAE', 'R2', 'RMSE', 'MAPE']:
                                mean = np.round(np.mean(spec_dict[metric]), rounding)
                                std  = np.round(np.std(spec_dict[metric]), rounding)
                                print(metric+': ', mean, '('+str(std)+')')
                    print('\n')
                print('\n')
    # Reset standard output to the console
    sys.stdout = sys.__stdout__
                    

