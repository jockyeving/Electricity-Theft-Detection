# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:31:54 2021

@author: jocky
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import natsort as ns
from natsort import natsorted
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import time

seed=47

import os
for dirname, _, filenames in os.walk('../sgcc_data/data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/data.csv',low_memory=False)
orig_df = df.copy()

### preprocessing ###

##sorting columns
df = df.reindex(natsorted(df.columns), axis=1) ## 1,10,11,....,2,3 => 1,2,3....,10,11


df = df.drop(columns=['CONS_NO'])
column_names = list(df.columns)
column_names.pop(len(column_names)-1)



##drop empty rows

df = df.dropna(how='all',subset=column_names)                       ## full NaN

## missing values 

def fill_missing_in_row(n):
    curr_row = df.iloc[n]
    for k in range(0,len(curr_row)-1):
        if (pd.isna(curr_row[k]) and (k == 0)):
            curr_row[k] = 0                                           ##first element
        elif (pd.isna(curr_row[k]) and ((not pd.isna(curr_row[k-1])) and (not pd.isna(curr_row[k+1])))):
            curr_row[k] = (curr_row[k-1] + curr_row[k+1])/2                 ##átlag
        elif (pd.isna(curr_row[k]) and (pd.isna(curr_row[k-1]) or pd.isna(curr_row[k+1]))):
            curr_row[k] = 0                                                 ##nulláz
    df.iloc[n] = curr_row

for k in range (0,len(df.index)):
    fill_missing_in_row(k)
    
## drop all 0 rows (only dropped all Na before)
def empty_row(n):
    sum_row = sum(df.iloc[n])
    flag = df.iloc[n].FLAG
    if ((sum_row - flag) == 0):
        df.drop(n, inplace = True)

##for k in range (0,len(df.index)):                                   
##    empty_row(k)
        
                
## outliers
def outliers_in_row(n):
    curr_row = df.iloc[n]
    curr_row_noflag = curr_row[0:(len(curr_row)-1)]
    mean = curr_row_noflag.mean()
    std = curr_row_noflag.std()
    for k in range(0,len(curr_row)-1):
        if (curr_row[k] > (mean+2*std)):
            curr_row[k] = mean+2*std
    df.iloc[n] = curr_row
    
for x in range (0,len(df.index)):
    outliers_in_row(x)

## normalization

def normalize_row(n):
    curr_row = df.iloc[n]
    curr_row_noflag = curr_row[0:(len(curr_row)-1)]
    max_row = max(curr_row_noflag)
    min_row = min(curr_row_noflag)
    for k in range(0,len(curr_row)-1):
        curr_row[k] = ( curr_row[k] - min_row ) / ( max_row - min_row )
    df.iloc[n] = curr_row
    
for k in range (0,len(df.index)):
    normalize_row(k)
    
df = df.dropna(how='all',subset=column_names)
df.reset_index(drop=True,inplace = True)

df.to_csv('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/preprocessed_data.csv',index = False)
