# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 01:50:13 2021

@author: jocky
"""

import pandas as pd
import sklearn
import module1
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/preprocessed_data.csv',low_memory=False)

#df = module1.df_firsthalf(df)
#df = module1.df_secondhalf(df)
df = module1.df_bothhalves(df)

models = ['LR','SVM','RF','LightGBM']
model_type = models[3]
q= 0.2
threshold = 0.5

y = df.FLAG
x = df.drop('FLAG', axis = 1)


#x, y = module1.over_sampler(x, y)
x, y = module1.under_sampler(x, y)
#x, y = module1.smote(x, y)
  

   
      
        
model, model_result, predictions, probabilities, y_test, x_test = module1.classify(x, y, test_ratio = q,model_type=model_type,threshold = threshold)
TP, FP, TN, FN = module1.perf_measure(y_test.to_numpy(), predictions)
 

## Optimal threshold and ROC Curve
#maximum = 0
#f1_scores = []
#for i in range(100):
#    predictions = (probabilities >= (i/100)).astype(bool)
#    curr = sklearn.metrics.f1_score(y_test,predictions)
#    f1_scores.append(curr)
#    if curr > maximum:
#        maximum = curr
#        curr_index = i
#sklearn.metrics.plot_roc_curve(model,x_test,y_test)



# Implement all sampling methods on a model from 0.05 -> 0.95 test to train ratios
#results_basecase, results_oversampling, results_undersampling, results_smote = apply_all_sampling(x, y)
#filenames = ['regression_base.npy','regression_oversampling.npy','regression_undersampling.npy','regression_smote.npy']
#np.save('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/'+filenames[0],results_basecase)
#np.save('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/'+filenames[1],results_oversampling)
#np.save('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/'+filenames[2],results_undersampling)
#np.save('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/'+filenames[3],results_smote)
    