# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:24:32 2021

@author: jocky
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import module1

test_size = 0.18


df = pd.read_csv('C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/preprocessed_data.csv',low_memory=False)

y = df.FLAG
x = df.drop('FLAG', axis = 1)
x, y = module1.under_sampler(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
x_2d_train =  module1.create_2d_array(x_train)
x_2d_test =  module1.create_2d_array(x_test)
np.save("x_2d_train_18_us",x_2d_train)
np.save("x_2d_test_18_us",x_2d_test)
np.save("y_train_18_us",y_train)
np.save("y_test_18_us",y_test)