from scipy import *
import sys
#import pybrain
sys.path.append('C:/Users/max/OneDrive - Universität Potsdam/Master/WS24_25/AIBAS/VSCode/AIBAS_exercise_WorkingDirectory/pybrain')
print(sys.path)
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split

#from pybrain.structure import FeedForwardNetwork 

# Step 1: Load data
path = '/tmp/exampleRepository/dataset03.csv'
data = pd.read_csv(path)

# Step 2: Data Cleaning
data['x'] = data['x'] = pd.to_numeric(data['x'], errors='coerce')

data['y'] = data['y'] = pd.to_numeric(data['y'], errors='coerce')

data_clean = data.dropna()

#Step 3: Z-scores, 


