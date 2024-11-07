#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Step 1: Load data
path = 'tmp/exampleRepository/dataset02.csv'
data = pd.read_csv(path)

# Step 2: Data Cleaning
# Drop cloumns with NaN
data = data.dropna()
print(data.tail(10))
