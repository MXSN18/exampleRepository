# import pandas and statsmodels

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file

path = '/tmp/exampleRepository/dataset02.csv'
data = pd.read_csv(path)

print(data.head())

# 1. Number of data entries of column 'y'

#num_entries_y = data['y'].count()
#print('Number of y entries: ',num_entries_y)

# 2. Mean of y
y_mean = data['y'].mean()
#print('Mean value of y: ',y_mean)
# 3. Standard Deviation of y
y_std = data['y'].std()
#print('Standard deviation of y: ',y_std)
# 4. Variance of y
#y_var = data['y'].var()
#print('Variance of y: ',y_var) 
# 5. Min and Max of y
#y_max = data['y'].max()
#print('max value: ', y_max)
#y_min = data['y'].min()
#print('min value: ',y_min)
# 6. OLS Model influence 'x', target 'y'
# response variable endog = y
y = data['y']
# explanatory variable exog = x
x = data['x']

# add constant to predictor variables
#x = sm.add_constant(x)

# fit linear regression model
#model = sm.OLS(y,x).fit()

#view model summary
#ols_model_file_path = '/tmp/exampleRepository/OLS_model.txt'
#with open(ols_model_file_path, 'w') as file:
#	file.write(model.summary().as_text())


#Homework 3

# Step 1: Data Cleaning
# Keep only numeric columns
data = data.apply(pd.to_numeric, errors='coerce').dropna()
# drop rows with NaN values
data = data.dropna()
print(data['y'].count())
print(data.tail(10))

# Step 2: Outlier Removal
# Using Z-Score method to identify outliers in each column
z = np.abs(stats.zscore(data['x']))
print(z)
z_scores = np.abs(zscore(data))
print(z_scores)
data = data[(z_scores < 3).all(axis=1)]  # Filter rows with Z-scores less than 3 for all columns

# Step 4: Data Normalization
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Step 5: Split Data
train_data, test_data = train_test_split(data_normalized, test_size=0.2, random_state=42)
train_data.to_csv('dataset02_training.csv', index=False)
test_data.to_csv('dataset02_testing.csv', index=False)

# Step 6: OLS Model Training (on training data)
X_train = sm.add_constant(train_data['x'])  # Add constant for intercept
y_train = train_data['y']
ols_model = sm.OLS(y_train, X_train).fit()

# Step 7: Scatter Plot Visualization
plt.figure(figsize=(10, 6))
plt.scatter(train_data['x'], train_data['y'], color='orange', label='Training Data')
plt.scatter(test_data['x'], test_data['y'], color='blue', label='Testing Data')
plt.plot(train_data['x'], ols_model.predict(sm.add_constant(train_data['x'])), color='red', label='OLS Model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot with OLS Model')
plt.legend()
plt.savefig('UE_04_App2_ScatterVisualizationAndOlsModel.pdf')

# Step 8: Box Plot
plt.figure(figsize=(8, 5))
data_normalized.boxplot()
plt.title('Box Plot of Data Dimensions')
plt.savefig('UE_04_App2_BoxPlot.pdf')

# Step 9: Diagnostic Plots (Assuming UE_04_LinearRegDiagnostic.py is present)
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
diagnostic_plots = LinearRegDiagnostic(ols_model)
diagnostic_plots.plot()
plt.savefig('UE_04_App2_DiagnosticPlots.pdf')
