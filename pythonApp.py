# import pandas and statsmodels

import pandas as pd
import statsmodels.api as sm

# Load the CSV file

path = '/tmp/exampleRepository/dataset01.csv'
data = pd.read_csv(path)

print(data.head())

# 1. Number of data entries of column 'y'
num_entries_y = data['y'].count()
print('Number of y entries: ',num_entries_y)

# 2. Mean of y
y_mean = data['y'].mean()
print('Mean value of y: ',y_mean)
# 3. Standard Deviation of y
y_std = data['y'].std()
print('Standard deviation of y: ',y_std)
# 4. Variance of y
y_var = data['y'].var()
print('Variance of y: ',y_var) 
# 5. Min and Max of y
y_max = data['y'].max()
print('max value: ', y_max)
y_min = data['y'].min()
print('min value: ',y_min)
# 6. OLS Model influence 'x', target 'y'
# response variable endog = y
y = data['y']
# explanatory variable exog = x
x = data['x']

# add constant to predictor variables
x = sm.add_constant(x)

# fit linear regression model
model = sm.OLS(y,x).fit()

#view model summary
ols_model_file_path = '/tmp/exampleRepository/OLS_model.txt'
with open(ols_model_file_path, 'w') as file:
	file.write(model.summary().as_text())
