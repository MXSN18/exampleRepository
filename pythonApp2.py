#import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split

# Step 1: Load data

path = '/tmp/exampleRepository/dataset02.csv'
data = pd.read_csv(path)

# Step 2: Data Cleaning

# Convert non-numerical Values into numeric ones
data = data.apply(pd.to_numeric, errors='coerce')

# Drop cloumns with NaN
data = data.dropna()
print(data.tail(10))

num_entries_y = data['y'].count()
print('Number of y entries: ', num_entries_y)

# Step 3: Z-Scores
#z_score = stats.zscore(data, axis=1)
#print(z_score)

# Step 4: IQR-Filter
def remove_outliers_iqr(df):
	Q1 = df.quantile(0.25)
	Q3 = df.quantile(0.75)
	IQR = Q3 - Q1
	return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Apply IQR filters to remove outliers
data = remove_outliers_iqr(data)
print(data.tail(10))

num_entries_y2 = data['y'].count()
print(num_entries_y2)

# Step 5: Mean Normalization
data_normalized = (data-data.mean())/data.std()
#data.iloc[:,0:-1] = data.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
print(data_normalized)

# Step 6: Splitting the data into train and test data

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
