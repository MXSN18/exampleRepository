#import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load data

path = '/tmp/exampleRepository/dataset02.csv'
data = pd.read_csv(path)

# Step 2: Data Cleaning

# Convert non-numerical Values into numeric ones

data = data.apply(pd.to_numeric, errors='coerce')
print(data.tail(10))
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Drop cloumns with NaN
data = data.dropna()
data.reset_index(drop=True, inplace=True)
print(data.tail(10))

data = data.iloc[:200]

# Step : Quantile Filter

def quantile_filter(df, columns, lower_quantile=0.10, upper_quantile=0.90):
    for col in columns:
        if col in df.columns:
            # Quantilgrenzen berechnen
            lower_bound = df[col].quantile(lower_quantile)
            upper_bound = df[col].quantile(upper_quantile)
            print(lower_bound)
            print(upper_bound)
            # Filter anwenden
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        else:
            raise ValueError(f"Spalte {col} ist nicht im DataFrame vorhanden.")
    
    return df

# Step 3: Z-Scores
def z_score_filter(df, columns, threshold=3.0):
    for col in columns:
        if col in df.columns:
            # Z-Score berechnen
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) / std <= threshold]
        else:
            raise ValueError(f"Spalte {col} ist nicht im DataFrame vorhanden.")
    
    return df



# Step 4: IQR-Filter
def remove_outliers_iqr(df, columns, multiplier = 1.5):
    for col in columns:
        if col in df.columns:
            # Quartile berechnen
            Q1 = df[col].quantile(0.30)
            Q3 = df[col].quantile(0.70)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            # Filter anwenden
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        else:
            raise ValueError(f"Spalte {col} ist nicht im DataFrame vorhanden.")
    return df


# Anwendung der Filter
quantile_filter(data, ["x","y"], 0.01, 0.99)
print(data.tail(10))
remove_outliers_iqr(data, ["x","y"], 1.5)
print(data.tail(10))
z_score_filter(data, ["x","y"], 3)
print(data.tail(10))


# Step 5: Mean Normalization
data_normalized = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#scaler = MinMaxScaler()
#data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
#data_normalized = (data-data.mean())/data.std()
#data_normalized= data.iloc[:,0:-1] = data.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=1)
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
diagnostic = LinearRegDiagnostic(ols_model)
vif, fig, ax = diagnostic()
fig.savefig('UE_04_App2_DiagnosticPlots.pdf')
