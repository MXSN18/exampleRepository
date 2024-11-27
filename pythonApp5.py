import requests
import pandas as pd
import csv
from bs4 import BeautifulSoup

# Import necessary modules
import sys
# Step 1: Add PyBrain to the system path
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, LinearLayer, SigmoidLayer
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# set the path for the data scraping
URL = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"
data = requests.get(URL).text

# Creating BeautifulSoup Object
soup = BeautifulSoup(data, 'html.parser')

# Defines the table
tables = soup.find_all('table')

# Scrape data from the website by iterating over the column and rows
'''for table in tables:
    rows = table.find_all('tr') # table row
    for row in rows:
        cols = row.find_all('td') # table data/cell
        cols = [ele.text.strip() for ele in cols]
        print(cols)'''

for table in tables:
    headers = [header.text.strip() for header in table.find_all('th')] # table head
    for row in table.find_all('tr'): # table row
        cells = [cell.text.strip() for cell in row.find_all('td')] # table data/cell
        if cells:
            data = dict(zip(headers, cells))
            print(data)

with open('dataset5.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # write headers
    for table in tables:
        for row in table.find_all('tr'):
            cells = [cell.text.strip() for cell in row.find_all('td')]
            if cells:
                writer.writerow(cells)

# Step 2: Load the dataset
data_path = '/tmp/exampleRepository/dataset5.csv'  # Path to the uploaded file
data = pd.read_csv(data_path)

data = data.iloc[:10000]

# Define input and output columns
input_columns = data.columns[:-1]  # All columns except the last are inputs
output_columns = data.columns[-1]  # The last column is the output

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)  # 80% training data
test_data = data.drop(train_data.index)  # Remaining 20% as test data

# Prepare the dataset for PyBrain
input_size = len(input_columns)
output_size = 1  # Assuming a single output column
ds = SupervisedDataSet(input_size, output_size)
for _, row in train_data.iterrows():
    ds.addSample(row[input_columns].values, row[output_columns])

# Step 3: Create and train the ANN
# Define the feedforward network
net = buildNetwork(input_size, 3, output_size, hiddenclass=TanhLayer, bias=True)

# Train the network using BackpropTrainer
trainer = BackpropTrainer(net, ds)
for epoch in range(100):  # Train for 100 epochs
    trainer.train()

# Save the trained model
model_path = '/tmp/exampleRepository/UE_06_App3_ANN_Model.pdf'
with open(model_path, 'wb') as f:
    pickle.dump(net, f)
print(f"Trained ANN model saved to: {model_path}")

# Step 4: Load the model and demonstrate performance
# Load the trained model
with open(model_path, 'rb') as f:
    loaded_net = pickle.load(f)

# Select two sample inputs from the test dataset
sample_1 = test_data.iloc[0][input_columns].values
sample_2 = test_data.iloc[1][input_columns].values

# Get activations from the original and loaded models
original_output_1 = net.activate(sample_1)
loaded_output_1 = loaded_net.activate(sample_1)
original_output_2 = net.activate(sample_2)
loaded_output_2 = loaded_net.activate(sample_2)

# Print results
print("Sample 1 Activation:")
print("Original Model Output:", original_output_1)
print("Loaded Model Output:", loaded_output_1)

print("\nSample 2 Activation:")
print("Original Model Output:", original_output_2)
print("Loaded Model Output:", loaded_output_2)

# Prepare training and testing data for plotting
train_inputs = train_data[input_columns].values
train_targets = train_data[output_columns].values

test_inputs = test_data[input_columns].values
test_targets = test_data[output_columns].values

# Generate predictions for the training and testing data
train_predictions = np.array([net.activate(inp) for inp in train_inputs]).flatten()
test_predictions = np.array([net.activate(inp) for inp in test_inputs]).flatten()

# Create scatter plots
plt.figure(figsize=(10, 6))

# Training data scatter plot
plt.scatter(train_data['x'], train_data['y'], color='orange', label='Training Data', alpha=0.6)

# Testing data scatter plot
plt.scatter(test_data['x'], test_data['y'], color='blue', label='Testing Data', alpha=0.6)
#plt.plot(train_data['x'], train_predictions, color='red', label='FFN')
# Add labels and legend
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values - Training and Testing Data")
plt.legend()
plt.grid(True)

# Show plot
plt.savefig('UE_04_App5_ScatterVisualization.pdf')
