# Import necessary modules
import sys
# Step 1: Add PyBrain to the system path
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')
import pandas as pd
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



# Step 2: Load the dataset
data_path = '/tmp/exampleRepository/dataset03.csv'  # Path to the uploaded file
data = pd.read_csv(data_path)

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
model_path = '/tmp/exampleRepository/UE_05_App3_ANN_Model.pdf'
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




