from scipy import *
import sys
#import pybrain
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork


# Step 1: Load data
path = '/tmp/exampleRepository/dataset03.csv'
data = pd.read_csv(path)
x = data.iloc[:, :-1].values # all columns execpt the last one
y = data.iloc[:, -1].values # the last column as target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# Step 2: Create Dataset for pybrain
train_ds = SupervisedDataSet(x_train.shape[1], 1)
for i in range(len(x_train)):
    train_ds.addSample(x_train[i], [y_train[i]])

test_ds = SupervisedDataSet(x_test.shape[1], 1)
for i in range(len(x_test)):
    test_ds.addSample(x_test[i], [y_test[i]])


#Step 3: Create FeedForwardNetwork Model

ffn = buildNetwork(train_ds.indim, train_ds.outdim, bias=True,hiddenclass = LinearLayer, outclass = LinearLayer)


# Create input layer, hidden layer and output layer
inLayer = LinearLayer(x_train.shape[1])
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

# Add the layers to the Feed Forward Network
ffn.addInputModule(inLayer)
ffn.addInputModule(hiddenLayer)
ffn.addInputModule(outLayer)

# Create Connections between the layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer,outLayer)

# Connecting layers to the network
ffn.addConnection(in_to_hidden)
ffn.addConnection(hidden_to_out)

ffn.sortModules()

print(ffn)

trainer = BackpropTrainer(ffn, train_ds)
trainer.trainUntilConvergence(maxEpochs=10)

train_error = []
test_error = []

for i in range(10):
    train_error.append(trainer.train())
    test_error.append(trainer.testOnData(train_ds, verbose=False))
    print(f"Epoch: {i + 1}")
    print(f"Train-Error: {train_error[-1]}")
    print(f"Test-Error: {test_error[-1]}")
    print("-----------------------------------")

#loaded_model = NetworkReader.readFrom(r"")

