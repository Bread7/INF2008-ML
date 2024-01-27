import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch, os
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.neural_network import MLPClassifier

centers = [(-1, -1), (1, 1)]
cluster_std = [1.5, 1.5]

x, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

plt.scatter(x[y == 0, 0], x[y == 0, 1], color="red", s=10, label="cluster1")
plt.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", s=10, label="cluster2")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color="red", s=10, label="cluster1")
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color="blue", s=10, label="cluster2")
plt.show()

plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color="red", s=10, label="cluster1")
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color="blue", s=10, label="cluster2")
plt.show()

# %matplotlib inline


#
#   A.0 Normalization (2 marks)
#
x_train_copy = x_train
y_train_copy = y_train
mean_x_train = np.mean(x_train_copy, axis=0)
std_x_train = np.std(x_train_copy, axis=0)

x_train = (x_train_copy - mean_x_train) / std_x_train
x_test = (x_test - mean_x_train) / std_x_train

#
#   A 1.1 Create simple MLP in pytorch (5 marks)
#
X_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.LongTensor(y_train)

X_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.LongTensor(y_test)


# Create dataset and dataloader
dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(dataset_train, batch_size=8, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=8, shuffle=False)

# Create and instantiate MLP class
class MLP(nn.Module):
    def __init__(self):
        # Initialise nn.module class
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
        # Layer definitions
        # self.layer1 = nn.Linear(2, 64)
        # self.relu1 = nn.ReLU()
        # self.layer2 = nn.Linear(64, 32)
        # self.relu2 = nn.ReLU()
        # self.layer3 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.layers(x)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)

# Set the optimizer to Adagrad with learning rate of 0.001. The loss should be CrossEntropyLoss

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)

# Run training loop. Set number of epochs to be 100 and run each epoch
num_epochs = 100

# To look here again, the values are being randomized through the model
# Not sure if it is intended
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        # Zero out gradients in optimizer
        optimizer.zero_grad()
        # Forward pass
        output = model(inputs)
        # Calculate the loss
        loss = criterion(output, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Set model to evaluation mode
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Set forward pass
        output = model(inputs)
        # Get predicted values
        # print("output = ", output)
        _, predicted = torch.max(output.data, 1)
        # Get number of observations in batch
        total += labels.size(0)
        # Get number of correct predictions
        correct += (predicted == labels).sum().item()


accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#
#   A 1.2 Create simple perceptron using sklearn (3 marks)
#
classifier = MLPClassifier(alpha=0.001, hidden_layer_sizes=(), max_iter=1000, random_state=1, solver='lbfgs')
y_pred = classifier.fit(x_train, y_train)
print("Classifier coefficients = ", classifier.coefs_)
print("Classifier intercepts = ", classifier.intercepts_)
score = classifier.score(x_test, y_test)
print("Classifier score = ", score)

#
#   A 1.3 Create simple perceptron using python (5 marks)
#
class Perceptron:
    def __init__(self):
        self.weights = np.zeros((2,1))
        self.bias = np.zeros((1,1))

    def predict(self, inputs):
        # Forward pass
        net_input = np.dot(inputs, self.weights) + self.bias 
        # Apply step function (binary threshold) as activation function. The value of t
        ## of the step function should be 0.
        predictions = self.step_function(net_input)
        return predictions
        # return 1 if net_input[0] > 0 else 0
    
    def train(self, inputs, targets, learning_rate=0.001, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # Forward pass
                predictions = self.predict(inputs[i])
                # Compute error
                errors = targets[i] - predictions
                # Update weights and bias
                # self.weights += learning_rate * errors * inputs[i]
                self.weights += learning_rate * errors * inputs[i].reshape(self.weights.shape)
                self.bias += learning_rate * errors

    def step_function(self, net_input):
        return np.where(net_input > 0, 1, 0)

perceptron = Perceptron()

# Train perceptron
perceptron.train(x_train_copy, y_train_copy, learning_rate=0.01, epochs=1000)

# Make predictions
predictions = perceptron.predict(x_train_copy)
# print("predictions = ", predictions)
correct = (predictions.flatten() == y_train).sum().item()
# correct = (predictions == y_train).sum().item()
print(correct / y_train.shape[0])

#
#   A 1.4 Creation of Activation Function (6 marks)
#
class Sigmoid():
    def __init__(self):
        return
    
    def forward(self, z):
        sigmoid_z = 1 / (1 + np.exp(-z)) 
        
        return sigmoid_z
    
    def derivative(self, z):
        output = self.forward(z) - (1 - self.forward(z))
        return output
    
class Tanh():
    def __init__(self):
        return
    
    def forward(self, z):
        return np.tanh(z)
    
    def derivative(self, z):
        return 1 - np.tanh(z)**2 # sech^2 (z) 
   
class Relu():
    def __init__(self):
        return
    
    def forward(self, z):
        return np.max(z, 0)
    
    def derivative(self, z):
        return 1 if z >= 0 else 0

#
#   A 1.5 Creation of MSE (7 marks)
# 
class MSELoss():
    def forward(self, A, Y):
        # N      = #dimension 0 of A
        # C      = #dimension 1 of A
        N, C = A.shape              # N dimension 0 of A, C dimension 1 of A 
        se     = A - Y              # Sum of errors
        sse    = se ** 2            # Squared sum of errors
        mse    =  sse / (2 * N * C)
        return mse
    
    def backward(self, A, Y):
        N, C = A.shape
        dldA = (1 / (N * C)) * (A - Y)
        return dldA
    
#
#   A 1.6 Creation of Forward Functions (3 marks)
#
class Linear:
    def __init__(self, in_features, out_features, debug=False):
        self.W      = np.zeros((out_features, in_features), dtype="f")
        self.b      = np.zeros((out_features, 1), dtype="f")
        self.dldW   = np.zeros((out_features, in_features), dtype="f")
        self.dldb   = np.zeros((out_features, 1), dtype="f")

    def forward(self, A):
        self.A      = A
        self.N      = A.shape[0]
        self.Ones   = np.ones((self.N, 1), dtype="f")
        Z           = np.dot(A, self.W.T) + np.dot(self.Ones, self.b.T) 

        return Z
    
#
#   A 1.7 Creation of Backward Functions (9 marks)
#
def backward(self, dLdZ):    
    dZdA      = self.W.T
    dZdW      = self.A
    dZdb      = np.eye(self.b.shape[0])
    dLdA      = np.dot(dLdZ, dZdA.T)
    dLdW      = np.dot(dLdZ.T, dZdW)
    dLdb      = np.dot(dLdZ.T, dZdb)
    dLdi      = None
    dZdi      = None