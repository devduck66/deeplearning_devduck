import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define three linear layers
        self.layer1 = nn.Linear(in_features=10, out_features=50)
        self.layer2 = nn.Linear(in_features=50, out_features=20)
        self.layer3 = nn.Linear(in_features=20, out_features=1)
        # Define ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Apply the layers with ReLU activation in between
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # No activation function in the last layer
        return x

# Initialize the model
model = SimpleNN()
print(model)

# Create a dummy input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10

# Forward pass through the model
output = model(input_tensor)
print("Output:", output)
