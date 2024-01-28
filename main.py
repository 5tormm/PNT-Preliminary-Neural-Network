import torch
import torch.nn as nn
import torch.nn.functional as F

#Create Model Class that inherits nn.Module
class Model(nn.Module):
    #Input Layer -> 
    #Hidden Layer(number of neurons) -> 
    #Other hidden layer... -> Output
    def __init__(self, in_features=4, h1=8, h2=9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

#pick random seed    
torch.manual_seed(69)
#Create instance of model
model = Model()
