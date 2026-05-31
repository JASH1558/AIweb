import torch 
import torch.nn as nn


device=torch.device("cpu")

class conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,1,1)
        self.conv2=nn.Conv2d(8,16,3,1,1)
        self.fc1=nn.Linear(16*7*7,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
        self.pool=nn.MaxPool2d(2,2)
        self.relu=nn.ReLU()
    def forward(self,x):
        print("A", flush=True)
    
        x=self.conv1(x)
        print("B", flush=True)
    
        x=self.relu(x)
        x=self.pool(x)
        print("C", flush=True)
    
        x=self.conv2(x)
        print("D", flush=True)
    
        x=self.relu(x)
        x=self.pool(x)
        print("E", flush=True)
    
        x=torch.flatten(x,1)
        print("F", flush=True)
    
        x=self.fc1(x)
        print("G", flush=True)
    
        x=self.fc2(x)
        print("H", flush=True)
    
        x=self.fc3(x)
        print("I", flush=True)

        return x
    

