import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
train_data=datasets.MNIST(root="./data",train=True,download=True,transform=transforms.ToTensor())
test_data=datasets.MNIST(root="./data",train=False,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(train_data,batch_size=6000,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000,shuffle=False)

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.layers=nn.Sequential(
            nn.Linear(784,384),
            nn.ReLU(),
            nn.Linear(384,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
            x=self.flatten(x)
            x = self.layers(x)
            return x

model=neural_network().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
num_epochs=15
for epoch in range(num_epochs):
    model.train()
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(),"mnist_model.pth")