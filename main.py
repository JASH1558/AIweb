import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
train_data=datasets.MNIST(root="./data",train=True,download=True,transform=transforms.ToTensor())
test_data=datasets.MNIST(root="./data",train=False,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=100,shuffle=False)

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
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x




model=conv().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
num_epochs=10
for epoch in range(num_epochs):
    model.train()
    for images,labels in train_loader:
        optimizer.zero_grad()
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
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
        print(outputs)
        print(outputs.shape)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(),"mnist_model.pth")