import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # first convolutional layer takes in 1 channel (grayscale image) and outputs 16 channels, with a kernel size of 3x3. We get 26*26*16 output from formula n-f+2*p/s+1 where n=28,f=3,p=0,s=1
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=0)
        # 1 → 16 → 32: as giving the network progressively more feature detectors
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=0)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        # each feature has 32*5*5 = 800 features, and we have 10 classes (digits 0-9) to classify, so the output layer has 10 neurons.
        self.fc=nn.Linear(in_features=32*5*5,out_features=10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
model=CNN()
loss_function=nn.CrossEntropyLoss()
# The problem is that the gradient can change direction a lot between batches. Adam keeps a memory of previous gradients and and uses them to make the movement smoother..
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# Each pixel represents brightness of the handwritten digit so have to convert the image formate into tensor formate to have that spatial arrangement or relationship between pixels.
# SGD:
# "Gradient says move this much → I'll move this much."
# Batch 1 → gradient → update
# Batch 2 → gradient → update
# Batch 3 → gradient → update
# Adam:
# "Let me look at the current gradient + what happened before
# and decide a suitable step."
# Previous gradients ──┐
#                      ↓
# Current gradient ─→  Adam → better direction + suitable step
#                      ↓
#                   update weight
# Previous gradient:  ↓
# Current gradient:   ↑

# then Adam knows the direction has changed and smooths the update rather than blindly following only the current gradient.

# And Adam also looks at the size of the gradients to decide how large the update should be.
transform=transforms.ToTensor()
# Create dataset
train_data = datasets.MNIST(
    root="data",
    train=True,
    # training dataset load
    download=True,
    transform=transform
    # converts each image from an image format into a PyTorch tensor
)
test_data=datasets.MNIST(
    train=False,
    root="data",
    download=True,
    transform=transform
    # download automatically at first time
)
# As the dataset is large, we will use DataLoader to load the data in batches. DataLoader load 64 images at a time
train_loader=DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True
)
test_loader=DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False
)
images,labels=next(iter(train_loader))
# give number of images beging processed at a time, number of channels, height and width of each image.
print(images.shape)
print(labels.shape)
print(images.dtype)
print(labels.dtype)
for epoch in range(5):
    for X, Y in train_loader:
        prediction=model(X)
        loss=loss_function(prediction,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# MNIST
#        ↓
# ToTensor
#        ↓
# 64 × 1 × 28 × 28
#        ↓
# Conv1: 1 → 16
#        ↓
# 64 × 16 × 26 × 26
#        ↓
# ReLU
#        ↓
# MaxPool 2×2
#        ↓
# 64 × 16 × 13 × 13
#        ↓
# Conv2: 16 → 32
#        ↓
# 64 × 32 × 11 × 11
#        ↓
# ReLU
#        ↓
# MaxPool 2×2
#        ↓
# 64 × 32 × 5 × 5
#        ↓
# Flatten
#        ↓
# 64 × 800
#        ↓
# Linear: 800 → 10
#        ↓
# 64 × 10
#        ↓
# CrossEntropyLoss
#        ↓
# Backward
#        ↓
# Adam updates parameters

# Adam
# Same direction repeatedly
#         ↓
#    confidence ↑
#         ↓
#    smooth movement


# Direction suddenly changes
#         ↓
#    use history
#         ↓
#    don't react blindly


# New direction continues
#         ↓
#  history gradually changes
#         ↓
#  move toward new direction
