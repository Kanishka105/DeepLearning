# CIFAR-10 which is the collection of 60000 images of 32*32 pixels with rgb of 3 channels
#MNIST dataset is a collection of 60000 images of handwritten digits (0-9) with 28*28 pixels and grayscale of 1 channel. 
# We use CIFAR-10 for real CNN design decision making and MNIST for learning and testing the CNN design.
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, random_split, Subset
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.conv3 = nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0
                )
        # self.conv4 = nn.Conv2d(
        #             in_channels=128,
        #             out_channels=256,
        #             kernel_size=3,
        #             stride=1,
        #             padding=0
        #         )

        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout(0.5)
        self.fc = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self, x):
# Conv block 1: conv2d--batchnorm--relu--maxpool: edges and simple pattern
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        # Reduce spatial size while keeping the strongest local features
        x = self.pool(x)
#conv block2: conv2d--batchnorm--relu--maxpool: textures and corners
        x = self.conv2(x)
        x = self.bn2(x)
        # Reduce spatial size while keeping the strongest local features
        x = torch.relu(x)
        x = self.pool(x)
# conv block3: conv2d--batchnorm--relu--maxpool: shapes and parts
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)  
        x=self.gap(x)
        x = torch.flatten(x, 1)
# Dropout: help all neuron to learn instead of lying on certain neuron.During training, it randomly turns off some neuron. To prevent overfitting.
        x=self.dropout(x)
        x = self.fc(x)

        return x
# Transformation : converts each image formate into a PyTorch tensor.Resize the image and Normalize it.
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# Here ToTensor() converts each image pixel value from 0-255 to 0-1. But it is not the same as normalization.
#Normalization means to scale pixel such that the mean is 0 and the standart deviation is 1.Calculate μ and σ from TRAINING DATA.
# :TIPS-------------------------------------------
# So the transform is applied when the sample is retrieved. ToTensor() happens when you retrieve an image. The transform is applied when an item is requested.
# images = torch.stack([image for image, label in train_data])
# mean = images.mean(dim=(0, 2, 3)) // dim means the parameter take for the mean here it is images height and width
# std = images.std(dim=(0, 2, 3))
#Resize: if 32*32 convert to 64*64=> Interpolation [Estimation of misssing pixel to create larger image]---Upsampling. DownSampling: 32*32 to 28*28[Some info lost]
# Type of resize: Nearest Neighbor, Bilinear[Looks at neighboring pixels and calculates a weighted average. e.g
#  A ───── B
# │   ?   │
# │       │
# C ───── D], Bicubic[more neighboring pixels and a more sophisticated interpolation calculation.], Lanczos
#Why need Resize: Beacause input image size of CNN is fixed. So if we want to use different size of images, we need to resize them to the same size.A CNN batch needs tensors with compatible dimensions. CIFAR 10 already have 32*32 pixels so no need to resize.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
      transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)
])
full_data=datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=None
)
# do need ToTensor() for the validation dataset and Data Augmentaion as well
# train_data, val_data = random_split(
#     full_data,
#     [45000, 5000]
# )
# Data Augmention applied to training transform. It inc variety of samples so that cnn learn efatures more robuts/generalization.
# train_data.dataset.transform = train_transform --Do not do this it will also transform the full data


generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(full_data),
    generator=generator
).tolist()


train_indices = indices[:45000]
val_indices = indices[45000:]
train_full = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=train_transform
)

val_full = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)
train_data = Subset(train_full, train_indices)
val_data = Subset(val_full, val_indices)

test_data=datasets.CIFAR10(
    download=True,
    root="data",
    train=False,
    transform=transform
)
train_loader=DataLoader(
    batch_size=64,
    dataset=train_data,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False
)
test_loader=DataLoader(
    batch_size=64,
    dataset=test_data,
    shuffle=False
)
# This creates your network with all the learnable parameters:
model=CNN()
# How wrong was the model's prediction compared with the correct class? and it is used for mutually multiple exclusive classes. 
loss_function=nn.CrossEntropyLoss()
# Adam adapts the learning rate for each parameter using information from current and previous gradients, which often makes optimization faster and more stable than basic SGD.
# Adam uses lr=0.001 as the base learning rate and adaptively adjusts the effective step size for each parameter
# Regularization: encourages model to find simpler solution rather an unnecessary complicated one and it is important technique for overfitting. Two types are weight decay and dropout.
# Weight Decay:Don't let the weights become unnecessarily large. To prevent Overfitting. With weight decay, conceptually we add a penalty
# "While updating the weights, also discourage the weights from becoming unnecessarily large."
# You don't want it too large because then the model may be forced to use weights that are too small and underfit.That is why we use 1e-4--strength of weight decay
# WITHOUT weight decay       WITH weight decay

# w = 10                     w = 10
#    ↓                          ↓
# gradient = 0.5             gradient = 0.5
#    ↓                          +
# update                     decay = 0.2
#    ↓                          ↓
# w = 9.95                   total = 0.7
#                               ↓
#                            w = 9.93
    #          LARGE WEIGHT
    #               ↓
    #          larger penalty
    #               ↓
    #       stronger pull toward 0
    #               ↓
    #       smaller weight values
    #               ↓
    #    less chance of overfitting
# Change the weight according to the data, but don't unnecessarily make the weight huge.
# Why larger weights get stronger pressure
# w = 2
# → decay effect = 0.04

# w = 10
# → decay effect = 0.20

# w = 12
# → decay effect = 0.24
# Weight decay helps the model reach weights that are smaller and often more stable/generalizable, which can make the learned solution smoother and reduce overfitting.
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
num_epochs = 50
# max train for 50 epochs
patience = 3
patience_counter = 0

best_val_loss = float("inf")
# Initially, we set the best validation loss to infinity.
for epoch in range(num_epochs):
    model.train()
    # Put my neural network into tarining mode.
    for X, Y in train_loader:
        prediction=model(X)
        loss=loss_function(prediction,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Early Stopping
# → controls the AMOUNT OF TRAINING
# Train epoch
#    ↓
# Calculate validation loss
#    ↓
# Is validation loss better?
#    ├── YES → save model + reset patience
#    └── NO  → increase patience
#                     ↓
#               patience reached?
#                     ↓
#                    STOP
# With early stopping + patience, you keep track of the best validation performance. if you have the best epoch at 3 and stop at 8 then still it will use the best validation loss epoch.
# Best validation loss → save it.
# Worse validation loss → increase patience.
# New best → reset patience.
# Patience limit → stop training and restore the best model.
    model.eval()
    # Put my neural network into evaluation/inference mode.
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader:
            prediction = model(X)
            loss = loss_function(prediction, Y)
            val_loss += loss.item()
            # print(loss) and got tensor(0.8234, grad_fn= <NllLossBackward0>) to get numerical value we have loss.items()
    val_loss /= len(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Early stopping tells you when to stop training; the saved best checkpoint tells you which model to actually keep
        # state_dict() give learnable parameters values of stored state
        # Take the current model's learned paramet ers and save them into a file
        # torch.load("best_model.pth") Used to read the saved weights from the file.
        # model.load_state_dict(torch.load("best_model.pth"))-- put those saved weights back into my current model.
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for X, Y in test_loader:
        prediction=model(X)
        loss=loss_function(prediction,Y)
        test_loss += loss.item()
        # Image 1 → [2.1, 0.3, 5.7, 1.2, ...]
        # Image 2 → [0.2, 7.8, 1.1, 0.5, ...]
        # Image 3 → [1.5, 0.4, 0.8, 6.2, ...]
        # Each column representsone class and these above are their scores 
        _, predicted = torch.max(prediction, 1)
        # Find the maximum across the class dimension. Number 1 means find the max across the class simensions.Img 1 max is 5.7 at index 2 so the predicated =2 then we finally get predicated=[2,1,3...] 
        # torch.max return two things: max value and the index of max value
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
test_loss /= len(test_loader)

accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
# print("Train data shape:",train_data.data.shape)
# Early Stopping: regularization technique. Keep training while validation performance improves. If it stops improving for a while, stop. 
# What is Patience in it? : do not stop after one bad epoch because validation loss can fluctuate. We have patience and wait till that number of epoch if my validation loss worse or can say Allow 3 consecutive epochs without improvement before stopping.
# Weight Decay
# → controls the SIZE of weights


# TIPS: Need Data augmentation for training not for validation/test.

                        #  ┌─────────────────────┐
                    #      │       DATASET       │
                    #      │ Images + Labels     │
                    #      └──────────┬──────────┘
                    #                 │
                    #                 ▼
                    #      ┌─────────────────────┐
                    #      │ DATA PREPROCESSING  │
                    #      │ Resize              │
                    #      │ Convert to Tensor   │
                    #      │ Normalize           │
                    #      └──────────┬──────────┘
                    #                 │
                    #                 ▼
                    # ┌────────────────────────────┐
                    # │     DATA AUGMENTATION      │
                    # │ Random Flip                │
                    # │ Random Rotation            │
                    # │ Random Crop                │
                    # │ Color Jitter               │
                    # └─────────────┬──────────────┘
                    #               │
                    #               ▼
                    # ┌────────────────────────────┐
                    # │       CUSTOM CNN           │
                    # │                            │
                    # │ Conv Block 1               │
                    # │ Conv Block 2               │
                    # │ Conv Block 3               │
                    # │ Conv Block 4               │
                    # │                            │
                    # │ Feature Extraction         │
                    # └─────────────┬──────────────┘
                    #               │
                    #               ▼
                    # ┌────────────────────────────┐
                    # │    CLASSIFICATION HEAD     │
                    # │ Global Average Pooling     │
                    # │ Fully Connected Layer      │
                    # │ Dropout                    │
                    # │ Output Classes             │
                    # └─────────────┬──────────────┘
                    #               │
                    #               ▼
                    # ┌────────────────────────────┐
                    # │         TRAINING           │
                    # │ Forward Pass               │
                    # │ Loss                       │
                    # │ Backpropagation            │
                    # │ Optimizer                  │
                    # │ Weight Update              │
                    # └─────────────┬──────────────┘
                    #               │
                    #               ▼
                    # ┌────────────────────────────┐
                    # │      REGULARIZATION        │
                    # │ Data Augmentation          │
                    # │ Dropout                    │
                    # │ Batch Normalization        │
                    # │ Weight Decay               │
                    # │ Early Stopping             │
                    # └─────────────┬──────────────┘
                    #               │
                    #               ▼
                    # ┌────────────────────────────┐
                    # │        EVALUATION          │
                    # │ Accuracy                   │
                    # │ Loss Curves                │
                    # │ Confusion Matrix           │
                    # │ Misclassified Images       │
                    # └────────────────────────────┘


    #                      CUSTOM CNN PROJECT
    #                               │
    #    ┌──────────────────────────┼─────────────────────────┐
    #    │                          │                         │
    #    ▼                          ▼                         ▼
    #  DATA                    CNN DESIGN                 TRAINING
    #    │                          │                         │
    #    ├─ Dataset                ├─ Conv2D                ├─ Forward
    #    ├─ Labels                 ├─ Filters               ├─ Loss
    #    ├─ Split                  ├─ Kernels               ├─ Backprop
    #    ├─ Normalize              ├─ Stride                ├─ Optimizer
    #    └─ Augment                ├─ Padding               └─ Epochs
    #                              ├─ ReLU
    #                              ├─ BatchNorm
    #                              └─ Pooling
    #                                   │
    #                                   ▼
    #                           FEATURE EXTRACTION
    #                                   │
    #                                   ▼
    #                            CLASSIFICATION HEAD
    #                                   │
    #                           ┌───────┴────────┐
    #                           ▼                ▼
    #                      GAP/Flatten       Fully Connected
    #                                            │
    #                                            ▼
    #                                         Dropout
    #                                            │
    #                                            ▼
    #                                        Classes


    #                      GENERALIZATION
    #                            │
    #         ┌──────────────────┼──────────────────┐
    #         ▼                  ▼                  ▼
    #    Augmentation         Dropout          Weight Decay
    #         │
    #         ▼
    #    More Data Variation


    #                       EVALUATION
    #                            │
    #       ┌────────────────────┼────────────────────┐
    #       ▼                    ▼                    ▼
    #    Accuracy            Loss Curves       Confusion Matrix
    #                                                   │
    #                                                   ▼
    #                                            Error Analysis

