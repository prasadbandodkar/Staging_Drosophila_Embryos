# import standard libraries
#
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision as tv

# import local modules
#
from Python.sdata import TorchDataset
from Python.cvimage import CVImage, NuclearLayer


# specify paths
#
project_folder = "C:\\Users\\Prasad.Bandodkar\\Biotechne\\Projects\\staging"
data_folder = os.path.join(project_folder, "data")

# training parameters
#  
image_length = 128
shufflenum   = 100
nBatchTrain  = 25
nBatchVal    = 25

# define train, val, test data
#
test = [6,7]
val  = [8,9]
ignore = [26, 31, 20, 21, 22, 23, 24, 25, 27]

# define model name
#
current_date = datetime.now().strftime("%Y%m%d")
model_name = f"staging_nl_{current_date}_test_{'-'.join(map(str, test))}_val_{'-'.join(map(str, val))}_ignore_{'-'.join(map(str, ignore))}"
print(f"Model name: {model_name}")

# create data generator
#
size        = (512, 512)
padding     = 44
npoints     = 60
inward      = 40
outward     = -24
trunc_width = 128

# create torch dataset
#
train_dataset = TorchDataset(data_folder, test, val, ignore, type='train',
                             size=size, padding=padding, npoints=npoints, 
                             inward=inward, outward=outward, trunc_width=trunc_width)
test_dataset  = TorchDataset(data_folder, test, val, ignore, type='test',
                             size=size, padding=padding, npoints=npoints, 
                             inward=inward, outward=outward, trunc_width=trunc_width)
val_dataset   = TorchDataset(data_folder, test, val, ignore, type='val',
                             size=size, padding=padding, npoints=npoints, 
                             inward=inward, outward=outward, trunc_width=trunc_width)

# print a few data from train dataset
#
print("Train dataset")
data_iter = iter(train_dataset)
image, id = next(data_iter)
print(f"Image shape: {image.shape}") # type: ignore
print(f"Image id: {id}")

# print a few data from test dataset
#
# print("Test dataset")
# data_iter = iter(test_dataset)
# image, id = next(data_iter)
# print(f"Image shape: {image.shape}") # type: ignore
# print(f"Image id: {id}")

# display image
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Image ID: {id}')
plt.axis('off')
plt.show()


# ############################################################
# Model
# ############################################################

# A PyTorch CNN model for embryo stage classification
# Architecture:
# - Input: grayscale image (64 x 128 x 1)
# - Conv layers with increasing channels (16->32->64->128) 
# - Max pooling and batch norm after each conv
# - Global average pooling to reduce spatial dims
# - Fully connected layers to output stage prediction
#

# ... existing code ...

class StagingNet(torch.nn.Module):
    def __init__(self):
        super(StagingNet, self).__init__()
        
        # Convolutional layers
        self.conv_layers = torch.nn.Sequential(
            # First conv block (16 channels)
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Second conv block (32 channels)
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Third conv block (64 channels)
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Fourth conv block (128 channels)
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        
        # Global average pooling
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),  # Single output for regression
            torch.nn.Sigmoid()       # Constrains output between 0 and 1
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Create model instance
model = StagingNet()

# Print model dimension changes
def print_model_summary(input_size=(1, 1, 64, 128)):
    print("\nModel Dimension Analysis:")
    print("-" * 80)
    print(f"Input size: {input_size}")
    
    x = torch.randn(input_size)
    
    # Conv layers
    x = model.conv_layers[0:4](x)  # First conv block
    print(f"After conv block 1: {x.shape} -> 16 channels, dims halved by maxpool")
    
    x = model.conv_layers[4:8](x)  # Second conv block
    print(f"After conv block 2: {x.shape} -> 32 channels, dims halved by maxpool")
    
    x = model.conv_layers[8:12](x)  # Third conv block
    print(f"After conv block 3: {x.shape} -> 64 channels, dims halved by maxpool")
    
    x = model.conv_layers[12:16](x)  # Fourth conv block
    print(f"After conv block 4: {x.shape} -> 128 channels, dims halved by maxpool")
    
    x = model.gap(x)
    print(f"After global avg pooling: {x.shape} -> spatial dims reduced to 1x1")
    
    x = x.view(x.size(0), -1)
    print(f"After flatten: {x.shape}")
    
    x = model.fc_layers(x)
    print(f"Final output: {x.shape} -> 9 class probabilities")

print_model_summary()



# ############################################################
# Training
# ############################################################

# Training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nBatchTrain, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=nBatchVal, shuffle=False)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training phase
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images = images.float().to(device)
        targets = targets.float().to(device)  # Changed to float for regression
        # print the first image diemnsion and type
        print(f"images.shape: {images.shape}, images.dtype: {images.dtype}")
        print(f"targets.shape: {targets.shape}, targets.dtype: {targets.dtype}")
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Train MSE: {avg_train_loss:.6f}")
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.float().to(device)
            targets = targets.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation MSE: {avg_val_loss:.6f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'{model_name}_best.pth')