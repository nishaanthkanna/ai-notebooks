import torch
from torch import nn
import sys
sys.path.insert(0, "..")
from helper import train, test
from data import get_dataloader
import torch.nn.functional as F
from time import time

device = "cuda" if torch.cuda.is_available() else "mps"

class SmallBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(SmallBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        # implement a block with two layers and a residual connection
        if in_channels != out_channels:
            stride = 2
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # First channel
        if in_channels == 3:
            stride = 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )   
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x
        if self.in_channels == 3:
            output = self.block(x)
            return self.relu(output)

        elif (self.in_channels == self.out_channels):
            output = self.block(x)
            return self.relu(output + x)
            
        else:
            output = self.block(x)
            x = self.conv1x1(x)
            return self.relu(output + x)


class ResNet(nn.Module):
    
    def __init__(self, in_channel = 3, out_channel = 64, bottleneck = True ,layers = [4, 4, 4, 4]):
        
        super(ResNet, self).__init__()

        # if not bottleneck:
        #     resnet = []
        #     for layer in layers:
        #         for n in range(layer):
        #             resnet.append(SmallBlock(in_channel, out_channel))
        #             in_channel = out_channel
        #         out_channel = in_channel * 2
            
        #     resnet.append(nn.AvgPool2d(kernel_size=3, stride=1))
        #     self.resnet = nn.Sequential(*resnet)

        #     self.classifier = nn.Sequential(
        #         nn.Linear(in_features=2048, out_features=2048),
        #         nn.ReLU(),
        #         nn.Linear(in_features=2048, out_features=1024),
        #         nn.ReLU(),
        #         nn.Linear(in_features=1024, out_features=10)
        #     )   

        
        self.resnet = nn.Sequential(
            # Input Dimension 32x32x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Output Dimension 32x32x64

            # Input Dimension 32x32x64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Output Dimension 16x16x128

            # Input Dimension 16x16x128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Output Dimension 8x8x256

            # Input Dimension 8x8x256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Output Dimension 4x4x512

            nn.AvgPool2d(kernel_size=3, stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10)
        ) 

    def forward(self, x):
        
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
        


# get the data
train_dl, test_dl = get_dataloader("cifar", batch_size=128)
# Training the model
model = ResNet(3, 64,bottleneck=False ,layers=[4,4,4,4]).to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 10

for t in range(epochs):
    start = time()
    print(f"Epoch {t+1}\n---")
    train(train_dl, model, loss_fn, optimizer, device)
    test(test_dl, model, loss_fn, device)
    print(f"Total time taken: {(time()-start):>0.1f} seconds")
    
