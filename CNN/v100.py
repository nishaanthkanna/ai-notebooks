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
        self.stride = 1
        # implement a block with two layers and a residual connection
        if in_channels != out_channels:
            self.stride = 2
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )   
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        if (self.stride == 2):
            x = self.conv1x1(x)
        output = self.block(x)
        return self.relu(output + x)

class BottleneckBlock(nn.Module):

    def __init__(self, first_channel, in_channels, out_channels, reduce=False) -> None:
        super(BottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 2 if reduce else 1
        
        # implement a block with three layers and a residual connection
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=first_channel, out_channels=in_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )   
        self.conv1x1 = nn.Conv2d(first_channel, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        output = self.block(x)
        x = self.conv1x1(x)
        return self.relu(output + x)


class ResNet(nn.Module):
    
    def __init__(self, bottleneck = True ,layers = [4, 4, 4, 4]):
        
        super(ResNet, self).__init__()

        if not bottleneck:
            resnet = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)]
            in_channel = 64
            out_channel = 64
            for layer in layers:
                for n in range(layer):
                    resnet.append(SmallBlock(in_channel, out_channel))
                out_channel = in_channel * 2
            
            resnet.append(nn.AvgPool2d(kernel_size=3, stride=1))
            self.resnet = nn.Sequential(*resnet)

            self.classifier = nn.Sequential(
                nn.Linear(in_features=2048, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=10)
            )
        else:
            # layers greater than 34
            resnet = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)]
            reduce=False
            in_channel = 64
            first_channel = 64
            out_channel = 256
            for layer in layers:
                for n in range(layer):
                    resnet.append(BottleneckBlock(first_channel ,in_channel, out_channel, reduce=reduce))
                    first_channel = out_channel
                    reduce=False
                reduce=True
                in_channel = in_channel * 2
                out_channel = out_channel * 2
            
            resnet.append(nn.AvgPool2d(kernel_size=3, stride=1))
            self.resnet = nn.Sequential(*resnet)

            self.classifier = nn.Sequential(
                nn.Linear(in_features=8192, out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=10)
            )


    def forward(self, x):
        
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
        

# get the data
train_dl, test_dl = get_dataloader("cifar", batch_size=128)
# Training the model
model = ResNet(bottleneck=True ,layers=[3,4,6,3]).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
 
epochs = 10

for t in range(epochs):
    start = time()
    print(f"Epoch {t+1}\n---")
    train(train_dl, model, loss_fn, optimizer, device)
    test(test_dl, model, loss_fn, device)
    print(f"Total time taken: {(time()-start):>0.1f} seconds")
    
