## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        # ((224 - 3)/1) + 1
        # output = 32 x 222 x 222
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # output = X x n/3 x n/3
        self.pool = nn.MaxPool2d(3, 3)
        # output = X x n/2 x n/2
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # input = 32 x 74 x 74
        # ((74 - 3)/1) + 1 = 72
        # output = 64 x 72 x 72
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # input = 64 x 24 x 24
        # ((24 - 3) / 1) + 1 = 22
        # output = 128 * 22 * 22
        # after pool output = 128 x 7 x 7
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*7*7, 512)
        self.dense1_bn = nn.BatchNorm1d(512)
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        
        # Flattenning
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
