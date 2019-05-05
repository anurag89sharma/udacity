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
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # ((224 - 5)/1) + 1
        # output = 32 x 220 x 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # output = X x n/3 x n/3
        self.pool = nn.MaxPool2d(3, 3)
        
        # input = 32 x 73 x 73
        # ((73 -5)/1) + 1 = 69
        # output = 64 x 69 x 69
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # input = 64 x 23 x 23
        # ((23 - 5) / 1) + 1 = 19
        # output = 128 * 19 * 19
        # after pool output = 128 x 6 x 6
        #self.conv3 = nn.Conv2d(64, 128, 5)
        #self.conv3_bn = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(64*23*23, 512)
        self.dense1_bn = nn.BatchNorm1d(512)
        self.fc1_drop = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        
        # Flattenning
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # final output
        # x = F.log_softmax(x, dim=1)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
