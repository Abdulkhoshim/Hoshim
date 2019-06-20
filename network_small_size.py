from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')

import torch
import torch.nn as nn
import torch.nn.functional as F

import paras
from torch_deform_conv.layers import ConvOffset2D

USE_GPU = paras.USE_GPU


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # 2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        #3
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #4
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        #5
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool5 = nn.MaxPool2d(2,2)
        # self.pool5 = nn.AvgPool2d(6,6)

        self.conv61 = nn.Conv2d(256, 1024, kernel_size=1,)
        self.conv62 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv63 = nn.Conv2d(1024, 2, kernel_size=1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input):

        # branch 1
        x1 = self.conv1(input)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)

        x1 = self.pool2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)

        x1 = self.pool3(x1)
        x1 = self.conv4(x1)
        x1 = self.bn4(x1)
        x1 = F.relu(x1)

        x1 = self.pool4(x1)
        x1 = self.conv5(x1)
        x1 = self.bn5(x1)
        x1 = F.relu(x1)

        # print(type(self.pool5))

        x1 = self.pool5(x1)
        x1 = F.relu(x1)

        # mainstream
        # x = torch.cat((x1, x2), 1)
        x = x1.view(-1, 256,1,1)
        x = self.conv61(x)
        x = F.relu(x)
        x = self.conv62(x)
        x = F.relu(x)
        x = self.conv63(x)
        x = x.squeeze()
        # x = self.softmax()
        # print(x.shape)
        return x