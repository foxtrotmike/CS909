# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:06:41 2020

@author: fayya
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        self.conv1 = nn.Conv2d(1,1, 3)  
    def setKernel(self,K):
        #self.conv1 = None
        #(in_channels, out_channels, kernel_size)
        K = torch.from_numpy(K).float() 
        #move from numpy to torch
        K = torch.unsqueeze(torch.unsqueeze(K,0),0) 
        #add extra dimensions for in_channels and out_channels
        self.conv1.weight.data = 0*self.conv1.weight.data + K 
        #set the kernel as weights (done this way to avoid data type changes)
        self.conv1.bias.data = 0*self.conv1.bias.data 
        # no bias
        return self
    def forward(self, x):
        x = self.conv1(x) 
        #perform convolution
        return x

plt.close('all')
from skimage import data
X = data.camera(); plt.imshow(X,cmap='gray',vmin=0,vmax=255)

K = np.array([[0 ,1, 0],[1,-4,1], [0, 1 ,0]])

X_torch = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(X).float(),0),0) 
#move image to torch
f = Filter().setKernel(K) 
#set the kernel in Filter object
Z_torch = f(X_torch) 
#convolution
Z = Z_torch.squeeze().detach().numpy() 
#move back to numpy

plt.figure();plt.imshow(Z,cmap='gray',vmin=0,vmax=255)
