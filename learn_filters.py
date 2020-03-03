# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:06:41 2020

@author: fayya
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        self.conv1 = nn.Conv2d(1,1, 5)  

    def forward(self, x):
        x = self.conv1(x) 
        #perform convolution
        return x

import skimage
X = skimage.io.imread('in.jpg')/255
T = np.zeros(X.shape,dtype=np.float) #create the target by putting 1.0 at target object locations
T[21,121]=1.0
T[34,36] =1.0
T[64,78] =1.0
T[83,142]=1.0

T = T[2:-2,2:-2] # reduce target filter size to compensate for padding loss in convolution
f = Filter()
optimizer = torch.optim.SGD(f.parameters(), lr=1e-1)
T_torch = torch.from_numpy(T).float()
X_torch = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(X).float(),0),0)
L = []
for _ in range(500):
    Z_torch = f(X_torch).squeeze()    
    Z_torch = torch.sigmoid(Z_torch) #output
    #Z_torch = (Z_torch-torch.min(Z_torch))/(torch.max(Z_torch)-torch.min(Z_torch)) #rescale
    loss = torch.mean((T_torch-Z_torch)**2) #error
    optimizer.zero_grad() #optimization
    loss.backward()
    optimizer.step()
    L.append(loss.item())

output = Z_torch.squeeze().detach().numpy()
output = output**2 #contrast stretching
output = (output-np.min(output))/(np.max(output)-np.min(output)) #rescale

plt.figure();plt.imshow(X,cmap='gray');plt.title('input');plt.colorbar()
plt.figure();plt.imshow(T,cmap='gray');plt.title('target');plt.colorbar()
plt.figure();plt.imshow(output,cmap='gray');plt.title('output');plt.colorbar()
plt.figure();plt.imshow(output>0.8,cmap='gray');plt.title('thresholded output')
plt.figure();plt.plot(np.log10(L));plt.title('loss function')
plt.figure();plt.imshow(f.conv1.weight.squeeze().detach().numpy(),cmap='gray');plt.title('learned filter');plt.colorbar()