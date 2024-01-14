# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:14:23 2024

@author: fayya
"""

from numpy.random import randn #importing randn
import numpy as np #importing numpy
import matplotlib.pyplot as plt #importing plotting module
import itertools
import warnings

def plotit(X,Y=None,clf=None,  conts = None, ccolors = ('b','k','r'), colors = ('c','y'), markers = ('s','o'), hold = False, transform = None,extent = None,**kwargs):
    """
    A function for showing data scatter plot and classification boundary
    of a classifier for 2D data
        X: nxd  matrix of data points
        Y: (optional) n vector of class labels
        clf: (optional) classification/discriminant function handle
        conts: (optional) contours (if None, contours are drawn for each class boundary)
        ccolors: (optional) colors for contours   
        colors: (optional) colors for each class (sorted wrt class id)
            can be 'scaled' or 'random' or a list/tuple of color ids
        markers: (optional) markers for each class (sorted wrt class id)
        hold: Whether to hold the plot or not for overlay (default: False).
        transform: (optional) a function handle for transforming data before passing to clf
        kwargs: any keyword arguments to be passed to clf (if any)        
    """
    if clf is not None and X.shape[1]!=2:
        warnings.warn("Data Dimensionality is not 2. Unable to plot.")
        return
    if markers is None:
        markers = ('.',)
    eps=1e-6
    d0,d1 = (0,1)
    if extent is None:
        minx, maxx = np.min(X[:,d0])-eps, np.max(X[:,d0])+eps
        miny, maxy = np.min(X[:,d1])-eps, np.max(X[:,d1])+eps
        extent = [minx,maxx,miny,maxy]
    else:
        [minx,maxx,miny,maxy] = extent
    if Y is not None:
        classes = sorted(set(Y))
        if conts is None or len(conts)<2:
            #conts = list(classes)
            vmin,vmax = classes[0]-eps,classes[-1]+eps
        else:            
            vmin,vmax= np.min(conts)-eps,np.max(conts)+eps
        
    else:
        vmin,vmax=-2-eps,2+eps
        if conts is None or len(conts)<2:            
            conts = sorted([-1+eps,0,1-eps])
        else:
            vmin,vmax= np.min(conts)-eps,np.max(conts)+eps
        
    if clf is not None:
        npts = 150
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf(t,**kwargs)
        
        z = np.reshape(z,(npts,npts)).T        
        
        
        plt.contour(x,y,z,conts,linewidths = [2],colors=ccolors,extent=extent, label='f(x)=0')
        #plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z,cmap=plt.cm.Purples,vmin=vmin,vmax=vmax);plt.colorbar()
        plt.axis(extent)
    
    if Y is not None:        
        for i,y in enumerate(classes):
            
            if colors is None or colors=='scaled':
                cc = np.array([[i,i,i]])/float(len(classes))
            elif colors =='random':
                cc = np.array([[np.random.rand(),np.random.rand(),np.random.rand()]])
            else:
                cc = colors[i%len(colors)]
            
            mm = markers[i%len(markers)]
            plt.scatter(X[Y==y,d0],X[Y==y,d1], marker = mm,c = cc, s = 50)     
         
    else:
        plt.scatter(X[:,d0],X[:,d1],marker = markers[0], c = 'k', s = 5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    if not hold:
        plt.grid()        
        plt.show()
    return extent


import imageio.v2 as imageio
import os

class Perceptron:       
    def __init__(self,alpha = 0.1, epochs = 200,save_gif = True):
        self.alpha = alpha
        self.epochs = epochs
        self.W = np.array([0])
        self.bias = np.random.randn()*0
        self.Lambda = 0.5
        self.save_gif = save_gif
    def fit(self,Xtr,Ytr):
        d = Xtr.shape[1]
        self.W = np.random.randn(d)   #always start at zero or a random weight vectir
        
        if self.save_gif:
            plt.figure()
            extent = [-1,+2,-1,+2]
            plotit(Xtr,ytr,clf=self.score,conts=[0],extent = extent)
            plt.title('epoch = '+str(0)+': w = '+(', '.join([f'{num:.3f}' for num in self.W]))+' b = '+f'{self.bias:.3f}')
            ii = 0; filenames = [str(ii)+'.jpg']; plt.savefig(filenames[-1]); plt.close()
        
        for e in range(self.epochs):            
            finished = True
            for i,x in enumerate(Xtr):
 
                if Ytr[i]*self.predict(np.atleast_2d(x))<1: #if error, use 1 on the RHS instead of 0 to implement hinge loss
                    finished = False
                    self.W += self.alpha*Ytr[i]*x
                    self.bias += self.alpha*Ytr[i]     
                if self.save_gif:
                    plt.figure()
                    plotit(Xtr,ytr,clf=self.score,conts=[0],extent = extent)
                    plt.plot(x[0],x[1],'r*')
                    plt.title('epoch = '+str(e)+': w = '+(', '.join([f'{num:.3f}' for num in self.W]))+' b = '+f'{self.bias:.3f}')
                    ii +=1; filenames.append(str(ii)+'.jpg'); plt.savefig(filenames[-1]); plt.close()
            if finished: break
        """
        #useful for l onger iterations
        with imageio.get_writer('my_plots.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                os.remove(filename)
                writer.append_data(image,duration = 0.3)
        """
        if self.save_gif:
            imageio.mimsave('my_animation.gif', [imageio.imread(f) for f in filenames], duration=0.5)
            for f in filenames: os.remove(f)

    def score(self,x):
        return np.dot(x,self.W) + self.bias
        
    def predict(self,x):
        return np.sign(self.score(x))    
    
if __name__=='__main__':
    Xtr = np.array([[0,0],[0,1],[1,0],[1,1]])
    ytr = np.array([-1,-1,-1,+1])
    clf = Perceptron(alpha = 0.1)
    clf.fit(Xtr,ytr)
    z = clf.score(Xtr)
    print("Prediction Scores:",z)
    y = clf.predict(Xtr)
    print("Prediction Labels:",y)
    plotit(Xtr,ytr,clf=clf.score,conts=[0],	extent = [-1,+2,-1,+2])
