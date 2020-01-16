# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:39:40 2020

@author: fayya
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

class Perceptron:
    def __init__(self,alpha = 0.1, epochs = 200):
        self.alpha = alpha
        self.epochs = epochs
        self.W = np.array([0])
        self.bias = np.random.randn()
        self.Lambda = 0.5 #not used in perceptron
    def fit(self,Xtr,Ytr):
        d = Xtr.shape[1]
        self.W = np.random.randn(d)          
        for e in range(self.epochs):
            finished = True
            for i,x in enumerate(Xtr):
                if self.score(np.atleast_2d(x))*Ytr[i]<0.0:#self.predict(np.atleast_2d(x))!=Ytr[i]:
                    ##
                    self.W += self.alpha*Ytr[i]*x
                    self.bias += self.alpha*Ytr[i]
            #self.W = self.W-self.alpha*self.Lambda*self.W           
             
    def score(self,x):
        return np.dot(x,self.W) + self.bias
        
    def predict(self,x):
        return np.sign(self.score(x))
    
if __name__=='__main__':
    from plotit import plotit
    Xtr = np.array([[-1,0],[0,1],[4,4],[2,3]])
    ytr = np.array([-1,-1,+1,+1])
    clf = Perceptron()
    clf.fit(Xtr,ytr)
    z = clf.score(Xtr)
    print("Prediction Scores:",z)
    y = clf.predict(Xtr)
    print("Prediction Labels:",y)
    plotit(Xtr,ytr,clf=clf.score,conts=[0],extent = [-5,+5,-5,+5])
    
    