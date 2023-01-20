"""
Plots the expected and experimental contrast (difference between closest and farthest examples) with respect to dimenionsality "d"
Aggarwal, Charu C., Alexander Hinneburg, and Daniel A. Keim. “On the Surprising Behavior of Distance Metrics in High Dimensional Space.” In Database Theory — ICDT 2001, edited by Jan Van den Bussche and Victor Vianu, 420–34. Lecture Notes in Computer Science 1973. Springer Berlin Heidelberg, 2001. http://link.springer.com/chapter/10.1007/3-540-44503-X_27. 
"""

from numpy.random import rand,randint #importing randn
import matplotlib.pyplot as plt #importing plotting module
from scipy import spatial
import numpy as np
n = 100000
k = 2
da = np.arange(1,100,2)
va = []
for d in da:
    Xp = rand(n,d)
    Xnorm = np.linalg.norm(Xp,axis = 1, ord = k)
    M = np.max(Xnorm)
    m = np.min(Xnorm)
    va.append((M,m,M-m))
    
plt.plot(da,va,'o-');plt.plot(da,da**(1.0/k));plt.grid()
plt.legend(['Max','Min','Difference','Expected'])
plt.xlabel('dimnesions')
plt.ylabel('Value')
