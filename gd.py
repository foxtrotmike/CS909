import numpy as np

def gd(f,df,v0=0.0,lr = 0.01,eps=1e-4,nmax=1000, history = True):
    """
    Implementation of a (single variable) gradient descent solver.
        f: function, f(v) returns value of f(v) for a given v
        df: gradient function df(v) returns the gradient at v
        v0: initial position [Default 0.0]
        lr: learning rate [0.001]
        eps: min step size threshold [1e-4]
        nmax: maximum number of iters [1000]
        history: whether to store history of x or not [True]
    Returns:
        v: argmin_v f(v)
        converged: True if the final step size is less than eps else false
        H: history
    """
    H = []
    v = v0
    if history:
        H = [[v,f(v)]]
    for i in range(nmax):
        dv = -lr*df(v) #gradient step
        if np.linalg.norm(dv)<eps: # if the step taken is too small, we have converged
            break
        if history:
            H.append([v+dv,f(v+dv)])
        v = v+dv #gradient update
    converged = np.linalg.norm(dv)<eps        
    return v,converged,np.array(H)
def plotGD(v,f,df,H,c):
    import matplotlib.pyplot as plt    
    """
    Just plotting code 
    """
    plt.figure()
    vplot = np.sort(np.hstack((v,H[:,0]))) #just ensure that the range of v over the optimization history is included in plotting
    plt.plot(vplot,f(vplot)); plt.plot(vplot,df(vplot));
    plt.legend(['f','df'])
    plt.xlabel('variable');plt.ylabel('value')
    s = 'Convergence in '+str(len(H))+' steps'
    if not c:
        s = 'No '+s
    plt.title(s)
    plt.plot(H[0,0],H[0,1],'ko',markersize=10)
    plt.plot(H[:,0],H[:,1],'r.-')
    plt.plot(H[-1,0],H[-1,1],'k*',markersize=10)    
    plt.grid(); plt.show()
    
if __name__=='__main__':
    
    def f(v):
        return np.sin(3*v)-v        
        
    def df(v):
        return 3*np.cos(3*v)-1        
        return vectorized_dloss(v)
       
    v = np.linspace(-3,3,1000)
    
    #select random initial point in the range
    v0 = np.min(v)+(np.max(v)-np.min(v))*np.random.rand()
    vout,c,H = gd(f,df,v0=v0,lr = 0.001,eps=1e-4,nmax=1000, history = True)    
    plotGD(v,f,df,H,c)
    
    #%% ignore in first attempt and solve a simpler optimization problem
    # Implementation of a very simple perceptron using the gradient descent function
    X = np.array([[5],[-10]]) 
    Y = np.array([-1,1])
    def loss(y,x,w): return max(0.0,1-y*x*w) #hinge loss in terms of y,x and w
    def loss_w(w): return np.mean([loss(Y[i],X[i],w) for i in range(len(Y))]) # loss in terms of w only
    
    vectorized_loss = np.vectorize(loss_w)  
    def dloss(y,x,w): #derivate of loss with respect to w for a given x and y
        if (1-y*w*x)<0:
            return 0.0
        else:
            return -y*x
    def dloss_w(w): return np.mean([dloss(Y[i],X[i],w) for i in range(len(Y))]) #derivative of the loss with respect to w only
    vectorized_dloss = np.vectorize(dloss_w)
    f = vectorized_loss
    df = vectorized_dloss
    vout,c,H = gd(f,df,v0=v0,lr = 0.001,eps=1e-4,nmax=1000, history = True)       
    plotGD(v,f,df,H,c)
