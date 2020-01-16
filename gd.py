import numpy as np

def gd(f,df,x0=0.0,lr = 0.01,eps=1e-4,nmax=1000, history = True):
    """
    Implementation of a gradient descent solver.
        f: function, f(x) returns value of f(x) for a given x
        df: gradient function df(x) returns the gradient at x
        x0: initial position [Default 0.0]
        lr: learning rate [0.001]
        eps: min step size threshold [1e-4]
        nmax: maximum number of iters [1000]
        history: whether to store history of x or not [True]
    Returns:
        x: argmin_x f(x)
        converged: True if the final step size is less than eps else false
        H: history
    """
    H = []
    x = x0
    if history:
        H = [[x,f(x)]]
    for i in range(nmax):
        dx = -lr*df(x) #gradient step
        if np.linalg.norm(dx)<eps: # if the step taken is too small, we have converged
            break
        if history:
            H.append([x+dx,f(x+dx)])
        x = x+dx #gradient update
    converged = np.linalg.norm(dx)<eps        
    return x,converged,np.array(H)
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    def f(x):
        y = (x-0.5)**2
        return y
    def df(x):
        dy = 2*(x-0.5)#+3*np.cos(3*x)
        return dy
    

    z = np.linspace(-3,3,100)
    #select random initial point in the range
    x0 = np.min(z)+(np.max(z)-np.min(z))*np.random.rand()
    
    x,c,H = gd(f,df,x0=x0,lr = 0.01,eps=1e-4,nmax=1000, history = True) 
    
    plt.plot(z,f(z)); plt.plot(z,df(z));
    plt.legend(['f(x)','df(x)'])
    plt.xlabel('x');plt.ylabel('value')
    s = 'Convergence in '+str(len(H))+' steps'
    if not c:
        s = 'No '+s
    plt.title(s)
    plt.plot(H[0,0],H[0,1],'ko',markersize=10)
    plt.plot(H[:,0],H[:,1],'r.-')
    plt.plot(H[-1,0],H[-1,1],'k*',markersize=10)    
    plt.grid(); plt.show()