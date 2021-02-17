import torch
import numpy as np
import matplotlib.pyplot as plt
dtype = torch.cuda.FloatTensor

def fitness(x,y):
    return -torch.abs(torch.sin(x)*torch.cos(y)*torch.exp(torch.abs(1-(torch.sqrt(x**2+y**2))/(3.1415))))

def velocity(v, gxbest, pxbest, pybest, x, pop, w, c1, c2):
    return w*torch.rand(pop).type(dtype)*v + \
       c1*torch.rand(pop).type(dtype)*(pxbest - x) + \
       c2*torch.rand(pop).type(dtype)*(gxbest.expand(x.size(0)) - x)

def PSO(pop, xmax, xmin, niter, wmax, wmin, c1, c2):

    vx                 = torch.rand(pop).type(dtype)
    vy                 = torch.rand(pop).type(dtype)
    best               = np.zeros(niter)
    x                 = (xmax -  xmin)*torch.rand(pop).type(dtype) + xmin
    y                 = (xmax -  xmin)*torch.rand(pop).type(dtype) + xmin
    z                 = fitness(x,y)
    [minz, indexminz] = z.min(0)
    gxbest            = x[indexminz] 
    gybest            = y[indexminz] 
    pxbest            = x
    pybest            = y
    pzbest            = z

    for K in range(niter):
        w      = wmax - ((wmax - wmin) / niter) * (K)
        vnextx = velocity(vx, gxbest, pxbest, pybest, x, pop, w, c1, c2)
        xnext  = x + vnextx
        vnexty = velocity(vy, gxbest, pxbest, pybest, y, pop, w, c1, c2)
        ynext  = y + vnexty

        xnext = xnext.cpu()
        ynext = ynext.cpu()
        idxmax        = (xnext > xmax) # elements that are bigger that upper limit
        idxmim        = (xnext < xmin)  # elements that are smaller that upper limit
        xnext[idxmax] = xmax
        xnext[idxmim] = xmin
        idymax        = (ynext > xmax) # elements that are bigger that upper limit
        idymim        = (ynext < xmin)  # elements that are smaller that upper limit
        ynext[idymax] = xmax
        ynext[idymim] = xmin

        xnext = xnext.cuda() 
        ynext = ynext.cuda() 

        znext = fitness(xnext,ynext)

        [minznext, indexminznext]  = znext.min(0)

        if (minznext[0] < minz[0]):
            minz   = minznext
            gxbest = xnext[indexminznext]
            gybest = ynext[indexminznext]

        indexpbest         = (znext < pzbest)
        pxbest[indexpbest] = xnext[indexpbest]
        pybest[indexpbest] = ynext[indexpbest]
        pzbest[indexpbest] = znext[indexpbest]
        x                  = xnext
        y                  = ynext
        vx                 = vnextx
        vy                 = vnexty

        best[K] = minz.cpu().numpy()
    return gxbest, gybest , minz, best

pop, xmax, xmin, niter = 10000, 10, -10, 10
wmax = 0.9
wmin = 0.4
c1   = 2.05
c2   = 2.05
xbest, ybest, fitbest, best = PSO(pop, xmax, xmin, niter, wmax, wmin, c1, c2)
print(xbest)
print(ybest)
print(fitbest)
t = np.linspace(0,niter,niter)
plt.plot(t, best, 'k.-')
plt.show()