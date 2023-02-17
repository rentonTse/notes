import numpy as np

def size(I,Ng,cen,rg):
    '''
    return se = size and ellip 
    '''
    se = np.zeros((1,4))
    w = 0.0
    w11 = 0.0
    w12 = 0.0
    w22 = 0.0
    scal = 0.5/(rg*rg)
    for i in range(Ng):
        for j in range(Ng):
            x = i-cen[0]
            y = j-cen[1]
            r2 = x*x+y*y
            weight = np.exp(-r2*scal)
            w    += I[i][j]*weight
            w11  += x*x*I[i][j]*weight
            w12  += x*y*I[i][j]*weight
            w22  += y*y*I[i][j]*weight
    w11 /= w
    w12 /= w
    w22 /= w
    # size
    se[0] = w11+w22
    # e1 
    se[1] = e1 = (w11-w22)/(w11+w22)
    # e2
    se[2] = e2 = 2.*w12/(w11+w22)
    # e 
    se[3] = np.sqrt(e1*e1+e2*e2)

    return se

    