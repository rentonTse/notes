import numpy as np
from moffatlets import Plr_moffat,Plr_gauss,Plr0_moffat  
from matplotlib import pyplot as plt 

def basisfs_initial(l0,mx,msign):
    '''
    // input
    // l0: the max order of basis function
    // mx:
    // msign: use angular

    // output
    //nbf: number of basis function
    // lm: 2D matrix (nbf,2) save l,m order of 
    // basis function

    '''
    if l0 > 40:
        print('the max order of basis function is too high, in basisfs_initial\n')    
    
    mt = [0 for i in range(100)]

    for i in range(13):
        mt[i] = 0 
    if mx[0]:
        mt[0] = 1
    if mx[1]:
        mt[1] = 2 
    if mx[2]:
        mt[2] = 3
    if mx[3]:
        mt[3] = 5
    if mx[4]:
        mt[4] = 7
    if mx[5]:
        mt[5] = 11
    if mx[6]:
        mt[6] = 13
    if mx[7]:
        mt[7] = 17
    if mx[8]:
        mt[8] = 19
    if mx[9]:
        mt[9] = 23
    if mx[10]:
        mt[10] = 29
    if mx[11]:
        mt[11] = 31
    if mx[12]:
        mt[12] = 37
    
    ibig = np.zeros((l0+1,2*l0+1))
    for i in range(l0+1):
        for j in range(0,2*l0+1):
            ibig[i][j] = 0

    for i in range(l0+1):
        ibig[i][l0] = 1

    if msign == 1:
        if mt[0]:
            for i in range(l0):
                ibig[i][l0-1] = ibig[i][l0+1] = 1
            # print(ibig)
        for k in range(1,13):
            if mt[k] < l0 and mt[k] > 0:
                for i in range(l0+1):
                    for j in range(l0,-l0-1,-1):
                        kt = i + np.abs(j)
                        if kt <= l0:
                            jd = j
                            md = int(jd/mt[k])
                            m = md
                            if m == md:
                                ibig[i][j+l0] = 1
    # print(ibig)

    lar = []
    mar = []
    for i in range(l0+1):
        for j in range(i,-i-1,-1):
            l = i - np.abs(j)
            if ibig[l][j+l0]:
                lar.append(l)
                mar.append(j)                    
                # print(l,j)
    larr = np.array(lar) 
    marr = np.array(mar)

    lm = np.zeros((larr.shape[0],2))
    lm[:,0] = larr
    lm[:,1] = marr
    # print(larr.shape[0])
    nbf =  lm.shape[0]

    return lm,nbf


def creat_moffatlets(rd,beta,l0,msign,mx,ng):
    '''
    subroutine 
    basisfs_initial Plr_moffat,Plr_gauss
    

    return basisf0 2D array (nbf,Mp)
    '''
    Mp = ng*ng
    lm,nbf = basisfs_initial(l0=l0,msign=msign,mx=mx)
    basisf0 = np.zeros((nbf,Mp))
    print('nbf =',nbf)

    for k in range(nbf):
        l = int(lm[k][0])
        m = int(lm[k][1])
        # print(l,m,k)
        
        kt = 0
        sum = 0
        for i in range(ng):
            x = i-ng/2+0.5
            for j in range(ng):
                y = j-ng/2+0.5
                tmpx = 0
                for ra in range(-5,6):
                    xt = x+ra/11.
                    for rb in range(-5,6):
                        yt = y+rb/11.
                        r = np.sqrt(xt*xt+yt*yt)
                        # print(xt,yt,r)
                        tmpx += Plr_moffat(rd=rd,beta=beta,r=r,l=l)
                r = np.sqrt(x*x+y*y)
                # print(r,tmpx)

                if m == 0:
                    basisf0[k][kt] = tmpx
                    sum += basisf0[k][kt]*basisf0[k][kt]
                    kt = kt + 1 
                else:
                    if r == 0:
                        basisf0[k][kt] = 0
                        kt = kt + 1
                    else:
                        theta = x/r
                        if theta > 1.:
                            theta = 1.
                        if theta < -1.:
                            theta = -1
                            theta = np.arccos(theta)
                        if y < 0:
                            theta = 2*np.pi - theta 
                        if m > 0:
                            basisf0[k][kt] = np.cos(m*theta)*tmpx
                            sum += basisf0[k][kt]*basisf0[k][kt]
                            kt = kt + 1
                        if m < 0:
                            basisf0[k][kt] = np.sin(-m*theta)*tmpx 
                            sum += basisf0[k][kt]*basisf0[k][kt]
                            kt = kt + 1
        # print(tmpx)
        sum = np.sqrt(sum)
        for i in range(kt):
            basisf0[k][i] /= sum 

    ebasisfs = np.zeros((nbf,ng,ng))
    for k in range(nbf):
        for i in range(ng):
            for j in range(ng):
                ebasisfs[k][i][j] = basisf0[k][i*ng+j]


    return nbf,basisf0,ebasisfs


def creat_gaussianlets(sigma,l0,msign,mx,Mp,ng):
    '''
    return basisf0 2D array (nbf,Mp)
    '''
    lm,nbf = basisfs_initial(l0=l0,msign=msign,mx=mx)
    basisf0 = np.zeros((nbf,Mp))
    print('nbf =',nbf)

    for k in range(nbf):
        l = lm[k][0]
        m = lm[k][1]
        # print(l,m)
        
        kt = 0 
        sum = 0
        for i in range(ng):
            x = i-ng/2+0.5
            for j in range(ng):
                y = j-ng/2+0.5
                tmpx = 0
                for ra in range(-5,6):
                    xt = x+ra/11.
                    for rb in range(-5,6):
                        yt = y+rb/11.
                        r = np.sqrt(xt*xt+yt*yt)
                        tmpx += Plr_gauss(sigma=sigma,r=r,l=l)
                r = np.sqrt(x*x+y*y)
                if m == 0:
                    basisf0[k][kt] = tmpx
                    sum += basisf0[k][kt]*basisf0[k][kt]
                    kt = kt + 1
                else:
                    if r == 0:
                        basisf0[k][kt] = 0
                        kt = kt + 1
                    else:
                        theta = x/r
                        if theta > 1.:
                            theta = 1.
                        if theta < -1.:
                            theta = -1
                            theta = np.arccos(theta)
                        if y < 0:
                            theta = 2*np.pi - theta 
                        if m > 0:
                            basisf0[k][kt] = np.cos(m*theta)*tmpx
                            sum += basisf0[k][kt]*basisf0[k][kt]
                            kt = kt + 1
                        if m < 0:
                            basisf0[k][kt] = np.sin(-m*theta)*tmpx 
                            sum += basisf0[k][kt]*basisf0[k][kt]
                            kt = kt + 1

        sum = np.sqrt(sum)
        for i in range(kt):
            basisf0[k][i] /= sum 

    ebasisfs = np.zeros((nbf,ng,ng))
    for k in range(nbf):
        for i in range(ng):
            for j in range(ng):
                ebasisfs[k][i][j] = basisf0[k][i*ng+j]


    return ebasisfs



