TINY = 1.0e-20 
'''

SPCA from SPCA.c
{EC,SPrepare_Dmatrix,Screat_Dmatrix,lubksb,ludcmp,Rebasis}
'''

import numpy as np


# SPCA
def EC(xj,phi,weight,Mp,nbf,coeff):
    '''
    from iSPCA.c 237
    E-step,solve coeffcients C through the linear operation
    I = PC + noise, weight = V^-1

    // input 
    // xj: 1D array, (jth star,Mp) image data I
    // phi: 2D array, PCs array
    // weight:2D array, from SPCA_entr.c L129 
    // Mp: int, total number of pixels 
    // nbf: int, number of basis functions

    // output
    // coeff:1D array, coeff[nbf]
    '''
    
    PtV = np.zeros((Mp,nbf)) #PtV = P^T * V^-1
    tmp = np.zeros((1,nbf))  #tmp = P^T * V^-1 * I   
    p3 = np.zeros((nbf,nbf)) #p3 = P^T * V^-1 * P
    
    for i in range(Mp):
        for j in range(nbf):
            PtV[i][j] = phi[j][i]*weight[i]

    for j in range(nbf):
        tmp[j] = 0
        for i in range(Mp):
            tmp[j] += PtV[i][j]*xj[i]

    for i in range(nbf):
        for j in range(nbf):
            p3[i][j] = 0 
            for k in range(Mp): 
                p3[i][j] = PtV[k][i]*phi[j][k]
    
    p3in = np.linalg.inv(p3)           
    
    for j in range(nbf):
        coeff[j]=0
        for i in range(nbf):
            coeff[j] += p3in[i][j]*tmp[i]
            # print(coeff)
    return coeff

#  SPrepare_Dmatrix(weight,rbasisf,Mp,Nstar,rnbf,sigmaka)
def SPrepare_Dmatrix(weight,basisf,Mp,Nstar,nbf):
    '''
    // input 
    // weight:2D array, (Nstar0,Mp0) from SPCA_entr.c L129 
    // basisf:2D array, basis function (nbf0,Mp)
    // Mp:
    // Nstar:int, number of stars 
    // nbf:

    // output
    // sigmaka:3D array,(Nstar,nbf,nbf)
    '''
    sigmaka = np.zeros((Nstar,nbf,nbf))
    for j in range(nbf):
        for l in range(nbf):
            for i in range(Nstar):
                sigmaka[i][j][l] = 0
                for k in range(Mp):
                    sigmaka[i][j][l] += weight[i][k]*basisf[j][k]*basisf[l][k]
    return sigmaka        

# Screat_Dmatrix(sigmaka,Mp,Nstar,rnbf,coeff[ipc],a,b,weight,star,rbasisf);
def Screat_Dmatrix(sigmaka,Mp,Nstar,nbf,coeff,weight,image,basisf):
    '''
    // input
    // weight:
    // images:2D array, star image
    // coeff
    // basisf
    // Mp,nimg,nbf
    // sigmaka:
    
    // output
    // a
    // b
    '''
    tmp = np.zeros((1,Mp))
    a = np.zeros((nbf,nbf))
    b = np.zeros((nbf,1))
    for k in range(Mp):
        tmp[k] = 0
        for i in range(Nstar):
            tmp[k] += weight[i][k]*image[i][k]*coeff[i]

    for j in range(nbf):
        for l in range(nbf):
            a[j][l]=0
            for i in range(Nstar):
                a[j][l] += sigmaka[i][j][l]*coeff[i]*coeff[i]
            a[l][j] = a[j][l]
        
        b[j] = 0
        for k in range(Mp):
            b[j] += basisf[j][k]*tmp[k]
    return a,b            

# Use LU decomposition to solve equations
def ludcmp(a,n,indx):
    '''
    from ludcmp.c
    '''
    d = 1.0
    vv = np.zeros((1,n))
    for i in range(n):
        big = 0.0
        for j in range(n):
            temp = np.fabs(a[i][j])
            if temp > big:
                big = temp
        if big == 0.0:
            print('Singular matrix in ludcmp')
        vv[i] = 1.0/big 

    for j in range(n):
        for i in range(j):
            sum = a[i][j]
            for k in range(i):
                sum -= a[i][k]*a[k][j]
            a[i][j] = sum                     
        big = 0.0

        for i in range(j,n):
            sum = a[i][j]
            for k in range(j):
                sum -= a[i][k]*a[k][j]
                a[i][j] = sum
                dum = vv[i]*np.fabs(sum)
                if dum >= big:
                    big = dum
                    imax = i    

        if j != imax:
            for k in range(n):
                dum = a[imax][k]
                a[imax][k] = a[j][k]
                a[j][k] = dum
            d = -d
            vv[imax] = vv[j]

        indx[j] = imax
        if a[i][j] == 0.0:
            a[i][j] = TINY   
        if j != (n-1):
            dum = 1.0/(a[i][j])
            for i in range(j+1,n):
                a[i][j] *= dum
    return a

def lubksb(a,n,indx,b):
    '''
    from lubksb.c
    '''
    ii = -1
    for i in range(n):
        ip = indx[i]
        sum = b[ip]
        b[ip] = b[i]
        if ii >= 0:
            for j in range(ii,i-1):
                sum -= a[i][j]*b[j]
        elif sum:
            ii = i
            b[i] = sum

    for i in range(n-1,-1,-1):
        sum = b[i]
        for j in range(i+1,n):
            sum -= a[i][j]*b[j]
            b[i] = sum/a[i][i] 
    return b        




# Rebasis 
def Rebasis(basisf0,nbf0,Mp):
    '''
    from rebasis.c

    // input 
    // basisf0: 2D array, created by creat_basisf  
    // Mp: total number of pixels
    // nbf0:int, created by creat_basisf

    // output 
    // basisf: 2D array, re basis function
    // nbf: int, remain nbf
    '''

    phi = np.zeros((Mp,1))
    phit = np.zeros((Mp,1))
    xij = np.zeros((nbf0,Mp))
    basisft = np.zeros((nbf0,Mp))
    basisf = np.zeros((nbf0,Mp))
    cj = np.zeros((nbf0,1))
    weight = np.zeros((nbf0,1))
    nmax = 5000
    coeff = np.zeros((nbf0,nbf0)) 

    for i in range(Mp):
        for j in range(nbf0):
            xij[j][i] = basisf0[j][i]
            basisft[j][i] = 0
    for j in range(nbf0):
        weight[j] = 1
        weight[0] = nbf0/2
    
    k = 0
    while True:
        for i in range(Mp):
            phi[i] = phit[i] = basisf0[k][i]

        niter = 0 
        while True:
            # E-step
            for i in range(nbf0):
                cj[i] = 0
                for j in range(Mp):
                    cj[i] += xij[i][j]*phi[j]
            # M-step
            absphi = 0
            for j in range(Mp):
                phi[j] = 0
                for i in range(nbf0):
                    phi[j] += weight[i]*cj[i]*xij[i][j]
                absphi += phi[j]*phi[j]
            # Renormalise
            absphi = np.sqrt(absphi)
            detphi = 0
            for j in range(Mp):
                phi[j] /= absphi
                detphi += (phi[j] - phit[j])**2 
                phit[j] = phi[j]
            detphi = np.sqrt(detphi)    
            niter += 1
            if niter >= nmax or detphi <= 1.e-14:
                break
        det = 0
        for i in range(nbf0):
            coeff[i][k] = cj[i]
            sum = 0
            for j in range(Mp):
                xij[i][j] -= cj[i]*phi[j]
                basisft[i][j] += cj[i]*phi[j]
                sum += basisft[i][j]*basisf0[i][j]
            sum = np.abs(1-sum)
            if det < sum:
                det = sum 
        for j in range(Mp):
            basisf[k][j] = phi[j]
        weight[k] = 1
        weight[k+1] = nbf0/2
        k += 1
        if k >= nbf0 or det <= 0.0000001:
            break
    nbf = k    

    return basisf,nbf



#  SPCA(a,weight,ebasisf,Mp,Nstar,nbf,compbfr,compcoeff,npc)
def SPCA(images,weight,basisf,Mp,Nstar,nbf,coeff,Nb):

    Ng = np.sqrt(Mp)
    xij = np.zeros((Nstar,Mp))
    sigmaka = np.zeros((Nstar,nbf,nbf))
    coff = np.zeros((Nstar,Nb))
    star = np.zeros((Nstar,Mp))
    basis = np.zeros((nbf,Mp))
    compbf = np.zeros((Nb,Mp))    
    
    for i in range(nbf):
        for k in range(Mp):
            basis[i][k] = basisf[i][k]
    nbfs = nbf 
    # Rebasis(basis,nbfs,Mp,rbasisf,rcoeff)
    rbasisf,rnbf = Rebasis(basisf0=basis,nbf0=nbfs,Mp=Mp)
    print('nbfs = %d  rnbf = %d\n',nbfs,rnbf)
    # SPrepare_Dmatrix(weight,rbasisf,Mp,Nstar,rnbf,sigmaka)
    sigmaka = SPrepare_Dmatrix(weight=weight,basisf=rbasisf,nbf=rnbf,Mp=Mp,Nstar=Nstar)

    niter = 500
    while True:
        for i in range(niter):
            # E-step
            for i in range(Nstar):
                for j in range(Mp):
                    star[i][j] = xij[i][j] = images[i][j] 

            for i in range(Nstar):
                # EC(xij[i],compbf[i],weight[i],Mp,Nb,coff[i])
                coff[i] = EC(xj=xij[i],phi=compbf[i],weight=weight[i],Mp=Mp,nbf=Nb)
            for i in range(Nstar):
                for j in range(Nb):
                    coeff[j][i] = coff[i][j]

            for ipc in range(Nb):
                # Screat_Dmatrix(sigmaka,Mp,Nstar,rnbf,coeff[ipc],a,b,weight,star,rbasisf)
                a,b = Screat_Dmatrix(sigmaka=sigmaka,Mp=Mp,Nstar=Nstar,nbf=rnbf,coeff=coeff[ipc],weight=weight,image=star,basisf=rbasisf)
                indx = []
                a = ludcmp(a=a,n=rnbf,indx=indx)
                b = lubksb(a=a,n=rnbf,indx=indx,b=b)
                sum = 0
                for k in range(Mp):
                    compbf[ipc][k] = 0
                    for i in range(rnbf):
                        compbf[ipc][k] += rbasisf[i][k]*b[i]
                for i in range(Nstar):
                    for k in range(Mp):
                        star[i][k] -= compbf[ipc][k]*coeff[ipc][i]
            
            # re-orthognal 
            for k0 in range(Nb-1):
                absphi = 0
                for j in range(Mp):
                    absphi += compbf[k0][j]*compbf[k0][j]
                    absphi = np.sqrt(absphi)
                for j in range(Mp):
                    compbf[k0][j] /= absphi
                for k in range(k0+1,Nb):
                    sum = 0
                    for j in range(Mp):
                        sum += compbf[k0][j]*compbf[k][j]
                    for j in range(Mp):
                        compbf[k][j] -= compbf[k0][j]*sum  
            
            # renormalization the last one 
            for i in range(Nb):
                absphi = 0 
                for j in range(Mp):
                    absphi += compbf[k][j]**2
                    absphi = np.sqrt(absphi)    
                for j in range(Nstar):
                    compbf[k][j] /= absphi

            # calculate chi^2
            sum1 = 0
            for j in range(Mp):
                for i in range(Nstar):
                    xij[i][j] = 0  
                    for k in range(Nb):
                        xij[i][j] += compbf[k][j]*coeff[k][i]
                        sum1 += (xij[i][j] - images[i][j])*(xij[i][j] - images[i][j])*weight[i][j]
            
            detchi2 = sum2/sum1 - 1
            detchi2 = np.fabs(detchi2)
            sum2 = sum1
            print('niter=%d  detchi2=%e   chi2=%e\n',i,detchi2,sum1)             
        if i == niter or detchi2 <= 1.e-12:
            break
    
    PCs = np.zeros((Nb,Ng,Ng))
    for k in range(Nb):
        for i in range(Ng):
            for j in range(Ng):
                PCs[k][i][j] = compbf[k][i*Ng+1]
    
    return PCs,compbf





























