import numpy as np
from spca import lubksb,ludcmp

def creat_matrix_coeff(order_PC,l0,Nstar,poly,Coeff,weight,PCs,ngp):
    '''
    '''
    m=order_PC*l0+l0
    a = np.zeros((m,m))
    b = np.zeros((m,1))
    for n in range(order_PC):
        for q in range(l0):
            i = n*l0+q
            for f in range(l0):
                j = n*l0+f
                for ic in range(Nstar):
                    a[i][j] += 2*poly[q][ic]*poly[f][ic]*weight[ic][n]  
    for n in range(order_PC):
        for q in range(l0):
            i=n*l0+q
            j=order_PC*l0+q
            for l in range(ngp):
                for f in range(ngp):
                    a[i][j] += PCs[n][l][f] 
    for n in range(order_PC):
        for q in range(l0):
            i=n*l0+q
            for ic in range(Nstar):
                b[i] += 2*poly[q][ic]*Coeff[ic][n]*weight[ic][n]
    for q in range(l0):
        i=order_PC*l0+q
        for l in range(order_PC):
            j=l*l0+q
            for r in range(ngp):
                for s in range(ngp):
                    a[i][j] += PCs[l][r][s]
    for q in range(l0):
        k = order_PC*l0+q
        if q == 0:
            b[k] = 1
        else:
            b[k] = 0
    return a,b                             
                    



def npsf(order_poly,order_PC,Nstar,pos,Coeff,weight,PCs,ngp):
    '''
    '''
    l0 = 0
    for i in range(order_poly+1,0,-1):
        l0 += i
        print('l0 = %d\n',l0)
    m = order_PC*l0+l0
    
    indx = np.zeros((m,1))
    poly = np.zeros((l0,Nstar))
    dcoeff = np.zeros((order_PC,l0))

    # creat polynomials
    for ic in range(Nstar):
        x = pos[ic][0]
        y = pos[ic][1]
        for k in range(l0):
            for l in range(order_poly+1):
                for f in range(order_poly+1):
                    ix = l
                    iy = f
                    if l+f <= order_poly:
                        poly[k][ic] = np.power(x,ix)*np.power(y,iy)
    print('\n')
    print('k = %d\n',k)

    a,b = creat_matrix_coeff(order_PC=order_PC,l0=l0,Nstar=Nstar,poly=poly,Coeff=Coeff,weight=weight,PCs=PCs,ngp=ngp)

    print('creat_matrix_coeff\n')
    # for i in range(m):
    #     for j in range(m):
    #         wa[i][j] = a[i][j]


    a = ludcmp(a=a,n=m,indx=indx)
    b = lubksb(a=a,n=m,indx=indx,b=b)  

    for l in range(order_PC):
        for f in range(l0):
            k = l*l0+f
            dcoeff[l][f] = b[k]

    return dcoeff

def psf_interp(Nobj,order_poly,order_PC,ngp,gpos,dcoeff,PCs):
    '''
    from npsf.c L523
    '''

    l0 = 0
    for i in range(order_poly):
        l0 += i
    print("l=%d in psf_interp\n",l0)

    rPSFs = np.zeros((Nobj,ngp,ngp))
    xy = np.zeros((Nobj,l0))
    Coeff = np.zeros((Nobj,order_PC))


    for l in range(order_PC):
        for f in range(l0):
            k=l*l0+f
            print("%f\t",dcoeff[l][f])
        print("\n")

    for ic in range(Nobj):
        x = gpos[ic][0]
        y = gpos[ic][1]
        for k in range():
            for l in range(order_poly):
                for f in range(order_poly):
                    ix = l
                    iy = f
                    if l+f<=order_poly:
                        xy[ic][k] = np.power(x,ix)*np.power(y,iy)
    print("k=%d\n",k)                    

    for ic in range(Nobj):
        for l in range(order_PC):
            Coeff[ic][l] = 0
            for f in range(l0):
                Coeff[ic][l]+=dcoeff[l][f]*xy[ic][f]
    print("Coeff\n")

    for ic in range(Nobj):
        sum = 0
        for i in range(ngp):
            for j in range(ngp):
                rPSFs[ic][i][j] = 0
                for l in range(order_PC):
                    # construct PSFs
                    rPSFs[ic][i][j] += Coeff[ic][l]*PCs[l][i][j]               
        for i in range(ngp):
            for j in range(ngp):
                rPSFs[ic][i][j] /= sum
    print("rPSFs\n")            

    return rPSFs


