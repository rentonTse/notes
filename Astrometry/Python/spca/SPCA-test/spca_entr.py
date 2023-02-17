import numpy as np
from creat_basisfs import creat_moffatlets
from spca import SPCA
from fit_gaus import star_gaus
from interp_bicubic import Interp_bicubic
from matplotlib import pyplot as plt


def gaus_estimate(star0,Ng1,Ng2):
    '''
    from gaus_estimate.c
    input Ng1 Ng2 star0

    return sigma mean

    '''
    Nh = Ng1/2-0.5
    nmax = Ng1*Ng2
    tmp = np.zeros((nmax,1))
    r2l = Nh*Nh
    ic = 0
    meant = 0
    for i in range(Ng1):
        det1 = i-Nh
        det1 = det1*det1
        for j in range(Ng2):
            det2 = j-Nh
            det2 = det2*det2
            r2 = det1+det2 
            if r2 >= r2l:
                tmp[ic] = star0[i][j]
                meant += tmp[ic]
                ic += 1

    if ic < 16:
        print('too few data to estimate the gausion noise in star\n')

    meant /= ic
    sum = 0
    for i in range(ic):
        r2 = tmp[i] - meant
        sum += r2*r2
    sum /= ic - 1
    sigma = np.sqrt(sum)
    mean = meant

    return sigma,mean     

def SPCA_entr(stars,spos,npc,Nstar0,Ng,gain):
    '''
    input 
    stars: 3D array, (()) 
    spos: 3D array, star position
    '''
    l0 = 9
    Ng0 = Ng + 5
    Mp = Ng*Ng
    Mp0 = Ng0*Ng0
    NPCmax = Nstar0
    rnbf0 = (l0+1)*(l0+1)
    means = np.zeros((Nstar0,1))
    a = np.zeros((Nstar0,Mp0))
    start = np.zeros((Ng0,Ng0))
    error0 = np.zeros((Nstar0,Mp0)) 
    star0 = np.zeros((Nstar0,Ng0,Ng0))
    star = np.zeros((Nstar0,Ng0,Ng0))
    compcoeff = np.zeros((npc,Nstar0))
    coefferr = np.zeros((Nstar0,npc))
    weight = np.zeros((Nstar0,Mp0))
    residual = np.zeros((Nstar0,Ng0,Ng0))
    SNR = np.zeros((Nstar0,1)) 

    if NPCmax < rnbf0:
        NPCmax = rnbf0

    
    for ic in range(Nstar0):
        for i in range(Ng0):
            for j in range(Ng0):
                star0[ic][i][j] = stars[ic][i][j]    

    nbond = (Ng0-Ng)/2
    nbond = int(nbond)
    for ic in range(Nstar0):
        # estimate the background noise as gasussian randoms
        sigma,mean = gaus_estimate(star0=star0[ic],Ng1=Ng0,Ng2=Ng0)
        means[ic] = mean
        
        k = 0
        for i in range(Ng0):
            for j in range(Ng0):
                # subtract the bacground sky intensity
                a[ic][k] = star0[ic][i][j] - mean
                k += 1
        k = 0
        sum = 0
        for i in range(Ng):
            for j in range(Ng):
                start[i][j] = star0[ic][i+nbond][j+nbond] - mean
                detx = sigma*sigma*gain*gain
                dety = gain*np.fabs(start[i][j])
                if dety < detx:
                    error0[ic][k] = sigma
                else:
                    error0[ic][k] = np.sqrt(detx+dety)/gain
                sum += start[i][j]
                k += 1      
    
        sigmasum = 0
        for k in range(Mp):
            sigmasum += error0[ic][k]*error0[ic][k]
        # calculate the SNR of each image   
        SNR[ic] = sum / np.sqrt(sigmasum)

    count = 0
    for ic in range(Nstar0):
        if SNR[ic] >= 0:
            for i in range(Ng0):
                for j in range(Ng0):
                    star[count][i][j] = a[ic][i*Ng0+j]
        spos[count][0] = spos[ic][0]
        spos[count][1] = spos[ic][1]
        means[count] = means[ic]
        count += 1

    Nstar = count
    Nobj = Nstar
    print('selected star number = %d\n'%Nstar)
    # write a fits image instar star


    nbond = (Ng0-Ng)/2+1
    for ic in range(Nstar):
        for i in range(Ng0):
            for j in range(Ng0):
                start[i][j] = star[ic][i][j]

        starf,para = star_gaus(y=start,nx=Ng0,ny=Ng0)

        xcen = np.zeros((2,1))
        xcen[0] = para[3]
        xcen[1] = para[4]
        
        # print("x = %f, y = %f\n"%(xcen[0],xcen[1]))
        starf = starf
        # starf = Interp_bicubic(nx=Ng0,ny=Ng0,a0=start,nbound=nbond,nxt=Ng,nyt=Ng,xcen=xcen)
        sigma,mean = gaus_estimate(star0=starf,Ng1=Ng,Ng2=Ng)

        k = 0
        sum = 0 
        for i in range(Ng):
            for j in range(Ng):
                a[ic][k] = starf[i][j]
                detx=sigma*sigma*gain*gain
                dety=gain*np.fabs(a[ic][k])
                if dety<detx:
                    error0[ic][k]=sigma
                else:
                    error0[ic][k]=np.sqrt(dety+detx)/gain
                sum += a[ic][k]
                k +=  1

        for k in range(Mp):
            error0[ic][k] /= sum
            a[ic][k] /= sum
            weight[ic][k]=1./(error0[ic][k]*error0[ic][k])
        for i in range(Ng):
            for j in range(Ng):
                star0[ic][i][j]=a[ic][i*Ng+j]
                stars[ic][i][j]=a[ic][i*Ng+j]

    # sigmasum = 0
    # for ic in range(Nstar):
    #     starf,residu,para = star_gaus(y=star0[ic],nx=Ng,ny=Ng)
    #     center[ic][0]=para[3]
    #     center[ic][1]=para[4]
    #     sigmasum += para[1]*2
    # sigmasum /= Nstar
    # estimate_para()



    mx = list(range(13))
    for k in range(13):
        mx[k]=0
    mx[0]=mx[1]=mx[2]=mx[3]=mx[4]=1
    
    nbf,ebasisf,ebasisfs = creat_moffatlets(rd=1.4,beta=3.5,l0=l0,msign=1,mx=mx,ng=Ng)

    # image of basis function 
    plt.figure()
    plt.imshow(ebasisfs[2])
    plt.savefig('spcafits/ebasisfs2.png')


    PCs,compbfr = SPCA(images=a,weight=weight,basisf=ebasisf,Mp=Mp,Nstar=Nstar,nbf=nbf,coeff=compcoeff,Nb=npc)
    # logger 

    # write fits image of PCs


    for ic in range(Nstar):
        for ipc in range(npc):
            sum = 0
            for i in range(Ng):
                for j in range(Ng):
                    sum += compbfr[ipc][i*Ng+j]*compbfr[ipc][i*Ng+j]*weight[ic][i*Ng+j]
            sum = np.sqrt()
            sum = 1/sum
            coefferr[ic][ipc] = sum
            sumt = np.fabs(compcoeff[ipc][ic]/sum)
            if sumt < 1.5:
                compcoeff[ipc][ic]=0

    #  write coeff and weight
    coeff = np.zeros((Nstar,npc,2))
    for ic in range(Nstar):
        for k in range(npc):
            coeff[ic][k][0] = compcoeff[k][ic]
            coeff[ic][k][1] = coefferr[ic][k]


    print('reconstruction\n')
    for ic in range(Nstar):
        sum = 0
        for i in range(Ng):
            for j in range(Ng):
                start[i][j] = 0
                for ipc in range(npc):
                    start[i][j] += compbfr[ipc][i*Ng+j]*compcoeff[ipc][ic]
                star[ic][i][j] = start[i][j]
                residual[ic][i][j] = star0[ic][i][j] - start[i][j]
                sum += residual[ic][i][j]*residual[ic][i][j]*weight[ic][i*Ng+j]
        sum /= Mp

    print('SPCA down\n')

    return PCs,coeff,Nobj
