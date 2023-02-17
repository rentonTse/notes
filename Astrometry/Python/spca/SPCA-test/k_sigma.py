import numpy as np

def sigma_clean_1D(data,r2,nsigma,dms,npp,indx):
    dm = dms[0]
    sigma = dms[1]
    np0 = npp[0]
    sigma2=nsigma*nsigma*sigma*sigma
    sum=0

    for k in range():
        for i in range(np0):
            if r2[i]<sigma2:
                r2[k] = r2[i]
                data[k] = data[i]
                sum += data[k]
                indx[k] = indx[i]

    npp[1] = k 
    if k == np0:
        return
    dm = sum/k
    sum = 0
    for i in range(k):
        det = data[i]-dm
        r2[i] = det*det
        sum += r2[i]
    sigma = np.sqrt(sum/(k-1))
    dms[0] = dm
    dms[1] = sigma 

    return dms

def k_sigma_1D(data,np0,dms,indx):
    r2 = np.zeros((np0,1))
    d1 = np.zeros((np0,1))
    npp = np.zeros((2,1))
    sum = 0 
    for i in range(np0):
        sum += data[i]    
    dm = sum/np0

    sum = 0 
    
    for i in range(np0):
        det = data[i]-dm
        r2[i] = det*det
        d1[i] = data[i]
        sum += r2[i]

    ktotal = np0
    sigma= np.sqrt(sum/(np0-1))
    dms[0]=dm
    dms[1]=sigma
    npp[0]=npp[1]=ktotal

    k = 0 
    while True:
        npp[0] = npp[1]
        dms = sigma_clean_1D(data=d1,r2=r2,nsigma=2.,dms=dms,npp=npp,indx=indx)
        k = k + 1
        if k >= 2: #npp[0] != npp[1]
            break
        indx[np0] = npp[1]

    return dms[1]