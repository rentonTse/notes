import numpy as np 
from npsf import npsf,psf_interp



def coeff_interp(order_PC,order_poly,Nstar,spos,Coeff,weight,PCs,ngp,Nobj,psfpos):
    '''
    from coeff_interp.c
    '''

    print('start npsf\n')
    dcoeff = npsf(order_poly=order_poly,order_PC=order_PC,Nstar=Nstar,pos=spos,Coeff=Coeff,weight=weight,PCs=PCs,ngp=ngp)

    print('npsf down start psf_interp\n')
    
    rPSFs = psf_interp(Nobj=Nobj,order_poly=order_poly,order_PC=order_PC,ngp=ngp,gpos=psfpos,dcoeff=dcoeff,PCs=PCs)
    print('psf_interp down\n')

    return rPSFs
