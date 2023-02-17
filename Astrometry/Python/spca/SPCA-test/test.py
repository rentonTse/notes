import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.io import fits
from spca_entr import SPCA_entr
from coeff_interp import coeff_interp
from fit_gaus import star_gaus
from size import size
# import logger

'''
complete test for SPCA 
'''
Ng=23
Ng0=Ng+5


# read the fits image

# imagehdu=fits.getheader(r'C:\Users\ASUS\Desktop\Archive\PCA\data\MCI\MCI_g_2021-06-09_1.fits')
# imagedata=fits.getdata(r'C:\Users\ASUS\Desktop\Archive\PCA\data\MCI\MCI_g_2021-06-09_1.fits')

imagehdu=fits.getheader('/home/lamost/xiezt/SPCA-test/fits/MCI_g_2021-06-09_1.fits')
imagedata=fits.getdata('/home/lamost/xiezt/SPCA-test/fits/MCI_g_2021-06-09_1.fits')
image=imagedata.astype(np.float64) 

print(image.shape)
# plt.imshow(imagedata, cmap='gray')
# plt.savefig('spcafits/image.png')

# read the value of GAIN from the header
# gain = imagehdu['GAIN']
gain = 1.5

# read the the star catalogue 

# df = pd.read_csv(r'C:\Users\ASUS\Desktop\Archive\PCA\data\MCI\catalogue\star\MCI_g_2021-06-09_1.fits.cat',
#                 skiprows=8,sep='\s+',header=None,
#                 names=['NUMBER','X_IMAGE','Y_IMAGE','FLUXERR_AUTO',
#                 'FLUX_AUTO','FWHM_IMAGE','FLUX_RADIUS','SNR'])
df = pd.read_csv('/home/lamost/xiezt/SPCA-test/fits/MCI_g_2021-06-09_1.fits.cat',
                skiprows=8,sep='\s+',header=None,
                names=['NUMBER','X_IMAGE','Y_IMAGE','FLUXERR_AUTO',
                'FLUX_AUTO','FWHM_IMAGE','FLUX_RADIUS','SNR'])

# read the size of the catalogue
Nstar0 = df.shape[0]
colum = df.shape[1]
print('Nstar0 = %d, Nhead = %d\n'%(Nstar0,colum))


# creat matrix to save data
mpara = np.array(df)
print(mpara.shape)
spos = np.zeros((Nstar0,2))

# star position
for ic in range(Nstar0):
    spos[ic][0] = mpara[ic][1]+1
    spos[ic][1] = mpara[ic][2]+1
    # print(mpara[ic][1],mpara[ic][2],ic)
# print(spos.shape,spos)

if Nstar0 < 40:
    print('The star number is less than 20, exit!\n')
    sys.exit()

# use 40 of the star to construct PC, 
# and the left used for interpolation test
Nstar = 40
Nstari = Nstar0 - Nstar

# creat the data matrix for star images
a0 = np.zeros((Nstar0,Ng0*Ng0))
# creat the matrix to save the center positions of each star image
center = np.zeros((Nstar0,2))


# read star images into stamps in matrix stars from image
nimgx = image.shape[0]
nimgy = image.shape[1]
Nh = Ng0/2

# creat the matrix for stars stamp
stars = np.zeros((Nstar0,Ng0,Ng0))
for ic in range(Nstar):
    rx = spos[ic][0]-1
    ry = spos[ic][1]-1
    # print(rx,ry)
    for i in range(Ng0):
        for j in range(Ng0):
            it = int(ry+i-Nh)
            jt = int(rx+j-Nh)
            # print(it,jt)
            if it >= 0 and it < nimgx and jt >= 0 and jt < nimgy:
                stars[ic][i][j] = image[it][jt]

if not os.path.isdir('spcafits'):
    os.mkdir('spcafits')
file_name = os.path.join('spcafits','stars.fits')
hdu = fits.PrimaryHDU(stars[0])
hdulist = fits.HDUList([hdu])
hdulist.writeto(file_name,overwrite=True)

# plt.imshow(stars[0], cmap='gray')
# plt.savefig('spcafits/star0.png')



# choose method and set parameters
npc = 8
# means
PCs,coeff,Nobj = SPCA_entr(stars=stars,spos=spos,
                            npc=npc,Nstar0=Nstar0,Ng=Ng,gain=gain)

file_name = os.path.join('spcafits','PCs.fits')
hdu = fits.PrimaryHDU(PCs[0])
hdulist = fits.HDUList([hdu])
hdulist.writeto(file_name,overwrite=True)

print('Nobj = %d\n'%Nobj )



# sys.exit(0)
'====================================='
order_poly = 3

Coeff = np.zeros((Nobj,npc))
weight = np.zeros((Nobj,npc))
psfpos = np.zeros((Nstar0,2))

print('start interp, Nstari = %d\n'%Nstari)
for ic in range(Nobj):
    for l in range(npc):
        Coeff[ic][l] = coeff[ic][l][0]
        weight[ic][l] = 1./coeff[ic][l][1]

# get the left star positions
for ic in range(Nstari):
    psfpos[ic][1] = spos[ic+Nstar][1]
    psfpos[ic][0] = spos[ic+Nstar][0]

# read the original star images 
# at the interpolated posiotions
print("start read\n")
Nh = Ng/2
lstars = np.zeros((Nstari,Ng,Ng))
for ic in range(Nstari):
    rx = psfpos[ic][0]-1.5
    ry = psfpos[ic][1]-1.5
    sum = 0
    for i in range(Ng):
        for j in range(Ng):
            it = int(rx+i-Nh)
            jt = int(ry+j-Nh)
            if it >= 0 and it < nimgx and jt >= 0 and jt < nimgy:
                lstars[ic][i][j] = image[it][jt]
file_name = os.path.join('spcafits','lstars.fits')
hdu = fits.PrimaryHDU(lstars[0])
hdulist = fits.HDUList([hdu])
hdulist.writeto(file_name,overwrite=True)

# plt.imshow(lstars[0], cmap='gray')
# plt.savefig('spcafits/lstar0.png')

oparas = np.zeros((Nstari,5))
for ic in range(Nstari):
    oparas[ic] = star_gaus(y=stars[ic],nx=Ng,ny=Ng)
    center[ic][0] = oparas[ic][3]
    center[ic][1] = oparas[ic][4]

se = np.zeros((Nstari,4))
for ic in range(Nstari):
    se[ic] = size(I=stars[ic],Ng=Ng,cen=center[ic],rg=1) 

np.set_printoptions(threshold=np.inf)
sys.stdout = open('star_size.csv',mode = 'w',encoding='utf-8')
for ic in range(Nstari):
    print(se[ic][0],se[ic][1],se[ic][2],
            psfpos[ic][0],psfpos[ic][1])
sys.stdout.close()


print('read down\n')

print('start coeff_interp\n')


rPSFs = coeff_interp(order_PC=npc,order_poly=order_poly,Nstar=Nstar,spos=spos,
                    Coeff=Coeff,weight=weight,PCs=PCs,ngp=Ng,Nobj=Nstari,psfpos=psfpos)

rpparas = np.zeros((Nstari,5))
for ic in range(Nstari):
    rpparas[ic] = star_gaus(y=stars[ic],nx=Ng,ny=Ng)
    center[ic][0] = rpparas[ic][3]
    center[ic][1] = rpparas[ic][4]

psfse = np.zeros((Nstari,4))
for ic in range(Nstari):
    psfse[ic] = size(I=stars[ic],Ng=Ng,cen=center[ic],rg=1) 

np.set_printoptions(threshold=np.inf)
sys.stdout = open('star_size.csv',mode = 'w',encoding='utf-8')
for ic in range(Nstari):
    print(psfse[ic][0],psfse[ic][1],psfse[ic][2],
            psfpos[ic][0],psfpos[ic][1])
sys.stdout.close()

print("calculated\n")



