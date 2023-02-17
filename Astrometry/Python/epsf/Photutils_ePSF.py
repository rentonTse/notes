import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder
from astropy.nddata import NDData
from photutils.psf import extract_stars
from photutils.detection import find_peaks
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.psf import EPSFBuilder
from photutils.psf import EPSFFitter
from astropy.stats import SigmaClip
from photutils.centroids import centroid_com,centroid_epsf
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable




'---------------------------------------------------------------------'
# Simulated data fits image and catalogue
path = '/home/xzt/Desktop/CSST.fits'
imagehdu=fits.getheader(path)
imagedata=fits.getdata(path)
imagedata=imagedata.astype(np.float64)
# print(imagedata.shape)

fn = '/home/xzt/Desktop/MSCtest.cat'
df = pd.read_table(fn,skiprows=0,header=0,sep='\s+',skipinitialspace=True)

'---------------------------------------------------------------------'
# cut the star image,the interval depends on catalogue  
imagestart = 850
imageend   = 950

# the number of column that used in the measurement 
ncl = 4

# the corresponding interval in catalogue
catstart = (ncl-1)*96 + 1
catend   = catstart + 92

data = imagedata[0:9000,imagestart:imageend]
print('the shape of cutting data is:',data.shape)

# plt.imshow(data,cmap='gray')
# plt.colorbar()
# plt.title('simulate image')
# plt.savefig('paper/sim_img.png')

'---------------------------------------------------------------------'
# find the star that satisfy our threshold
peaks_tbl = find_peaks(data, threshold=2000.)  
peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output  

size = 25
hsize = (size - 1) / 2 
x = peaks_tbl['x_peak']  
y = peaks_tbl['y_peak']  
mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
        (y > hsize) & (y < (data.shape[0] -1 - hsize)))  

# stars tabel is built from the peaks table by deleting the 'mask' position
stars_tbl = Table()
stars_tbl['x'] = x[mask]  
stars_tbl['y'] = y[mask]
# print(stars_tbl)   

# do the 3-sigma clip
mean_val, median_val, std_val = sigma_clipped_stats(data,sigma=3.)  
data -= median_val  

# invert to nddata type because of the function extract_stars
nddata = NDData(data=data)

# put all the good(to fit) stars to a list,each star has a size of 25, lenth of the list is the number of stars
stars = extract_stars(nddata, stars_tbl, size=(25,25))
print('The number of extract star',len(stars)) 

'---------------------------------------------------------------------'
# the extracted star's image
nrows = 2
ncols = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15),
                       squeeze=True)
ax = ax.ravel()
for i in range(nrows*ncols):
    norm = simple_norm(stars[i], 'log', percent=99.)
    ax[i].imshow(stars[i],norm =norm ,origin='lower', cmap='viridis')
    ax_divider = make_axes_locatable(ax[i])
plt.savefig('ex_stars.png')
plt.clf()   

'---------------------------------------------------------------------'
# build an ePSF and fit stars
# recentering_func can choose 1.controid_com, 2.centroid_epsf(delected in latest Photutlis version)
epsf_builder = EPSFBuilder(oversampling=4, shape=None,
                        smoothing_kernel='quartic', recentering_func=centroid_com,
                        recentering_maxiters=20, fitter=EPSFFitter(), maxiters=10,
                        progress_bar=False, norm_radius=5.5, shift_val=0.5,
                        recentering_boxsize=(5, 5), center_accuracy=1.0e-3,
                        flux_residual_sigclip=SigmaClip(sigma=3,cenfunc='median',
                                                        maxiters=10)) 

epsf, fitted_stars = epsf_builder(stars)   

'---------------------------------------------------------------------'
# the image of drived epsf 

# plt.figure()
# norm = simple_norm(epsf.data, 'log', percent=99.)
# plt.imshow(epsf.data,norm = norm ,origin='lower', cmap='viridis')
# plt.colorbar()
# plt.savefig('figure/epsf.png')

# profile of the central section
plt.plot(epsf.data[50, :])
plt.savefig('epsf1d.png')

'---------------------------------------------------------------------'
# compute residuals and save it
mag_set =[]
for i in range(len(stars)):
    magi = -2.5*np.log10(fitted_stars._data[i].flux)
    mag_set.append(magi)
mag_arr = np.array(mag_set)


xpos_set =[]
for i in range(len(stars)):
    x,y = fitted_stars._data[i].center
    x = x + imagestart
    xpos_set.append(x)
xpos_arr = np.array(xpos_set)

ypos_set =[]
for i in range(len(stars)):
    x,y = fitted_stars._data[i].center
    ypos_set.append(y)
ypos_arr = np.array(ypos_set)


mag_cat  = df['mag'][catstart:catend]
y_cat = df['yImage'][catstart:catend]
x_cat = df['xImage'][catstart:catend]

# -1 because the difference between python and ds9
res_x = x_cat - xpos_arr - 1 
xmean = np.average(res_x,axis=0)

res_y = y_cat - ypos_arr - 1
ymean = np.average(res_y,axis=0)

res_mag = mag_cat - mag_arr 
mmean = np.average(res_mag,axis=0)
res_mag = res_mag -mmean

print('mean of x res =',xmean)
print('mean of y res =',ymean)
print('mean of mag res =',mmean)

'---------------------------------------------------------------------'
# phase function of x and y
pyset =[]
for i in y_cat:
    i = i - math.floor(i+0.5)
    pyset.append(i)
phi_y = np.array(pyset)

pxset =[]
for j in x_cat:
    j = j - math.floor(j+0.5)
    pxset.append(j) 
phi_x = np.array(pxset)

'---------------------------------------------------------------------'
# center residuals
kk =np.ones(len(stars))
plt.scatter(x_cat,res_x ,marker='+')
plt.xlabel("X position", fontdict={'size': 15})
plt.ylabel("X res", fontdict={'size': 15})
plt.plot(x_cat,kk*xmean,label = '%s'%xmean)
plt.title('Residuals of X' )
plt.legend()
plt.savefig('%s_center_resx.png'%ncl)
plt.clf()

plt.scatter(y_cat,res_y ,marker='+')
plt.xlabel("Y position", fontdict={'size': 15})
plt.ylabel("Y res", fontdict={'size': 15})
plt.plot(y_cat,kk*ymean,label = '%s'%ymean)
plt.title('Residuals of Y' )
plt.legend()
plt.savefig('%s_center_resy.png'%ncl)
plt.clf()

# hist image
plt.hist(res_x,bins = 10,edgecolor='r',histtype='bar',alpha=0.5)
plt.xlabel('residual of x')
plt.savefig('%s_res_xhist.png'%ncl)
plt.clf()

plt.hist(res_y,bins = 10,edgecolor='r',histtype='bar',alpha=0.5)
plt.xlabel('residual of y')
plt.savefig('%s_res_yhist.png'%ncl)
plt.clf()

'---------------------------------------------------------------------'
# PPE of x
cx = np.zeros((len(stars),2))
cx[:,0] = phi_x
cx[:,1] = res_x
cx = sorted(cx,key=lambda x : x[0])
cx = np.array(cx)
cx = np.array_split(cx,10,axis=0)

xlist=[]
for j,c_j in enumerate(cx):
        mean_j = np.mean(c_j,axis = 0)
        xlist.append(mean_j)
mean_cx = np.array(xlist)        
# Polyfit
zx = np.polyfit(mean_cx[:,0],mean_cx[:,1],4)
px = np.poly1d(zx)
xvals = px(mean_cx[:,0])

plt.scatter(phi_x,res_x ,marker='+')
plt.plot(mean_cx[:,0],mean_cx[:,1],'o')
plt.plot(mean_cx[:,0],xvals,'r',label='polyfit curve')
plt.plot(phi_x,kk*xmean,label = '%s'%xmean)
plt.ylim(-0.1,0.1)
plt.xlabel("x phase", fontdict={'size': 15})
plt.ylabel("x ppe", fontdict={'size': 12})
plt.title('pixel phase error of x',fontdict={'size': 15} )
plt.legend()
plt.savefig('%s_ppe_x.png'%ncl)
plt.clf()

'---------------------------------------------------------------------'
# PPE of y
cy = np.zeros((len(stars),2))
cy[:,0] = phi_y
cy[:,1] = res_y
cy = sorted(cy,key=lambda x : x[0])
cy = np.array(cy)
cy = np.array_split(cy,10,axis=0)


ylist=[]
for i,c_i in enumerate(cy):
        mean_i = np.mean(c_i,axis = 0)
        ylist.append(mean_i)
mean_cy = np.array(ylist)        

# Polyfit
zy = np.polyfit(mean_cy[:,0],mean_cy[:,1],4)
py = np.poly1d(zy)
yvals = py(mean_cy[:,0])

plt.scatter(phi_y,res_y ,marker='+')
plt.plot(mean_cy[:,0],mean_cy[:,1],'o')
plt.plot(mean_cy[:,0],yvals,'r',label='polyfit curve')
plt.plot(phi_y,kk*ymean,label = '%s'%ymean)
plt.ylim(-0.1,0.1)
plt.xlabel("y phase", fontdict={'size': 15})
plt.ylabel("y ppe", fontdict={'size': 12})
plt.title('pixel phase error of y',fontdict={'size': 15} )
plt.legend()
plt.savefig('%s_ppe_y.png'%ncl)
plt.clf()

'---------------------------------------------------------------------'
# PPE of mag
cm = np.zeros((92,2))
cm[:,0] = phi_y
cm[:,1] = res_mag
cm = sorted(cm,key=lambda x : x[0])
cm = np.array(cm)
cm = np.array_split(cm,10,axis=0)

mlist=[]
for i,c_i in enumerate(cm):
        mean_i = np.mean(c_i,axis = 0)
        mlist.append(mean_i)
mean_cm = np.array(mlist)       
 
# Polyfit
zm = np.polyfit(mean_cm[:,0],mean_cm[:,1],4)
pm = np.poly1d(zm)
mvals = pm(mean_cm[:,0])


plt.scatter(phi_y,res_mag ,marker='+')
plt.plot(mean_cm[:,0],mean_cm[:,1],'o')
plt.plot(mean_cm[:,0],mvals,'r',label='polyfit curve')
plt.plot(phi_y,kk*0)
plt.ylim(-0.1,0.1)
plt.xlabel("y phase", fontdict={'size': 15})
plt.ylabel("mag ppe", fontdict={'size': 15})
plt.title('pixel phase error of mag',fontdict={'size': 15} )
plt.legend()
plt.savefig('%s_ppe_mag.png'%ncl)
plt.clf()