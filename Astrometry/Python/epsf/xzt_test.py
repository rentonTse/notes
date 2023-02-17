import numpy as np
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
from photutils.centroids import centroid_com
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils import IntegratedGaussianPRF
import sys
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


imagehdu=fits.getheader('/home/xzt/Desktop/CSST.fits')
imagedata=fits.getdata('/home/xzt/Desktop/CSST.fits')
imagedata=imagedata.astype(np.float64)
print(imagedata.shape)


data = imagedata[600:900,600:900]
print(data.shape)
plt.imshow(data,cmap='gray')
plt.colorbar()
plt.show()




nddata = NDData(data=data)

peaks_tbl = find_peaks(data, threshold=1000.)  
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

mean_val, median_val, std_val = sigma_clipped_stats(data,sigma=3.)  
data -= median_val  

# invert to nddata type because of the function extract_stars
nddata = NDData(data=data)

# put all the good(to fit) stars to a list,each star has a size of 25(11), lenth of the list is the number of stars
stars = extract_stars(nddata, stars_tbl, size=(25,25))
print(len(stars)) 




nrows = 2
ncols = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15),
                       squeeze=True)
ax = ax.ravel()
for i in range(nrows*ncols):
    norm = simple_norm(stars[i], 'log', percent=99.)

    ax[i].imshow(stars[i],norm =norm ,origin='lower', cmap='viridis')
    ax_divider = make_axes_locatable(ax[i])
    plt.show()    

epsf_builder = EPSFBuilder(oversampling=4, maxiters=10,
                           progress_bar=True)
epsf, fitted_stars = epsf_builder(stars)   
norm = simple_norm(epsf.data, 'log', percent=99.)
plt.imshow(epsf.data,norm = norm ,origin='lower', cmap='viridis')
plt.colorbar()
plt.show()

plt.plot(epsf.data[50, :])
plt.show()