from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from matplotlib.ticker import FuncFormatter
from astropy.modeling.functional_models import Moffat1D, Gaussian1D,Moffat2D,Gaussian2D





img_ratio = 100. # amplification factor for total flux in order to get a better imshow
pixel_scale = 1 # arcsec / pixel
image_size = 13
sigma = 2.  

f = image_size / 2.*img_ratio # devide each pixel into f subpixels
x,y = np.mgrid[-f:f:complex(0, 2*f),-f:f:complex(0, 2*f)] 
sigma = sigma * img_ratio 
gaussian = 1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2))
gaussian1 = gaussian * img_ratio * img_ratio  
print(gaussian1.shape)

plt.figure()
plt.imshow(gaussian1,cmap="jet")
plt.title("Precise Gaussian PSF")
# plt.grid(linestyle='-.', which="major")
plt.xlim(0,int(img_ratio * image_size))
plt.ylim(int(img_ratio * image_size),0)
plt.colorbar()
plt.savefig("gaussian2d.png")

