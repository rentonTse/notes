import galsim
import numpy as np


galimage = galsim.Image(psf, scale=1)

AM = galsim.hsm.FindAdaptiveMom(galimage, strict=False)

x_centroid_AM = AM.moments_centroid.x - 1
y_centroid_AM = AM.moments_centroid.y - 1
R = AM.moments_sigma * np.sqrt(2)
R2 = R**2

AM_shape = AM.observed_shape
e1 = AM_shape.e1
e2 = AM_shape.e2
e = AM_shape.e
q = AM_shape.q
pa = AM_shape.beta.deg % 180

# 如果计算二阶矩时不加权，选一个区域内（如中心0.5叫秒内的像素）图像计算的话可能噪声比较大。
def psfSecondMoments(psfMat, cenX, cenY, pixSize=1):
    apr = 0.5  # arcsec, 0.5角秒内测量
    fl = 28.0  # meters
    pxs = 2.5  # microns
    apr = np.deg2rad(apr / 3600.0) * fl * 1e6
    apr = apr / pxs
    apr = int(np.ceil(apr))

    I = psfMat
    ncol = I.shape[1]
    nrow = I.shape[0]
    w = 0.0
    w11 = 0.0
    w12 = 0.0
    w22 = 0.0
    for icol in range(ncol):
        for jrow in range(nrow):
            x = icol * pixSize - cenX
            y = jrow * pixSize - cenY
            rr = np.sqrt(x * x + y * y)
            wgt = 0.0
            if rr <= apr:
                wgt = 1.0
            w += I[jrow, icol] * wgt
            w11 += x * x * I[jrow, icol] * wgt
            w12 += x * y * I[jrow, icol] * wgt
            w22 += y * y * I[jrow, icol] * wgt
    w11 /= w
    w12 /= w
    w22 /= w
    sz = w11 + w22
    e1 = (w11 - w22) / sz
    e2 = 2.0 * w12 / sz
    r = np.sqrt(sz)
    e = np.sqrt(e1 * e1 + e2 * e2)
    return sz, e1, e2, r, e