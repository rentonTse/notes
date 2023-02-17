import numpy as np


def search_2D(x,y,npp,kc,kd,sigmac,sigmad,bgc,bgd,xc,xd,yc,yd,bg0,fbg):
    '''
    from fit_gaus.c L69
    retrun para[5]    
    '''
    para = np.zeros((6,1))
    sigma2v = -1./(2.*sigmac*sigmac)
    bg = bgc
    k20 = 0
    for m in range(npp):
        det = (x[m][0]-xc)*(x[m][0]-xc)+(x[m][1]-yc)*(x[m][1]-yc)
        det = kc*np.exp(det*sigma2v)+bgc-y[m]
        k20 += det*det

    for i in range(-4,5):
        k = kc+i*kd
        if k > 0:
            for j in range(-4,5):
                sigma = sigmac + j*sigmad
                if sigma > 0:
                    sigma2v = sigma2v = -1./(2*sigma*sigma)
                    for p in range(-4,5):
                        xt = xc+p*xd
                        for q in range(-4,5):
                            yt=yc+q*yd
                            k2 = 0
                            if fbg == 0:
                                bg = bg0
                                for m in range(npp):
                                    det = (x[m][0]-xt)*(x[m][0]-xt)+(x[m][1]-yt)*(x[m][1]-yt)
                                    det = k*np.exp(det*sigma2v)+bg-y[m]
                                    k2 += det*det
                            else:
                                for l in range(-4,5):
                                    bg = bgc+l*bgd
                                    for m in range(npp):
                                        det = (x[m][0]-xt)*(x[m][0]-xt)+(x[m][1]-yt)*(x[m][1]-yt)
                                        det = k*np.exp(det*sigma2v)+bg-y[m]
                                        k2 += det*det
                            if k2 <= k20:
                                k20 = k2
                            para[5] = k2 
                            para[0] = k
                            para[1] = sigma
                            para[2] = bg
                            para[3] = xt
                            para[4] = yt
    # dff
    return para                               

def gasfit_2D(x,y,npp,bg0,fbg):
    '''
    from fit_gaus.c L111
    '''
    imax = 0
    ymin = ymax = y[imax]
    for i in range(npp):
        if ymax<y[i]:
            imax = i
            ymax = y[i]
        if ymin>y[i]:
            ymin=y[i]
    kc = ymax
    kd = ymax/12
    det = ysigma = kc*np.exp(-0.5)
    for i in range(npp):
        dett = np.fabs(ysigma-y[i])
        if dett < det:
            det = dett
    isigma = i
    # dff
    xc = x[imax][0]
    yc = x[imax][1]
    sigmac=np.sqrt((xc-x[isigma][0])*(xc-x[isigma][0])+(yc-x[isigma][1])*(yc-x[isigma][1]))
    xd=yd=sigmac*0.25
    sigmad=0.25*sigmac
    bgc=0.
    bgd=np.fabs(ymin)

    para = search_2D(x=x,y=y,npp=npp,kc=kc,kd=kd,sigmac=sigmac,sigmad=sigmad,bgc=bgc,bgd=bgd,xc=xc,xd=xd,yc=yc,yd=yd,bg0=bg0,fbg=fbg)
    print(para.shape)
    kd*=0.33
    sigmad*=0.33
    bgd*=0.33
    xd*=0.33
    yd*=0.33
    kc=para[0]
    sigmac=para[1]
    bgc=para[2]
    xc=para[3]
    yc=para[4]
    if fbg == 0:
        para[2] = bg0
    return para


def star_gaus(y,nx,ny):
    '''
    from fit_gaus.c L69
    y(nx,ny)
    '''
    bg0 = 0
    fbg = 0    
    npp = nx*ny
    xs = np.zeros((npp,2))
    xi = np.zeros((npp,2))
    ys = np.zeros((npp,1))
    yt = np.zeros((nx,ny))
    residu = np.zeros((nx,ny))

    ymax = y[0][0]
    for i in range(int(nx/2)):
        for j in range(int(ny/2)):
            ix = int(i + nx/4)
            iy = int(j + ny/4)
            if ymax < y[ix][iy]:
                ymax = y[ix][iy]
    imx = ix
    jmx = iy         
    # dff fit_gaus.cL84

    npt = 0
    for i in range(-3,4):
        for j in range(-3,4):
            xs[npt][0]=xi[npt][0]=i+imx
            xs[npt][1]=xi[npt][1]=j+jmx
            ys[npt]=y[imx+i][jmx+j]
            npt += 1

    para = gasfit_2D(x=xs,y=ys,npp=npt,bg0=bg0,fbg=fbg)
    kc=para[0]
    sigmac=para[1]
    bgc=para[2]
    xc=para[3]
    yc=para[4]    
    sigma2v=-1./(2.*sigmac*sigmac)
    for i in range(nx):
        for j in range(ny):
            det = (i-xc)*(i-xc)+(i-yc)*(j-yc)
            ds = det*sigma2v
            yt[i][j] = kc*np.exp(ds) + bgc
            residu[i][j]=y[i][j]-yt[i][j]
    return yt,para     