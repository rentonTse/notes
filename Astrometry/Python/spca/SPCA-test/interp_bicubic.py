import numpy as np


def spline(y,n,yp1,ypn):
    
    u = np.zeros((n-1,1))
    # y2 = np.zeros((n,1))
    y2 = [[] for i in range(n)]
    if yp1 > 0.99e30:
        y2[0] = u[0] = 0.0
    else:
        y2[0] = -0.5
        u[0] = 3.0*(y[1]-y[0]-yp1)
    sig = 0.5
    for i in range(1,n-1):
        p = sig*y2[i-1]+2.0
        y2[i]=(sig-1.0)/p
        u[i]=(y[i+1]-2.*y[i]+y[i-1])
        u[i]=(3.0*u[i]-sig*u[i-1])/p
    if ypn > 0.99e30:
        qn = un = 0.0
    else:
        qn = 0.5
        un=3.0*(ypn-y[n-1]+y[n-2])
    y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0)
    for k in range(n-2,0,-1):
        y2[k]=y2[k]*y2[k+1]+u[k]
    
    return y2


def splint(ya,y2a,n,x):
    klo = x
    if klo < 0:
        klo = 0
    if klo == n-1:
        klo = n-2
    khi = klo + 1
    if klo < 0 or khi >= n:
        print('Bad xa input to routine splint')
    a = (khi-x)
    b = (x-klo)
    y = a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])/6.0     

    return y

def splie2(ya,m,n):
    y2a = np.zeros((m,n))
    for j in range(m):
        y2aj = spline(y=ya[j],n=n,yp1=1.0e30,ypn=1.0e30)
        y2a[j] = y2aj
    return y2a

def Interp_bicubic(nx,ny,a0,nbound,nxt,nyt,xcen):
    yytmp=np.zeros((ny,1))
    at = np.zeros((nxt,nyt))
    y2a = splie2(ya=a0,m=nx,n=ny)
    ic = nx*0.5
    jc = ny*0.5
    shift1 = xcen[0]-ic
    shift2 = xcen[1]-jc
    if np.fabs(shift1)>nbound or np.fabs(shift2)>nbound:
        print("bicubic shifts too much %e %e %d\n" %(shift1,shift2,nbound))

    for n in range(nyt):
        x2 = n+nbound+shift2
        for i in range(nx):
            yytmp[i] = splint(ya=a0[i],y2a=y2a[i],n=ny,x=x2)
            ytmp = spline(y=yytmp,n=ny,yp1=1.0e30,ypn=1.0e30)
        for m in range(nxt):
            x1=m+nbound+shift1
            at[m][n] = splint(ya=yytmp,y2a=ytmp,n=ny,x=x1)
    
    return at

def centriod_gaus2D(nx,ny,nxt,nyt,nbound,para):
    yt = np.zeros((nxt,nyt))
    kc=para[0]
    sigmac=para[1]
    bgc=para[2]
    xc=para[3]
    yc=para[4]
    sigma2v=-1./(2.*sigmac*sigmac)       
    ic=nx*0.5
    jc=ny*0.5
    shift1=para[3]-ic
    shift2=para[4]-jc

    for i in range(nxt): 
        x=i+shift1+nbound
        for j in range(nyt):
            y=j+shift2+nbound
            det = (x-xc)*(x-xc)+(y-yc)*(y-yc)
            yt[i][j]=kc*np.exp(det*sigma2v)+bgc    

    return yt        