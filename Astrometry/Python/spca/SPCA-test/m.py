import numpy as np
from matplotlib import pyplot as plt 

# Vr in eq.8
def vr_moffat(rd,beta,r):
    rs = r/rd
    tmp = np.power(1+rs*rs,-2*beta)
    f = (1-2*beta)/2*beta*np.log(tmp)
    return f

# lp for Ll(x),eq.5
def Plr_moffat(rd,beta,r,l):
    '''
    rd = 
    '''
    lp = [0 for i in range(100)]

    if l > 50:
        print('moffatlets order is larger than 50')
    if l < 0:
        print('moffatlets order is less than 50')

    x = vr_moffat(rd,beta,r)

    lp[0] = 1
    lp[1] = 1-x
    if l==0:
        f = lp[0]
    elif l==1:
        f = lp[1]
    else:
        cp1 = -1-x
        for i in range(2,int(l+1)):
            lp[i] = (2+cp1/i)*lp[i-1]-(1-1/i)*lp[i-2]
            f = lp[int(l)] 
    f = f*np.sqrt((2*beta-1)/np.arccos(-1))/rd*np.power(1+(r/rd)*(r/rd),-beta)

    return f
             


def Plr0_moffat(rd,beta,r,l):
    if l > 8 :
        print("moffatlets order is larger than 8,pause")
    if l < 0 :
        print("moffatlets order is less than 0,pause")
    x = vr_moffat(rd,beta,r)
    if l == 0:
        f = 1
    elif l == 1:
        f = -x + 1
    elif l == 2:
        f = (x**2 - 4*x + 2)/2   
    elif l == 3:
        f = (-x**3 + 9*x**2 - 18*x +6)/6
    elif l == 4:
        f = (x**4 - 16*x**3 + 72*x**2 - 96*x + 24)/24 
    elif l == 5:
        f = (-x**5 + 25*x**4 - 200*x**3 + 600*x**2 - 600*x + 120)/120
    elif l == 6:
        f = (x**6 - 36*x**5 + 450*x**4 - 2400*x**3 + 5400*x**2 - 4320*x + 720)/720
    elif l == 7:
        x2=x*x
        x3=x*x*x
        x4=x2*x2
        f = (-x3*x4+49*x3*x3-882*x2*x3+7350*x4-29400*x3+52920*x2-35280*x+5040)/5040
    elif l == 8:
        x2=x*x
        x3=x*x*x
        x4=x2*x2
        f=(x4*x4-64*x4*x3+1568*x3*x3-18816*x2*x3+117600*x4-376320*x3+564480*x2
       -322560*x+40320)/40320

    else: 
        print("Moffatlets order l is illegal")
    f = f * np.sqrt((2*beta -1)/2*np.pi) /rd * (1+(r/rd)**2)**(-beta)
    return f

def Plr_gauss(sigma,r,l):
    
    lp = [0 for i in range(100)]

    if l > 50:
        print('gaussianlets order is larger than 50')
    if l < 0:
        print('gaussianlets order is less than 50')

    r2 = r*r
    sigma2 = sigma*sigma
    x = r2/sigma2

    lp[0] = 1
    lp[1] = 1-x
    if l==0:
        f = lp[0]
    elif l==1:
        f = lp[1]
    else:
        cp1 = -1-x
        for i in range(2,int(l+1)):
            lp[i] = (2+cp1/i)*lp[i-1]-(1-1/i)*lp[i-2]
            f = lp[int(l)]
    f = 1/(np.sqrt(np.pi)*sigma)*np.exp((-0.5)*x)*f 
    return f
        









# x = np.arange(-10, 10, 0.1)
# y = np.arange(-10, 10, 0.1)
# xx, yy = np.meshgrid(x, y)

# radial = np.sqrt(xx**2+yy**2) 

# tt = Plr_moffat(rd=1.4,beta=3.5,r=radial,l=0)
# ss = Plr_gauss(sigma=2,r=radial,l=8)

# plt.figure()
# plt.imshow(,cmap="jet")
# plt.colorbar()
# plt.show()