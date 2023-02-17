
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gaussian function,u means mean value and s means sigma
def gauss(x, y, u=0, s=2):
    g = (1 / (2 * np.pi * s ** 2)) * np.exp(-(((x - u) ** 2 + (y - u) ** 2) / (2 * s ** 2)))
    return g

# Angular varieties 
def Angular(l, q, theta):
    angular = 1
    return angular

# Laguerre function
def Laguerre(l, x):
    from scipy.special import laguerre
    return laguerre(l)(x)

# Moffatlets basis function
def Moffatlets(l, q, r, beta = 3.5, rd = 2):
    Vr = (1/(2*beta) -1) * np.log((1 + (r/rd)**2)**(-2*beta))
    # moffat = np.sqrt((2*beta - 1) / (np.pi*rd**2) ) * Laguerre(l,Vr) * (1 + (r/rd)**2)**(-beta) * Angular(l,q,theta)
    moffatlet = np.sqrt((2*beta - 1) / (np.pi*rd**2) ) * Laguerre(l,Vr) * (1 + (r/rd)**2)**(-beta)
    return moffatlet 

# Gaussianlets basis function
def Gaussianlet(l, r, sd=1):
    R = r**2/sd**2
    gaussianlet = (1 / np.sqrt(np.pi*sd**2)) * np.exp(-r**2/(2*sd**2)) * Laguerre(l,R)
    return gaussianlet


'---------------------------------------------------------------------'
# Vset =[[] for i in range(r[0].shape[0])]
# for i in range(r[0].shape[0]):
#     V = (1/(2*beta) -1) * math.log((1 + (r[0][i]/rd)**2)**(-2*beta))
#     # print(Vx)
#     Vset[i].append(V)
# Vr = np.array(Vset) 

# Vr = (1/(2*beta) -1) * np.log((1 + (r/rd)**2)**(-2*beta))
# print(Vr.shape)   

# mm = []
# for i in range(r[0].shape[0]):
#     m = np.sqrt((2*beta - 1) / (np.pi*rd**2) ) * 
#         np.outer(Laguerre(l,Vr[i]) , (1 + (r[0][i]/rd)**2)**(-beta))
#     # print(m)
#     mm.append(m)

# moff = np.array(mm)
# moff = np.sqrt((2*beta - 1) / (np.pi*rd**2) ) * 
#        Laguerre(l,Vr) * (1 + (r/rd)**2)**(-beta
# print(moff.shape)
'---------------------------------------------------------------------'

x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
xx, yy = np.meshgrid(x, y)

radial = np.sqrt(xx**2+yy**2) 

# gaussian = gauss(xx,yy)
gaussianlet = Gaussianlet(l=0, r=radial ,sd=2 )
moffatlet = Moffatlets(l=0, q=2, r=radial, beta=3.5, rd=6)    
print(gaussianlet.shape)
print(moffatlet.shape)

'---------------------------------------------------------------------'
# Gaussian function

# plt.figure()
# plt.imshow(gaussian,cmap="viridis")
# plt.title("Gaussian 2D function")
# plt.colorbar()
# plt.savefig("gau2d.png")
# plt.clf()

# fig = plt.figure()
# ax = Axes3D(fig)
# plt.xlabel('x')
# plt.ylabel('y')
# ax.plot_surface(xx, yy, gaussian, rstride=1, cstride=1, cmap="viridis")
# plt.savefig("gau3d.png")

'---------------------------------------------------------------------'
# Gaussianlets basis function
plt.figure()
plt.plot(gaussianlet[:,70:130])
plt.title("Gaussianlets 1D function")
plt.savefig("gaulet1d.png")
plt.clf()

plt.figure()
plt.imshow(gaussianlet,cmap="jet")
plt.title("Gaussianlet 2D function")
plt.colorbar()
plt.savefig("gaulet2d.png")
plt.clf()

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(xx, yy, gaussianlet, rstride=1, cstride=1, cmap="jet")
plt.savefig("gaulet3d.png")

'---------------------------------------------------------------------'
# Moffatlets basis function

plt.figure()
plt.plot(moffatlet)
plt.title("Moffat 1D function")
plt.savefig("moflet1d.png")
plt.clf()

plt.figure()
plt.imshow(moffatlet,cmap="jet")
plt.title("Moffat 2D function")
plt.colorbar()
plt.savefig("moflet2d.png")
plt.clf()

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(xx, yy, moffatlet, rstride=1, cstride=1, cmap="jet")
plt.savefig("moflet3d.png")

'---------------------------------------------------------------------'
plt.figure()
plt.imshow(moffatlet - gaussianlet,cmap="jet")
plt.title("Moffat-Gauss res 2D function")
plt.colorbar()
plt.savefig("m-g_res2d.png")
plt.clf()

plt.figure()
plt.plot(gaussianlet[:,100],label='gaussianlet')
plt.plot(moffatlet[:,100],color ='r',label='moffatlet')
plt.title("Gaussianlets and Moffatlet")
plt.legend(loc="best",fontsize=6)
plt.savefig("gau_moff_let1d.png")
plt.clf()
















# Laguerre polynomials

# from scipy.special import laguerre
# x = np.arange(-1.0, 5.0, 0.01)
# fig, ax = plt.subplots()
# ax.set_ylim(-5.0, 5.0)
# ax.set_title(r'Laguerre polynomials $L_n$')
# for n in np.arange(0, 5):
#     ax.plot(x, laguerre(n)(x), label=rf'$L_{n}$')
# plt.legend(loc='best')
# plt.show()
