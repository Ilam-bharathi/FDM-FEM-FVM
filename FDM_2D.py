import numpy as np

import matplotlib.pyplot as plt

import scipy.sparse as sp

import scipy.sparse.linalg as la

LeftX=-1.5

RightX=1.5

LeftY=-1.5

RightY=1.5

Nx=100

Ny=100

dx=(RightX-LeftX)/Nx

dy=(RightY-LeftY)/Ny

LeftX=-1.5+dx

RightX=1.5-dx

LeftY=-1.5+dx

RightY=1.5-dx

Dx=(1/dx)*(sp.diags([-1, 1], [-1, 0], [Nx,Nx-1]))

Dy=(1/dy)*(sp.diags([-1, 1], [-1, 0],[Ny,Ny-1]))

Lxx=(Dx.transpose()).dot(Dx)

Lyy=(Dy.transpose()).dot(Dy)

Ix=sp.eye(Nx-1)

Iy=sp.eye(Ny-1)

A=sp.kron(Iy,Lxx)+sp.kron(Lyy,Ix)

plt.figure(figsize=(8,6),dpi=100)

plt.spy(A)

plt.show()

x,y=np.mgrid[LeftX:RightX:(Nx-1)*1j,LeftY:RightY:(Ny-1)*1j]

 

def FDLaplacian2D(Nx,Ny,dx,dy):

    LeftX=-1.5+dx

    RightX=1.5-dx

    LeftY=-1.5+dx

    RightY=1.5-dx

    Dx=(1/dx)*(sp.diags([-1, 1], [-1, 0], [Nx,Nx-1]))

    Dy=(1/dy)*(sp.diags([-1, 1], [-1, 0],[Ny,Ny-1]))

    Lxx=(Dx.transpose()).dot(Dx)

    Lyy=(Dy.transpose()).dot(Dy)

    Ix=sp.eye(Nx-1)

    Iy=sp.eye(Ny-1)

    A=sp.kron(Iy,Lxx)+sp.kron(Lyy,Ix)

    return A

def sourcefunc(x,y):

    f =  20*np.sin(np.pi*y)*np.sin(np.pi*x+np.pi)

    return f

f=sourcefunc(x,y)

def domainfunc(x,y):

    b = ((x**2+y**2-1)**3)-(x**2)*(y**3)

    return b

domain=domainfunc(x,y)


rows,cols,vals=sp.find(domain<0)

plt.figure(figsize=(8,6),dpi=100)

plt.plot(x,y)

plt.show()

minc=np.min(f)

ffill=minc*np.ones([Ny-1,Nx-1])

ffill[rows,cols]=f[rows,cols]

plt.ion()

plt.figure(1)

plt.clf()

plt.subplot(1,2,1)

plt.imshow(f)
plt.gca().invert_yaxis()
plt.colorbar(orientation='horizontal')

plt.subplot(1,2,2)

plt.imshow(ffill.transpose())
plt.gca().invert_yaxis()

plt.colorbar(orientation='horizontal')

domainLX = np.reshape(domain,((Nx-1)*(Ny-1),1))

rowsLX,colsLX,valsLX=sp.find(domainLX<0)

fLX = np.reshape(f,((Nx-1)*(Ny-1),1))

fLXd=fLX[rowsLX]


A=FDLaplacian2D(Nx,Ny,dx,dy)

Ad = A.tocsr()[rowsLX,:].tocsc()[:,rowsLX]


u=la.spsolve(Ad,fLXd)

minc=np.min(u)

ufill=minc*np.ones([Ny-1,Nx-1])

ufill[rows,cols]=u

plt.ion()

plt.figure(2)

plt.clf()

plt.subplot(1,2,2)

plt.imshow(ufill.transpose())
plt.gca().invert_yaxis()

plt.colorbar(orientation='horizontal')
