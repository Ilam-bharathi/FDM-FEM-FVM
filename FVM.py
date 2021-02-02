import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
Lx = 8  # length in x direction
Ly = 8  # length in y direction
Nx =100 # number of intervals in x-direction
Ny = 100# number of intervals in y-direction
dx = Lx/Nx  # grid step in x-direction
dy = Ly/Ny  # grid step in y-direction
x=np.linspace(0,Lx,Nx+1) #x co-ordinates 
y=np.linspace(0,Ly,Ny+1) #y co-ordinates 

def sourceF(x,y): #source function
    alp=-10
    F=np.exp(alp*((x-3)**2)+alp*((y-5.5)**2))+np.exp(alp*((x-5)**2)+alp*((y-5.5)**2))+np.exp(alp*((x-1)**2)+alp*((y-1.7)**2))+np.exp(alp*((x-7)**2)+alp*((y-1.7)**2))+np.exp(alp*((x-2)**2)+alp*((y-2.2)**2))+np.exp(alp*((x-6)**2)+alp*((y-2.2)**2))+np.exp(alp*((x-3)**2)+alp*((y-2.5)**2))+np.exp(alp*((x-5)**2)+alp*((y-2.5)**2))+np.exp(alp*((x-4)**2)+alp*((y-2.6)**2))
    return(F) 

def coeffK1(x,y): #co-efficient function -1
    if(x<4):
        K = 0.1
    else:
        K = 1
    return K


def coeffK2(x,y): #co-efficient function -2
    K0 = 1
    alp=-10.0
    K = K0+np.exp(alp*((x-3)**2)+alp*((y-5.5)**2))+np.exp(alp*((x-5)**2)+alp*((y-5.5)**2))+np.exp(alp*((x-1)**2)+alp*((y-2.5)**2))+np.exp(alp*((x-7)**2)+alp*((y-2.5)**2))+np.exp(alp*((x-2)**2)+alp*((y-2)**2))+np.exp(alp*((x-6)**2)+alp*((y-2.2)**2))+np.exp(alp*((x-3)**2)+alp*((y-1.7)**2))+np.exp(alp*((x-5)**2)+alp*((y-1.7)**2))+np.exp(alp*((x-4)**2)+alp*((y-1.6)**2))
    return K

def createFun(x,y,funcName): #source function
    Fvec=np.zeros((len(x)-2)*(len(y)-2))
    k=0
    for j in range (len(y)):
        for i in range (len(x)):
            if (((x[i]>0) and y[j] > 0) and ((x[i] <8) and (y[j] < 8))):
                Fvec[k] = funcName(x[i],y[j])
                k=k+1
    return Fvec

fvec = createFun(x,y,sourceF)
kvec1 = createFun(x,y,coeffK1)
kvec2 = createFun(x,y,coeffK2)
f=(fvec.reshape(Nx-1,Ny-1)) 
K1=(kvec1.reshape(Nx-1,Ny-1))
K2=(kvec2.reshape(Nx-1,Ny-1))
plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(1,3,1)
plt.title("f(x,y)")
plt.imshow(f)
plt.gca().invert_yaxis()
x_label_list=np.arange(2,8,2)
plt.xticks((x_label_list/dx)-1,x_label_list)
plt.yticks((x_label_list/dx)-1,x_label_list)
plt.colorbar(orientation='horizontal')
plt.subplot(1,3,2)
plt.title("K1(x,y)")
plt.imshow(K1)
plt.gca().invert_yaxis()
plt.xticks((x_label_list/dx)-1,x_label_list)
plt.yticks((x_label_list/dx)-1,x_label_list)
plt.colorbar(orientation='horizontal')
plt.subplot(1,3,3)
plt.title("K2(x,y)")
plt.imshow(K2) 
plt.gca().invert_yaxis()
plt.colorbar(orientation='horizontal')
plt.xticks((x_label_list/dx)-1,x_label_list)
plt.yticks((x_label_list/dx)-1,x_label_list)
plt.show()

def create2DLFVM(Nx,Ny,coeffFunc,x,y):
    dm=[]
    h=dx
    dl=[]
    dr=[]
    dll=[]
    drr=[]
    for j in range (Ny):
        for i in range (Nx):
            if (x[i] and y[j] > 0.0):
                    if (x[i] and y[j] < 8.0):
                        a=-coeffFunc(x[i]-0.5*h,y[j])
                        a1=a
                        if(x[i]==h):
                            a=0
                        if (i,j)>(1,1):
                            dl=np.append(dl,a)
                        b=-coeffFunc(x[i],y[j]+0.5*h)
                        if ( j<Ny-1):
                            drr=np.append(drr,b)
                        c=-coeffFunc(x[i]+0.5*h,y[j])
                        c1=c
                        if(x[i]==8-h):
                            c=0
                        if (i,j)<(Nx-1,Ny-1):
                            dr=np.append(dr,c)
                        d=-coeffFunc(x[i],y[j]-0.5*h)
                        if (j>1):
                            dll=np.append(dll,d)
                        e = -(a1+b+c1+d)
                        dm=np.append(dm,e)                 
    A=sp.diags([dl,dm,dr,drr,dll],[-1,0,1,Nx-1,1-Nx],format='csc')
    return A

K = ((1/dx)**2)*create2DLFVM(Nx,Ny,coeffK1,x,y)
plt.spy(K)
plt.spy(K==1,color='black',label='1')
plt.spy(K==-0.25,color='red',label='-0.25')
plt.spy(K==-0.025,color='green',label='-0.025')
plt.spy(K==0.775,color='blue',label='0.775')
plt.legend()
plt.show()
u=la.spsolve(K,fvec)
u1=(u.reshape(Nx-1,Ny-1))
plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(1,1,1)
plt.imshow(u1)
plt.gca().invert_yaxis()
plt.xticks((x_label_list/dx)-1,x_label_list)
plt.yticks((x_label_list/dx)-1,x_label_list)
plt.colorbar(orientation='horizontal')
plt.show()

K = ((1/dx)**2)*create2DLFVM(Nx,Ny,coeffK2,x,y)
u=la.spsolve(K,fvec)
u1=(u.reshape(Nx-1,Ny-1))
plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(1,1,1)
plt.imshow(u1)
plt.gca().invert_yaxis()
plt.xticks((x_label_list/dx)-1,x_label_list)
plt.yticks((x_label_list/dx)-1,x_label_list)
plt.colorbar(orientation='horizontal')


