import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from numpy import linalg as LA
from time import process_time 
from scipy.sparse import coo_matrix, bmat
import matplotlib.animation as animation
# Start the stopwatch / counter  
t1_start = process_time()  

plt.figure()
Ix=sp.eye(4)
Iy=sp.eye(4)
Lxx=Lyy=[[1, -1, 0, 0],[-1, 2, -1, 0],[0, -1, 2, -1],[0, 0, -1, 1]]
K=sp.kron(Iy,Lxx)+sp.kron(Lyy,Ix)
plt.spy(K)
plt.spy(K==-1,color='black',label='-1')
plt.spy(K==2,color='red',label='2')
plt.spy(K==3,color='green',label='3')
plt.spy(K==4,color='blue',label='4')

plt.figure()
Dx=np.array([[1, 0, 0, 0, 0],[-1, 1, 0, 0, 0],[0, -1, 1, 0, 0], [0, 0, -1, 0, 0]])
Axx=(Dx).dot(Dx.transpose())
K=sp.kron(Iy,Axx)+sp.kron(Axx,Ix)
plt.spy(K)
plt.spy(K==-1,color='black',label='-1')
plt.spy(K==2,color='red',label='2')
plt.spy(K==3,color='green',label='3')
plt.spy(K==4,color='blue',label='4')
plt.legend()
plt.show()
x=np.linspace
Nx=Ny=100 #number of intervals
Lx=Ly=4
h=Lx/Nx
Ix=sp.eye(Nx)

Iy=sp.eye(Ny)
b=np.append(np.ones(Nx-1),0)
k=sp.diags([-1], [1], [Nx+1, Ny])
m=sp.diags(b,0,[Nx+1,Ny])
Dxx=k+m
Axx=(Dxx.transpose()).dot(Dxx)
A=(1/h**2)*(sp.kron(Iy,Axx)+sp.kron(Axx,Ix))
D_u=0.05
D_v=1

rk=5 #reaction constant
a=0.1305
b=0.7695
rxy=0.01*(a+b)*np.random.rand(Nx*Ny,1)
u0=a+b+rxy 
v0=b/(a+b)**2
#given
T = 20 
#Nt
nt = 50000
dt = T / nt 
u_n=(u0)
s_n= (v0*np.ones((Nx*Ny,1)))
t = 0
uss=[]
#Figure
plt.figure()
p=1
plt.imshow(s_n.reshape(Nx,Ny))
plt.show()
while t < T:
    for i in range (Nx*Ny):
        uss=np.append(uss,u_n[i]*u_n[i]*s_n[i])
    uss=uss.reshape(Nx*Ny,1)
    #Solve
    u=(u_n + dt*((-D_u*A*u_n)+rk*(a-u_n+uss)))
    s=(s_n + dt*((-D_v*A*s_n)+rk*(b-uss)))
    #Update new u
    u_n=u
    s_n=s
    if (t>p):
        plt.imshow(u.reshape(Nx,Ny))
        plt.draw()
        plt.pause(0.2)
        plt.clf()
        p=p+1
    # Update time
    t = t + dt
    uss=[]
# Stop the stopwatch / counter
plt.imshow(u.reshape(Nx,Ny))
plt.draw()
plt.pause(0.2)
plt.clf()
t1_stop = process_time() 
print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)
print("Nt="+str(nt))

# #Newton_Raphson
# def split_list(a_list):
#     half = len(a_list)//2
#     return a_list[:half], a_list[half:]

# uu=[]
# uv=[]
# us=[]
# t=0
# u_n=np.array(u0)
# s_n= np.array(v0*np.ones((Nx*Ny,1)))
# u_k=np.concatenate((u_n,s_n))

  
# T=20
# nt = 4500 
# dt = T / nt 

# while t < T:
#     u_i=u_k
#     u_a,s_a=split_list(u_i)
#     for i in range (Nx*Ny):
#         us=np.append(us,(u_a[i]*u_a[i]*s_a[i]))
#     us=us.reshape(Nx*Ny,1)
#     uex=np.concatenate(((a-s_a+us),(b-us)),axis=0)
#     ID=sp.diags([D_u,D_v],0)
#     dA=sp.kron(ID,A.toarray()).toarray()
#     f=u_i+dt*(np.dot(-dA,u_i)+rk*uex)
#     nm=np.array((u_k+(dt*f)-u_i))
#     us=[]
#     while (LA.norm(nm)>10**-3):
#         u_a,s_a=split_list(u_i)
#         for i in range (Nx*Ny):
#             us=np.append(us,(s_a[i]*u_a[i]**2))
#         us=us.reshape(Nx*Ny,1)
#         ID=sp.diags([D_u,D_v],0)
#         dA=sp.kron(ID,A.toarray()).toarray()
#         uex=np.concatenate(((a-s_n+us),(b-us)),axis=0)
        
#         for i in range (Nx*Ny):
#             uu=np.append(uu,u_a[i]*u_a[i])
#         uu=np.array(uu.reshape((Nx*Ny,1)))
#         uui=(sp.diags(uu.reshape((1,Nx*Ny)),[0]).toarray())

#         for j in range (Nx*Ny):
#             uv=np.append(uv,u_a[i]*s_a[i])
#         uv=np.array(uv.reshape((Nx*Ny,1)))
#         uvi=sp.diags(uv.reshape((1,Nx*Ny)),[0]).toarray()
#         j11=coo_matrix(-D_u*A+rk*(-sp.eye(Nx*Ny)+2*uvi))
#         j12=coo_matrix(rk*uui)
#         j21=coo_matrix(-2*rk*uvi)
#         j22=coo_matrix(-D_v*A+(rk*uui))
#         J=bmat([[j11, j12], [j21, j22]]).toarray()
#         f=u_i+dt*(np.dot(-dA,u_i)+rk*uex)
#         nm=np.array((u_k+(dt*f)-u_i))
#         r=np.linalg.inv((sp.eye(len(J))).toarray()-dt*J)
#         u_i1=u_i+np.dot(r,nm)
#         u_i=u_i1
#         uu=[]
#         uv=[]
#         us=[]
#     u_k=u_i1
#     t=t+dt
#     plt.figure(1)
#     if(t==5):
#         plt.imshow(s_a.reshape(Nx,Ny))
#         print(5)
# plt.imshow(u_a.reshape(Nx,Ny))
# plt.show()
# plt.imshow(s_a.reshape(Nx,Ny))
# plt.show()