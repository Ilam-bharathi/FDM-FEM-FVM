from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

#given
T = 10            
ns = 20     
dt = T / ns 
alp = 50         
omega=2*3.14/(4*T)
D=Constant(0.005)
A=5
B=0.5
Lx=10
Ly=10
nx = 200
ny = 200


# Rectangle domain with mesh
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),nx,ny)
# Finite element with Lagrange basis function
P1 = FiniteElement('Lagrange',triangle,1) #linear
# Function space V for scalar functions
V = FunctionSpace(mesh, P1)

u_D = Constant(0.)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# initial value
u_n = interpolate(u_D, V)


# Galerkin
u = TrialFunction(V)
v = TestFunction(V)
f=Expression('A*exp(-alp*(pow((x[0]-1.2),2))-alp*pow((x[1]-2),2))+A*exp(-alp*pow(x[0]-1.5,2)-alp*pow(x[1]-2.5,2))+A*exp(-alp*pow(x[0]-4,2)-alp*pow(x[1]-3,2))',degree=2,alp=alp,A=A)
f = interpolate(f, V)
# Function space W for vector functions
W = FunctionSpace(mesh, MixedElement([P1,P1]))
# Velocity vector field
velocity = Expression(('B*cos(omega*t)','B*sin(omega*t)'),degree=2,B=B,omega=omega,t=0.0)
w = Function(W)
w.interpolate(velocity)

#Weak-form Convection-Diffusion equation
a= u*v*dx + D*dt*dot(grad(u), grad(v))*dx +dt*dot(w,grad(u))*v*dx
L =  (u_n + dt*f)*v*dx

# Time-stepping
u = Function(V)
t = 0

#Figure
plt.ion()
plt.figure(1)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
points = mesh.coordinates()
unow = u.compute_vertex_values(mesh)
im = ax.tripcolor(points[:,0],points[:,1],unow)
ax.set_aspect('equal', adjustable='box')
cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
ax.set_title(r'$u(t)$,t = '+str(np.round(t,2))) 


for n in range(ns):

    # Update time
    t = t + dt
    
    velocity.t = t # updating time inside velocity
    w.interpolate(velocity) # updating w

    #Solve
    solve(a == L, u, bc)
    plot(u)

    # Update new u
    u_n.assign(u)

    #Figure
    unow = u.compute_vertex_values(mesh)
    im.remove()
    im=ax.tripcolor(points[:,0],points[:,1],unow)
    cbar.update_normal(im)
    ax.set_title(r'$u(t)$, at t='+str(np.round(t,2)))
    plt.pause(0.2) 
    plt.show()
    if(t==5): #5 or 10
        break
