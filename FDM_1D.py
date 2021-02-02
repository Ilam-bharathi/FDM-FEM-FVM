# Solution of 1D Poisson's equation with FDM
# Ilambharathi Govindasamy (c) 2020
import numpy as np
import matplotlib.pyplot as plt

n=5
s=n+1
h=2/n
xgrid=np.linspace(-1,1,s)
uo=1
un=2
def func1(x):
    f=np.ones(s)
    return(f)

def func2(x):
    f=np.exp(x)
    return(f)

def func3(x):
    f=2+(x/2)-(x**2)/2
    return(f)

def func4(x):
    f=-np.exp(x)+(1.675201194
*x)+3.043080635

    return(f)

f1=func1(xgrid)
f1rhs=f1[1:n]
f2=func2(xgrid)
f2rhs=f2[1:n]
u1ex=func3(xgrid)
u2ex=func4(xgrid)
plt.figure(figsize=(7, 5), dpi=100)
plt.plot(xgrid,f1,color="blue",marker="o",fillstyle="full",label=r'$f_1$')
plt.plot(xgrid,f2,color="red",marker="o",fillstyle="full",label=r'$f_2$')
plt.xlabel("x position")
plt.ylabel("f(x)")
plt.title("Functions")
plt.legend(loc=0)
plt.savefig("Func.pdf")

plt.figure(figsize=(7, 5), dpi=100)
plt.plot(xgrid,u1ex,color="blue",marker="o",fillstyle="full",label=r'$u_1^{ex}$')
plt.plot(xgrid,u2ex,color="red",marker="o",fillstyle="full",label=r'$u_2^{ex}$')
plt.xlabel("x position")
plt.ylabel("f(x)")
plt.title("Exact solutions")
plt.legend(loc=0)
plt.savefig("Exact_sol.pdf")

q=np.ones(n-2)
w=np.ones(n-1)*(-2)
au=np.diag(q,k=1)+np.diag(w)+np.diag(q,k=-1)
a=-(1/h**2)*au
em,ev=np.linalg.eig(a)
print("eigen value=", em)
plt.figure(figsize=(7, 5), dpi=100)
plt.spy(a,color="green",marker="o")
plt.savefig("Struc.pdf")

f2rhs[0]=f2rhs[0]+(uo/h**2)
f2rhs[n-2]=f2rhs[n-2]+(un/h**2)
f1rhs[0]=f1rhs[0]+(uo/h**2)
f1rhs[n-2]=f1rhs[n-2]+(un/h**2)

ua1 = np.linalg.solve(a,f1rhs)
ua2 = np.linalg.solve(a,f2rhs)

u1=np.insert(ua1,0,uo)
u1=np.append(u1,un)

u2=np.insert(ua2,0,uo)
u2=np.append(u2,un)

plt.figure(figsize=(7,5), dpi=100)
plt.plot(xgrid,u1ex,color="blue",marker="o",fillstyle="full",label=r'$u_1^{ex}$')
plt.plot(xgrid,u2ex,color="red",marker="o",fillstyle="full",label=r'$u_2^{ex}$')
plt.plot(xgrid,u1,color="blue",marker="o",fillstyle="none",linestyle="dashed",label=r'$u_1$')
plt.plot(xgrid,u2,color="red",marker="o",fillstyle="none",linestyle="dashed",label=r'$u_2$')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison")
plt.legend(loc=0)
plt.savefig("Compare.pdf")
e1=((1/np.sqrt(n-1))*np.linalg.norm(u1-u1ex))
print("Error for function 1 :",e1)
e2=((n-1)**(-1/2))*np.linalg.norm(u2-u2ex)
print("Error for function 2 :",e2)
ex1=[]
ex2=[]
nx=[]
i=0
for n in range(5, 100):
    s=n+1
    h=2/n
    xgrid=np.linspace(-1,1,s)
    uo=1
    un=2
    def func1(x):
        f=np.ones(s)
        return(f)

    def func2(x):
        f=np.exp(x)
        return(f)

    def func3(x):
        f=2+(x/2)-(x**2)/2
        return(f)

    def func4(x):
        f=-np.exp(x)+(1.675201194
    *x)+3.043080635

        return(f)

    f1=func1(xgrid)
    f1rhs=f1[1:n]
    f2=func2(xgrid)
    f2rhs=f2[1:n]
    u1ex=func3(xgrid)
    u2ex=func4(xgrid)
    q=np.ones(n-2)
    w=np.ones(n-1)*(-2)
    au=np.diag(q,k=1)+np.diag(w)+np.diag(q,k=-1)
    a=-(1/h**2)*au
    em=np.linalg.eig(a)


    f2rhs[0]=f2rhs[0]+(uo/h**2)
    f2rhs[n-2]=f2rhs[n-2]+(un/h**2)
    f1rhs[0]=f1rhs[0]+(uo/h**2)
    f1rhs[n-2]=f1rhs[n-2]+(un/h**2)

    ua1 = np.linalg.solve(a,f1rhs)
    ua2 = np.linalg.solve(a,f2rhs)

    u1=np.insert(ua1,0,uo)
    u1=np.append(u1,un)
    u2=np.insert(ua2,0,uo)
    u2=np.append(u2,un)
    ex1.append(((n-1)**(-1/2))*np.linalg.norm(u1-u1ex))
    ex2.append(((n-1)**(-1/2))*np.linalg.norm(u2-u2ex))
    nx.append(n)
plt.figure(figsize=(7,5), dpi=100)
plt.loglog(nx,ex2)
plt.xlabel("Log(n)")
plt.ylabel("Log(Error)")
plt.savefig("Er2.pdf")
plt.figure(figsize=(7,5), dpi=100)
plt.loglog(nx,ex1)
plt.xlabel("Log(n)")
plt.ylabel("Log(Error)")
plt.savefig("Er1.pdf")
slope, intercept = np.polyfit(np.log(nx), np.log(ex2), 1)
print("Order for error=",slope)