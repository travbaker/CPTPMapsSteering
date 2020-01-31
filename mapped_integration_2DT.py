# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:33:37 2018

File to construct displaced T-states (t_1=t_2, t_3, a_z all nonzero) with LHS models, considering deterministic maps on Alice's side.
This script performs the optimization over u, v in Eq. (24), which are used to create Fig. 3(a).
Optimization results coincide with Eq. (25) of the text.

@author: Trav Baker, travis.baker@griffithuni.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
import time
import qutip as qt

# First, pick the size of Alice's Bloch vector.
a_z = 0.1 #=sin(u)*sin(v) in terms of map parameters.

#Functions to compute Eqs. (7) and (22)
def NS(phi,z,u,v,r,theta):
    return np.sqrt((1-z**2)*np.cos(phi)**2*(r*np.sin(theta)/np.cos(u))**2/2 + \
                   (1-z**2)*np.sin(phi)**2*(r*np.sin(theta)/np.cos(v))**2/2 + \
                   z**2*(r*np.cos(theta)/(np.cos(u)*np.cos(v)))**2)

def func(r,u,v,theta):
    I,err=dblquad(NS,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(u,v,r,theta))
    return 2*np.pi - I
        
def integrand(phi,z,u,theta):
        v=np.arcsin(a_z/np.sin(u))
        return np.sqrt((1-z**2)*np.sin(phi)**2*(np.sin(theta)/np.cos(u))**2 + \
                   (1-z**2)*np.cos(phi)**2*(np.sin(theta)/np.cos(v))**2 + \
                   z**2*(np.cos(theta)/(np.cos(u)*np.cos(v)))**2)

def objective(u,theta): #v is defined by x (i.e. u) and a_z.
    I,err=dblquad(integrand,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(u,theta))
    r = 2*np.pi/I
    return -r

def bowles_objective(z,theta,a_z,tau_3,sign=-1):
    return sign*((a_z*z)**2 + 2*np.abs(tau_3)*np.sqrt(np.tan(theta)**2*(1-z**2) + z**2))

def bowles_t3_root(tau_3,theta,a_z):
    res = minimize_scalar(bowles_objective, method='Bounded', bounds=(-1,1), args=(theta,a_z,tau_3))
    return -res.fun - 1

#Solve Eq. (7):
bow_1=[]
bow_3=[]
start=time.time()
nbow=100 #resolution of angular points for Eq. (7)
for theta in np.linspace(0,np.pi/2,nbow)[1:-1]:
    bow_res,out = brentq(bowles_t3_root,0,1,args=(theta,a_z),xtol=1e-12, maxiter=100, full_output=True)
    if out.converged == True:
        bow_3.append(bow_res)
        bow_1.append(bow_res*np.tan(theta))

#Compute Eq. (22)
ntheta = 100 #number of points to construct curve
theta_vec = np.linspace(0,np.pi/2,ntheta)
tau_1=[]
tau_3=[]
tau_1.append(0)
tau_3.append(1-a_z)
u_opt=[]
for theta in theta_vec[1:-1]:
    res = minimize_scalar(objective, method='Bounded', bounds=(np.arcsin(a_z),np.pi/2), args=(theta))
    I, err = dblquad(integrand,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(res.x,theta))
    u_opt.append(res.x)
    tau_3.append(np.cos(theta)*2*np.pi/I)
    tau_1.append(np.sin(theta)*2*np.pi/I)
    print('Finished angle %s of %s' %(np.where(theta_vec == theta)[0][0],ntheta-2))

#Quantum state boundary
def least_eig(r,theta,a_z):
    rho=(1/4)*(qt.tensor(qt.qeye(2), qt.qeye(2)) + qt.tensor(a_z*qt.sigmaz(), qt.qeye(2))\
               - r*np.sin(theta)/np.sqrt(2)*qt.tensor(qt.sigmax(),qt.sigmax())\
                   - r*np.sin(theta)/np.sqrt(2)*qt.tensor(qt.sigmay(),qt.sigmay())\
                       - r*np.cos(theta)*qt.tensor(qt.sigmaz(),qt.sigmaz()))
    eigs = np.linalg.eigvalsh(rho.full())
    return min(eigs)

npoints=100 #resolution. 
qangle=np.linspace(0,np.pi/2,npoints) #angle from -y axis
x_border=[]
y_border=[]
for i in qangle:
    root=brentq(least_eig,0,2,args=(i,a_z))
    x_border.append(-root*np.sin(i)/np.sqrt(2))
    y_border.append(-root*np.cos(i))
    print(r'%.2f%% of Q-boundary points' %(2*i/np.pi*npoints))
    
#Entangled state boundary
def least_ent_eig(r,theta,a_z):
    rho=(1/4)*(qt.tensor(qt.qeye(2), qt.qeye(2)) + qt.tensor(a_z*qt.sigmaz(), qt.qeye(2))\
               - r*np.sin(theta)/np.sqrt(2)*qt.tensor(qt.sigmax(),qt.sigmax())\
                   - r*np.sin(theta)/np.sqrt(2)*qt.tensor(qt.sigmay(),qt.sigmay())\
                       - r*np.cos(theta)*qt.tensor(qt.sigmaz(),qt.sigmaz()))
    rho_pt = qt.partial_transpose(rho, [0,1])
    eigs = np.linalg.eigvalsh(rho_pt.full())
    return min(eigs)

npoints=100
qangle=np.linspace(0,np.pi/2,npoints) #angle from -y axis
xent_border=[]
yent_border=[]
for i in qangle:
    root=brentq(least_ent_eig,0,2,args=(i,a_z))
    xent_border.append(-root*np.sin(i)/np.sqrt(2))
    yent_border.append(-root*np.cos(i))
    print(r'%.2f%% of Entangled-boundary points' %(2*i/np.pi*npoints))

end=time.time()
minutes=(end-start)/60
print('%f minutes elapsed' %minutes)
bow_1[-1]=bow_1[-2]
bow_3[-1]=0

#save data to a file in the present 
np.savez('a_z%s_2DT_%sintegrals_data.npz' %(a_z,ntheta), tau_1, tau_3, bow_1, bow_3, x_border, y_border, xent_border, yent_border)
      
data = np.load('a_z%s_2DT_%sintegrals_data.npz' %(a_z,ntheta))
tau_1, tau_3, bow_1, bow_3, x_border, y_border, xent_border, yent_border = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6'], data['arr_7']

#Analytic curve from Eq. (25), based on the ansatz u=v:
alphas=np.linspace(0,np.pi/2,1000)
def radius(alpha,a):
    c= 2/(np.tan(alpha)**2*(1-a)) - 1
    if c<0:
        return np.sqrt(2*(1-a))/np.sin(alpha)*(np.sqrt(1+c) + np.arcsin(np.sqrt(-c))/np.sqrt(-c))**-1
    elif c>0:
        return np.sqrt(2*(1-a))/np.sin(alpha)*(np.sqrt(1+c) + (2*np.sqrt(c))**(-1)*np.log((np.sqrt(1+c)+np.sqrt(c))/(np.sqrt(1+c)-np.sqrt(c))))**(-1)

t1prime=[]
t3prime=[]
for alpha in alphas[1:-129]:
    t1prime.append(radius(alpha,a_z)*np.sin(alpha)/np.sqrt(2))
    t3prime.append(radius(alpha,a_z)*np.cos(alpha))
    
# Plot all data
fig, ax = plt.subplots(1,1,figsize=(5,4))
plt.rcParams['font.size'] = 12
ax.plot(-np.array(tau_1)[1:164],-np.array(tau_3)[1:164],'.',label=r'Optimized Data')
ax.set_xlim([-1,0])
ax.set_ylim([-1,0])
ax.set_xlabel(r'$t_1^\prime = t_2^\prime$', fontsize=14)
ax.set_ylabel(r'$t_3^\prime$', fontsize=14)
ax.plot(-np.array(t1prime),-np.array(t3prime),'-', label=r'${\rm Eq. (25)}$')
ax.plot(-np.array(bow_1)[:992],-np.array(bow_3)[:992],'-', label=r'${\rm Eq. (7)}$')

# Borders of quantum states and entanglement
ax.plot(x_border, y_border,'k-')
ax.fill(np.append(x_border,[-1,-1,0]), np.append(y_border,[0,-1,-1]), 'k', alpha=1)
ax.fill(np.append(xent_border[:-1],[-1,-1,0]), np.append(yent_border[:-1],[0,-1,-1]), 'k', alpha=0.1)
ax.legend(loc=2,framealpha=1)
ax.set_xticklabels([r'$%.1f$'%i for i in ([i for i in np.linspace(-1,0,6)])])
ax.set_yticklabels([r'$%.1f$'%i for i in ([i for i in np.linspace(-1,0,6)])])
ax.text(-1,0.02,'$(a)$', fontsize=14)
plt.subplots_adjust(top=0.88,
bottom=0.125,
left=0.145,
right=0.9,
hspace=0.2,
wspace=0.2)