#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:43:08 2019

File to construct displaced T-states (t_1,t_2, t_3, a_z all nonzero) with LHS models, considering deterministic maps on Alice's side.
In this script Alice's (steering party) Bloch vector is in the z-direction.
This script performs the second optimization over u, v in Eq. (24), which are used to create Fig. 3(b).

@author: Trav Baker, travis.baker@griffithuni.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import qutip as qt
import time

# First, pick the size of Alice's Bloch vector.
a_z = 0.2 #=sin(u)*sin(v) in terms of map parameters.
alpha=np.arctan(1/2) #defined as the angle from the t1 axis. For the example in Fig. 3(b); t2=t1*tan(alpha), t3, a_z nonzero

def NS(phi,z,u,v,r,theta,alpha):
    n = np.array([(1-z**2)*np.cos(phi)**2,(1-z**2)*np.sin(phi)**2,z**2])
    Lambda = np.array([1/np.cos(u), 1/np.cos(v), 1/(np.cos(u)*np.cos(v))])
    tau_corr=np.array([r*np.sin(theta)*np.cos(phi),\
                      r*np.sin(theta)*np.sin(phi),\
                      r*np.cos(theta)])
    M = np.sum(tau_corr**2*Lambda**2*n)
    return np.sqrt(M)

def func(r,u,v,theta,alpha):
    I,err=dblquad(NS,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(u,v,r,theta,alpha))
    return 2*np.pi - I
        
def integrand(phi,z,u,theta,alpha):
        v=np.arcsin(a_z/np.sin(u))
        return np.sqrt((1-z**2)*np.sin(phi)**2*(np.sin(theta)*np.cos(alpha)/np.cos(u))**2 + \
                   (1-z**2)*np.cos(phi)**2*(np.sin(theta)*np.sin(alpha)/np.cos(v))**2 + \
                   z**2*(np.cos(theta)/(np.cos(u)*np.cos(v)))**2)

def objective(u,theta,alpha): #v is defined by x (i.e. u) and a_z.
    I,err=dblquad(integrand,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(u,theta,alpha))
    r = 2*np.pi/I
    return -r

def bowles_objective(x,theta,alpha,a_z,tau_3,sign=-1):
    return sign*((a_z*x[0])**2 + 2*np.abs(tau_3)*np.sqrt(np.tan(theta)**2*(1-x[0]**2)*(np.cos(alpha)**2*np.cos(x[1])**2 + np.sin(alpha)**2*np.sin(x[1])**2) + x[0]**2))

def bowles_t3_root(tau_3,theta,alpha,a_z):
    res = minimize(bowles_objective, [1,0], method='SLSQP', bounds=((-1,1),(0,2*np.pi)), args=(theta,alpha,a_z,tau_3))
    return -res.fun - 1
#starting points above can be important

#compare with result derived by Bowles et. al. in PRA, also
bow_1=[]
bow_3=[]
start=time.time()

#for theta in np.linspace(0,np.pi/2,50)[1:-1]:
#    bow_res,r = brentq(bowles_t3_root,0,4,args=(theta,alpha,a_z),xtol=1e-12, maxiter=100, full_output=True)
#    if r.converged == True:
#        bow_3.append(bow_res*np.cos(theta))
#        bow_1.append(bow_res*np.sin(theta))
#    print('Finished u-LHS %.4f percent' %(theta*200/np.pi))

nbow=100
for theta in np.linspace(0,np.pi/2,nbow)[1:-1]:
    bow_res,out = brentq(bowles_t3_root,0,1,args=(theta,alpha,a_z),xtol=1e-12, maxiter=100, full_output=True)
    if out.converged == True:
        bow_3.append(bow_res)
#        r=bow_res/np.cos(theta)
#        x=r*np.sin(theta)*np.cos(alpha)
        bow_1.append(bow_res*np.cos(alpha)*np.tan(theta))
    print('Finished u-LHS %.4f' %(2*theta/np.pi*nbow))
    
ntheta = 200
theta_vec = np.linspace(0,np.pi/2,ntheta)
tau_1=[]
tau_3=[]
tau_1.append(0)
tau_3.append(1-a_z)
u_opt=np.empty(ntheta-2)
for k, theta in enumerate(theta_vec[1:-1]):
    res = minimize_scalar(objective, method='Bounded', bounds=(np.arcsin(a_z),np.pi/2), args=(theta,alpha))
    I, err = dblquad(integrand,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(res.x,theta,alpha))
    u_opt[k]=res.x
    tau_3.append(np.cos(theta)*2*np.pi/I)
    tau_1.append(np.sin(theta)*np.cos(alpha)*2*np.pi/I)
    print('Finished angle %s of %s' %(np.where(theta_vec == theta)[0][0],ntheta-2))

# =============================================================================
# Quantum state boundary.
# =============================================================================
def least_eig(r,theta,alpha,a_z):
    rho=(1/4)*(qt.tensor(qt.qeye(2), qt.qeye(2)) + qt.tensor(a_z*qt.sigmaz(), qt.qeye(2)) - r*np.sin(theta)*np.cos(alpha)*qt.tensor(qt.sigmax(),qt.sigmax()) - r*np.sin(theta)*np.sin(alpha)*qt.tensor(qt.sigmay(),qt.sigmay()) - r*np.cos(theta)*qt.tensor(qt.sigmaz(),qt.sigmaz()))
    eigs = np.linalg.eigvalsh(rho.full())
    return min(eigs)

npoints=100
qangle=np.linspace(0,np.pi/2,npoints) #angle from -y axis
x_border=[]
y_border=[]
for i in qangle:
    root=brentq(least_eig,0,2,args=(i,alpha,a_z))
    x_border.append(-root*np.sin(i)*np.cos(alpha))
    y_border.append(-root*np.cos(i))
    print(r'%.2f%% of Q-boundary points' %(2*i/np.pi*npoints))
    
# =============================================================================
# Now entangled states
# =============================================================================

def least_ent_eig(r,theta,alpha,a_z):
    rho=(1/4)*(qt.tensor(qt.qeye(2), qt.qeye(2)) + qt.tensor(a_z*qt.sigmaz(), qt.qeye(2)) - r*np.sin(theta)*np.cos(alpha)*qt.tensor(qt.sigmax(),qt.sigmax()) - r*np.sin(theta)*np.sin(alpha)*qt.tensor(qt.sigmay(),qt.sigmay()) - r*np.cos(theta)*qt.tensor(qt.sigmaz(),qt.sigmaz()))
#    print(rho)
    rho_pt = qt.partial_transpose(rho, [0,1])
    eigs = np.linalg.eigvalsh(rho_pt.full())
    return min(eigs)

npoints=100
qangle=np.linspace(0,np.pi/2,npoints) #angle from -y axis
xent_border=[]
yent_border=[]
for i in qangle:
    root=brentq(least_ent_eig,0,2,args=(i,alpha,a_z))
    xent_border.append(-root*np.sin(i)*np.cos(alpha))
    yent_border.append(-root*np.cos(i))
    print(r'%.2f%% of Entangled-boundary points' %(2*i/np.pi*npoints))

end=time.time()
minutes=(end-start)/60
print('%f minutes elapsed' %minutes)
bow_1[-1]=bow_1[-2]
bow_3[-1]=0

np.savez('a_z%s_3DT_%sintegrals_data.npz' %(a_z,ntheta), tau_1, tau_3, bow_1, bow_3, x_border, y_border, xent_border, yent_border)

# Plot results to compare.
data = np.load('a_z%s_3DT_%sintegrals_data.npz' %(a_z,ntheta))
tau_1, tau_3, bow_1, bow_3, x_border, y_border, xent_border, yent_border = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6'], data['arr_7']

plt.rcParams['axes.labelsize'] = 12
fig, ax = plt.subplots(1,1,figsize=(5,4))
tau_chop=182
ax.plot(-np.array(tau_1)[:tau_chop],-np.array(tau_3)[:tau_chop],'-',label=r'${\rm Eq. (24)}$')
ax.set_xlim([-1,0])
ax.set_ylim([-1,0])
ax.set_xlabel(r'$t_1^\prime = 2t_2^\prime$', fontsize=14)
ax.set_ylabel(r'$t_3^\prime$', fontsize=14)
ax.plot(-np.array(bow_1),-np.array(bow_3),'-', label=r'${\rm Eq. (7)}$')
ax.plot(x_border, y_border,'k-')
ax.fill(np.append(x_border,[-1,-1,0]), np.append(y_border,[0,-1,0-1]), 'k', alpha=1)
ax.fill(np.append(xent_border[:-1],[-1,-1,0]), np.append(yent_border[:-1],[0,-1,-1]), 'k', alpha=0.1)
ax.legend(loc=3,framealpha=1)
ax.set_xticklabels([r'$%.1f$'%i for i in ([i for i in np.linspace(-1,0,6)])])
ax.set_yticklabels([r'$%.1f$'%i for i in ([i for i in np.linspace(-1,0,6)])])
ax.text(-1,0.02,'$(b)$', fontsize=14)
plt.subplots_adjust(top=0.88,
bottom=0.13,
left=0.15,
right=0.9,
hspace=0.2,
wspace=0.2)