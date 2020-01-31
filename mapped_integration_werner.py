# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:33:37 2018

File to construct necessary conditions for steerability of the displaced Werner state, utilising deterministic maps on Alice's side.
This script performs the optimizations over u, v in Eqs. (20) and (21), which are used to create Fig. 2.

@author: Trav Baker, travis.baker@griffithuni.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar

#We define the integrand in Eq. (20) with respect to (z,\phi) coordinates.
def integrand(phi,z,u,a_z):
        v=np.arcsin(a_z/np.sin(u))
        return np.sqrt((1-z**2)*np.sin(phi)**2*(1/np.cos(u))**2 + \
                   (1-z**2)*np.cos(phi)**2*(1/np.cos(v))**2 + \
                   z**2*(1/(np.cos(u)*np.cos(v)))**2)

def objective(u,a_z): #v is defined by x (i.e. u) and a_z.
    I,err=dblquad(integrand,-1,1,lambda phi: 0,lambda phi:2*np.pi, args=(u,a_z))
    xi = 2*np.pi/I
    return -xi

# Next, perform the optimizations for fixed values of |a|.
npoints = 50 
a_vec = np.linspace(0,1,npoints)
u_optimal =[]
xi_optimal = []
for a_z in a_vec:
     res = minimize_scalar(objective, method='Bounded', bounds=(np.arcsin(a_z),np.pi/2), args=(a_z))
     xi_optimal.append(-res.fun)
     u_optimal.append(res.x)

#Plot, to compare the optimal values of u and v 
v_optimal = np.arcsin(a_vec/np.sin(u_optimal))
plt.plot(u_optimal, v_optimal)
plt.xlabel(r'$u$')
plt.ylabel(r'$v$')

#analytic curve from Eq. (22), entangled and non-quantum regions:
def analytic_curve(a):
    u=np.arcsin(np.sqrt(a))
    return np.cos(u)/(np.sqrt(1+np.tan(u)**2) + np.arcsinh(np.tan(u))/np.tan(u))

def entangled_curve(x):
    y=np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= 1/3:
            y[i] = (np.sqrt(1-2*x[i]-3*x[i]**2))
    return y

def pos_operator_curve(x):
    return 1 - x
            
y=np.linspace(0,1,1000)
fig, ax = plt.subplots(1,figsize=(5,4))
ax.plot(a_vec, xi_optimal, '.', label=r'Optimized')
ax.plot(y, analytic_curve(y), '-', label=r'${\rm Eq. (22)}$')
ax.plot(np.sqrt(1-2*np.linspace(0,1/2,1000)), np.linspace(0,1/2,1000), '-', label=r'${\rm Eq. (7)}$')
ax.fill_between(pos_operator_curve(y), y, 1, facecolor='k', alpha=1)
ax.fill_between(entangled_curve(y), y, 1, facecolor='k', alpha=0.2)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r'$a$', fontsize=14)
ax.set_ylabel(r'$\xi$', fontsize=14)
ax.set_xticklabels([r'$%.1f$'%i for i in ([i for i in np.linspace(0,1,6)])])
ax.set_yticklabels([r'$%.1f$'%i for i in ([i for i in np.linspace(0,1,6)])])

ax.legend(prop = {'weight':'light'}, facecolor = 'w', framealpha=1, frameon=True)
plt.subplots_adjust(top=0.88,
bottom=0.125,
left=0.145,
right=0.9,
hspace=0.2,
wspace=0.2)