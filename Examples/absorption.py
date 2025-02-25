import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

import sys
import multilayers

plt.rcParams.update({'lines.markersize':5,'font.size':40,'lines.linewidth':3,'legend.fontsize':40,'axes.linewidth':2})

labda0=400.0e-9 #nm
FWHM=50.0e-9 
#option='CW'
option='numerical'
steps = 5
ang = 0.25*np.pi
pol = 'p'

direct_save = os.path.dirname(__file__)+os.sep+'Figures'+os.sep

header = ['dAl','R','T','a']

# Al thickness
fn = 'BK7500mu_BSTrue_'+option
n=['Air','Al2O3','Al','BK7','Air'] # Materials
n_adjust = {'Air':1.0, 'Al2O3':1.7865, 'Al': 0.31448+1j*3.8005,'BK7': 1.4701}
#n=['Air','Al','BK7','Air'] # Materials

d=[4.0e-9,16.0e-9,0.0005] #Thickness of materials, except for the first and last one (they are infinite)
#k_off = ['BK7','BK7_Cauchy']
k_off = []
ml = multilayers.ML(n_list=n,d_list=d,labda=labda0,FWHM=FWHM,k_off=k_off,n_adjust=n_adjust,backscatter=True)
R = lambda dal: ml.run(d=[d[0],dal,d[-1]],option=option,steps=steps,ang=ang,pol=pol)[0:3] #option: 'numerical' wil integrate over steps and 'pulse' will integrate over a incoming pulse (scipy.integrate.quad)

# dAl
f,ax = plt.subplots(1,1,figsize=(14,14))

dal_vec = np.linspace(1.0e-9,50.0e-9,50)
R_vec = np.array([R(dal) for dal in dal_vec])

labels = ['R','T','a']

for i in range(0,len(R_vec[0])):
    if i==3:
        val = np.array([np.linalg.norm(x) for x in R_vec[:,i]])
        ax.plot(dal_vec*1.0e9,val**2,label=labels[i])
    else:
        ax.plot(dal_vec*1.0e9,R_vec[:,i],label=labels[i])

ax.set_title('R, T and a versus Al thickness'+'\n'+str(n_adjust)[1:-1]+'\n'+'Al2O3: 4nm, BK7 500mu, backscater: On')
ax.set_xlabel('Al thickness (nm)')
ax.legend(loc='best')   

if os.path.exists(direct_save)==False:
    os.makedirs(direct_save)
    print('Made: '+direct_save)

f.savefig(direct_save+fn+'.png',bbox_inches='tight')
print('Saved: '+ direct_save+fn+'.png')

out = np.transpose(np.array([dal_vec,R_vec[:,0],R_vec[:,1],R_vec[:,2]]))
#pd.DataFrame(out).to_csv(direct_save+fn+'.csv',header=header,index=False)

# Al thickness
fn = 'BK7500mu_BSTrue'
n=['Air','Al2O3','Al','BK7','Air'] # Materials
n_adjust = {'Air':1.0, 'Al2O3':1.7865, 'Al': 0.31448+1j*3.8005,'BK7': 1.4701}
#n=['Air','Al','BK7','Air'] # Materials

d=[4.0e-9,16.0e-9,0.0005] #Thickness of materials, except for the first and last one (they are infinite)
#k_off = ['BK7','BK7_Cauchy']
k_off = []
steps = 401
#ml = multilayers.ML(n_list=n,d_list=d,labda=labda0,FWHM=FWHM,k_off=k_off,n_adjust=n_adjust,backscatter=True)
ml = multilayers.ML(n_list=n,d_list=d,labda=labda0,FWHM=FWHM,k_off=k_off,backscatter=True)

R1 = ml.run(d=d,option=option,steps=steps,ang=ang,pol=pol,a=2) #option: 'numerical' wil integrate over steps and 'pulse' will integrate over a incoming pulse (scipy.integrate.quad)

# Al thickness
fn = 'BK7500mu_BSTrue'
n=['Air','Al2O3','Al','BK7','Air'] # Materials
n_adjust = {'Air':1.0, 'Al2O3':1.7865, 'Al': 0.31448+1j*3.8005,'BK7': 1.4701}
#n=['Air','Al','BK7','Air'] # Materials

d=[0.0e-9,20.0e-9,0.0005] #Thickness of materials, except for the first and last one (they are infinite)
#k_off = ['BK7','BK7_Cauchy']
k_off = []
steps = 401
ml = multilayers.ML(n_list=n,d_list=d,labda=labda0,FWHM=FWHM,k_off=k_off,n_adjust=n_adjust,backscatter=True)
R2 = ml.run(d=d,option=option,steps=steps,ang=ang,pol=pol) #option: 'numerical' wil integrate over steps and 'pulse' will integrate over a incoming pulse (scipy.integrate.quad)

z_vec = np.linspace(0.0e-9,30.0e-9,101)
dz = z_vec[1]-z_vec[0]
abs_val1 = np.array([R1[-1](z) for z in z_vec])
abs_val2 = np.array([R2[-1](z) for z in z_vec])

print(str((((np.nansum(abs_val1)*dz)-R1[2])/R1[2])*100)+' % off')
print(str((((np.nansum(abs_val2)*dz)-R2[2])/R2[2])*100)+' % off')

f,ax = plt.subplots(1,1,figsize=(14,14))
ax.plot(z_vec*1.0e9,abs_val1*1.0e-9)
ax.plot(z_vec*1.0e9,abs_val2*1.0e-9)

ax.set_title('Absorption profile')
ax.set_xlabel('Depth (nm)')
ax.set_ylabel('Absorption (nm-1)')
ax.yaxis.set_major_locator(MultipleLocator(0.002))

fn = 'Example_absoption_profile_'+option
f.savefig(direct_save+fn+'.png',bbox_inches='tight')
print('Saved: '+ direct_save+fn+'.png')
