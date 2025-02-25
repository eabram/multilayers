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
FWHM=20.0e-9 
option='CW'
#option='numerical'

direct_save = os.path.dirname(__file__)+os.sep+'Figures'+os.sep

header = ['dAl','R','T','a']

# Si thickness
fn = 'SiN_'+option
n=['Air','Si3N4','Si_oxide - NTVE_JAW','Si_JAW','Air'] # Materials

d=[20.0e-9,4.0e-9,2000.0e-9] #Thickness of materials, except for the first and last one (they are infinite)
#k_off = ['BK7','BK7_Cauchy']
k_off = []
steps = 51
ang = 0.25*np.pi
pol = 'p'
ml = multilayers.ML(n_list=n,d_list=d,labda=labda0,FWHM=FWHM,k_off=k_off,backscatter=False)
R = lambda dsin: ml.run(d=[dsin,d[1],d[2]],option=option,steps=steps,ang=ang,pol=pol) #option: 'numerical' wil integrate over steps and 'pulse' will integrate over a incoming pulse (scipy.integrate.quad)

# dSi
f,ax = plt.subplots(1,1,figsize=(14,14))

dsin_vec = np.linspace(1.0e-9,50.0e-9,50)
R_vec = np.array([R(dsin)[0:3] for dsin in dsin_vec])

labels = ['R','T','a']

for i in range(0,len(labels)):
    if i==3:
        val = np.array([np.linalg.norm(x) for x in R_vec[:,i]])
        ax.plot(dsin_vec*1.0e9,val**2,label=labels[i])
    else:
        ax.plot(dsin_vec*1.0e9,R_vec[:,i],label=labels[i])

ax.set_title('R, T and a versus SiN thickness') #...adjust +'\n'+str(n_adjust)[1:-1]+'\n'+'Al2O3: 4nm, BK7 500mu, backscater: On')
ax.set_xlabel('SiN thickness (nm)')
ax.legend(loc='best')   

if os.path.exists(direct_save)==False:
    os.makedirs(direct_save)
    print('Made: '+direct_save)

f.savefig(direct_save+fn+'.png',bbox_inches='tight')
print('Saved: '+ direct_save+fn+'.png')

out = np.transpose(np.array([dsin_vec,R_vec[:,0],R_vec[:,1],R_vec[:,2]]))

#pd.DataFrame(out).to_csv(direct_save+fn+'.csv',header=header,index=False)
DSI = np.linspace(1.0e-9,100e-9,11)
R_abscalc = []
for dsi in DSI:
    R_abscalc.append(R(dsi))

z_vec = np.linspace(0.0e-9,200.0e-9,101)
dz = z_vec[1]-z_vec[0]
abs_val = []
for i in range(0,len(DSI)):
    abs_val.append(np.array([R_abscalc[i][-1](z) for z in z_vec]))
    print(str((((np.nansum(abs_val[-1])*dz)-R_abscalc[i][2])/R_abscalc[i][2])*100)+' % off')

f,ax = plt.subplots(1,1,figsize=(14,14))
for i in range(0,len(R_abscalc)):
    ax.plot(z_vec*1.0e9,abs_val[i]*1.0e-9,label=str(DSI[i]*1.0e9))

ax.set_title('Absorption profile')
ax.set_xlabel('Depth (nm)')
ax.set_ylabel('Absorption (nm-1)')
#ax.yaxis.set_major_locator(MultipleLocator(0.002))

fn = 'Example_absoption_profile_'+option
f.savefig(direct_save+fn+'.png',bbox_inches='tight')
print('Saved: '+ direct_save+fn+'.png')

f,ax = plt.subplots(1,1,figsize=(14,14))
ax.plot(DSI*1.0e9,np.nanmax(abs_val,axis=1))
ax.set_title('Peak absorption')
ax.set_xlabel('SiN thickness (nm)')
ax.set_ylabel('Absorption (nm-1)')

fn = 'Max_absorption'
f.savefig(direct_save+fn+'.png',bbox_inches='tight')
print('Saved: '+ direct_save+fn+'.png')

