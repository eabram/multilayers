import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy.integrate
import pandas as pd
import scipy.interpolate
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("default") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses

import warnings

#warnings.filterwarnings('error')
#
#try:
#    warnings.warn(Warning())
#except Warning:
#    print('Warning was raised as an exception!')



direct_REF_INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))+os.sep+'Refractive_Index'+os.sep

class ML():
    def __init__(self,n_list=[],d_list=[],labda=800.0e-9,FWHM=50.0e-9,fill_value=np.nan,direct_n=direct_REF_INDEX,k_off=[],pos_sub='Default',n_adjust={},backscatter=False):
        """Initializes the ML object"""
        self.labda0 = labda # Set central wavelength
        
        # n and k-values
        self.fill_value = fill_value # Set the values to fill it with ousite the interpolation limits
        self.n_dict = self.get_n(direct=direct_n,k_off=k_off) #Gets all nk values as function of wavelength
        self.n = self.get_n_value(n_list,n_adjust) #Selects the n-values per layers from self.n_dict or imports n_adjust directly
        self.mat = n_list #Material names for each layer
        

        self.d = d_list # Thicknesses for the layers. Omitten first and second infinite layers
        self.FWHM = FWHM # FWHM of spectrum, not important when option='CW'
        self.backscatter=backscatter # If True, backscatter from the substrate is considered, otherwise not

        #Position of the substrate
        if type(pos_sub)!=str:
            self.pos_sub = pos_sub
        else:
            self.pos_sub = len(n_list)-2
        print('Substrate is: '+n_list[self.pos_sub])
        if self.backscatter==False:
            print('Will not consider backscattering from substrate')
        
        print('Initialization Done')

### Refractive index ###

    def get_n(self,direct=direct_REF_INDEX,sel='model',k_off=[]) -> dict:
        """Reads all refractive index values from direct and interpolates those values over the wavelength

        self.interp_n interpolates the nk-values over wavelength"""
        
        print(direct)
        out = {}
        fn_all = []
        for (dp, dn, fn) in os.walk(direct):
            for f in fn:
                fn_all.append(dp+os.sep+f)
                opt = dp.split(direct)[-1]
                if fn_all[-1].split(os.sep)[-1].split('.')[0] in k_off:
                    k_off_sel = True
                else:
                    k_off_sel = False
                n_func, labda, n, k = self.interp_n(fn_all[-1],opt=opt,sel=sel,k_off=k_off_sel)
                out[fn_all[-1].split(os.sep)[-1].split('.')[0]] = n_func

        out['Vacuum'] = lambda labda: 1.0

        return out # Dictionary with all materials as keys holding the interpolated n,k values
    
    def interp_n(self,fn,opt='Site',sel='model',k_off=False):
        """Interpolates the nk values over wavelength"""

        if opt=='Site':
            A = pd.read_csv(fn,sep=';')
            keys = list(A.keys())
            if len(keys)==1:
                A = pd.read_csv(fn,sep=',')
                keys = list(A.keys())
            if len(keys)==1:
                A = pd.read_csv(fn,sep='\t')
                keys = list(A.keys())

            out = {}
            for i in range(0,len(keys)):
                if 'Wavelength' in keys[i]:
                    if 'µm' in keys[i] or 'μm' in keys[i]:
                        scale = 1.0e-6
                    if 'nm' in keys[i]:
                        scale = 1.0e-9
                    labda = np.array(A[keys[i]])*scale
                else:
                    out[keys[i]] = A[keys[i]]
            n0 = np.array(out['n'])
            n = np.where(np.isnan(n0),0,n0)
            k0 = np.array(out['k'])
            k = np.where(np.isnan(k0),0,k0)
        
        elif opt=='Ellipsometry':
            A = pd.read_csv(fn,sep=';')
            keys = list(A.keys())
            if len(keys)==1:
                A = pd.read_csv(fn,sep=',')
                keys = list(A.keys())
            if len(keys)==1:
                A = pd.read_csv(fn,sep='\t',index_col=False)
                keys = list(A.keys())
            
            out = {}
            for i in range(0,len(keys)):
                if 'Wavelength' in keys[i]:
                    if 'µm' in keys[i]:
                        scale = 1.0e-6
                    if 'nm' in keys[i]:
                        scale = 1.0e-9
                    labda = A[keys[i]]*scale

                else:
                    out[keys[i]] = A[keys[i]]
            if sel=='model':
                try:
                    n = out['n (model)']
                    k = out['k (model)']
                except KeyError:
                    n = out['n']
                    k = out['k']
                    #print('No model data available, used measured data insted.')
            elif sel=='data':
                n = out['n']
                k = out['k']
        
        labdan = np.where(np.isnan(n),np.nan,labda)
        if self.fill_value=='last':
            fill_value = [x for x in n if np.isnan(x) == False][-1]
        else:
            fill_value = self.fill_value

        f_n = scipy.interpolate.interp1d(labdan,n,fill_value=fill_value)

        if self.fill_value=='last':
            fill_value = [x for x in k if np.isnan(x) == False][-1]
        else:
            fill_value = self.fill_value

        labdak = np.where(np.isnan(k),np.nan,labda)
        f_k = scipy.interpolate.interp1d(labdak,k,fill_value=fill_value)
        
        if k_off==False:
            n_func = lambda l: f_n(l)+1j*f_k(l)
        elif k_off==True:
            n_func = lambda l: f_n(l)
            print('Set '+fn.split(os.sep)[-1].split('.')[0]+' k-values to zero')
            print()
        
        return n_func, labda, n, k

    def get_n_value(self,mat: list,n_adjust: dict) -> list:
        """Selects the proper n_values from n_adjust or from self.n_dict otherwise"""
        ret = []
        for m in mat:
            if m in n_adjust:
                ret.append(n_adjust[m])
            else:
                ret.append(self.n_dict[m])

        return ret

    def get_n_val(self,labda='Default',n='Default') -> list:
        """Calculates proper n_values per layer for lambda=labda"""
        if n=='Default':
            n = self.n

        if labda=='Default':
            labda = self.labda0
        
        n_out = []
        for i in range(0,len(n)):
            try:
                nval = n[i](labda)
            except:
                nval = n[i]
            n_out.append(nval)

        return n_out

### Multilayer calculation
    def get_Mij(self,n,d,j,labda='Default',pol='p',ang=0.0):
        '''Calculate Mij and Mj matrices for all layers'''

        if labda=='Default':
            labda = self.labda0
        
        k0=(2.0*np.pi)/labda
        n_val = self.get_n_val(labda=labda,n=n)
        n0 = n_val[0]
        n1 = n_val[j-1]
        n2 = n_val[j]
        
        realn1 = np.real(n1)
        realn2 = np.real(n2)
        imagn1 = np.imag(n1)
        imagn2 = np.imag(n2)

        sinth1=(np.sin(ang)*np.real(n0))/realn1
        sinth2=(np.sin(ang)*np.real(n0))/realn2
        n1par=(np.sin(ang)*np.real(n0))
        n2par=(np.sin(ang)*np.real(n0))
        b1 = realn1**2.0-imagn1**2.0-n1par**2.0
        b2 = realn2**2.0-imagn2**2.0-n2par**2.0
        n1perp = ((b1+((b1**2.0+(4.0*(realn1**2.0)*(imagn1**2.0)))**0.5))/2.0)**0.5
        n2perp = ((b2+((b2**2.0+(4.0*(realn2**2.0)*(imagn2**2.0)))**0.5))/2.0)**0.5
        k1perp = ((-b1+((b1**2.0+(4.0*(realn1**2.0)*(imagn1**2.0)))**0.5))/2.0)**0.5
        k2perp = ((-b2+((b2**2.0+(4.0*(realn2**2.0)*(imagn2**2.0)))**0.5))/2.0)**0.5

        if imagn1==0:
            N1 = realn1/n1perp
        else:
            N1 = (1.0+((n1par**2.0)/((b1**2.0+4.0*(realn1**2.0)*(imagn1**2.0))**0.5)))
        if imagn2==0:
            N2 = realn2/n2perp
        else:
            N2 = (1.0+((n2par**2.0)/((b2**2.0+4.0*(realn2**2.0)*(imagn2**2.0))**0.5)))
        
        adjust=False
        if imagn2==0 or realn2==0:
            adjust = True
            #print(N1/N2,realn2,imagn2,n2)

        n1perptilde = n1perp+1j*k1perp
        n2perptilde = n2perp+1j*k2perp
        if pol=='s':
            alpha=1.0
            beta=n2perptilde/n1perptilde
        elif pol=='p':
            alpha = N1/N2
            beta = (n2**2.0)/(n1**2.0)*(n1perptilde/n2perptilde)*(N1/N2)
        r = (alpha-beta)/(alpha+beta)
        t = 2.0/(alpha+beta)
        
        Mjm1j=np.matrix([[(alpha+beta)/2.0, (alpha-beta)/2.0],[(alpha-beta)/2.0, (alpha+beta)/2.0]])

        if j!=len(n)-1:
            dj = d[j-1]
            kd = k0*n2perptilde
            if (np.real(kd)*dj)>100:
                dj = 100 / np.real(kd)
                print(dj)
                print('Found thick absorbing layer')
                self.d[j-1] = dj
                print('Set thick absorbing layer to: '+str(dj))

            if j==self.pos_sub and self.backscatter==False:
                Mj = np.array([[np.exp(-1j*kd*dj),0],[0,0]])
                Mjz = lambda z: np.array([[np.exp(-1j*kd*z),0],[0,0]])
            else:
                Mj = np.array([[np.exp(-1j*kd*dj),0],[0,np.exp(1j*kd*dj)]])
                Mjz = lambda z: np.array([[np.exp(-1j*kd*z),0],[0,np.exp(1j*kd*z)]])
            
        else:
            Mj = np.array([[1.0,0],[0,1.0]])
            Mjz = lambda z: np.array([[1.0,0],[0,1.0]])
            phi = 0.0
            kd = k0
        #if np.abs(Mj[0][0])==np.inf:
        #    Mj[0][0] = (1+1j)*10e100 # Put in high number to work with
        
        return Mjm1j, Mj, Mjm1j @ Mj, Mjz
     
    def calc_multilayer(self,labda,n='Default',d='Default',pol='p',ang=0.0):
        '''Caluclates R and T for multilayer system at wavelength labda'''

        if n=='Default':
            n = self.n
        if d=='Default':
            d = self.d
        
        M = np.array([[1.0,0.0],[0.0,1.0]])
        M_parts = [np.matrix(M)]
        M_all = []
        for j in range(1,len(n)):
            M_new = self.get_Mij(n,d,j,labda=labda,pol=pol,ang=ang)
            M_all.append(M_new)
            M_parts.append(M_new[2])
            M = M @ M_new[2]
        [Ein1,Eout1] = np.array(M @ np.array([1.0,0.0]))[0]
        
        R = np.abs((Eout1/Ein1))**2
        r = (Eout1/Ein1)
        try:
            nrealstart = np.real(n[0](labda))
            nrealend = np.real(n[-1](labda))
        except TypeError:
            nrealstart = np.real(n[0])
            nrealend = np.real(n[-1])

        T = (np.abs((1.0/Ein1))**2)*(nrealend/nrealstart)
        t = (1.0/Ein1)*(nrealend/nrealstart) #...check if this is correct
        ABS = 1.0-T-R
        

        return R, T, ABS, r, t, M_parts, M_all

### Run calculation ###
    def run(self,n='Default',d='Default',labda0='Default',FWHM='Default',option='numerical',steps=401,ang=0.0,pol='p',a=3):
        ''' Runs the calculation to obtain R, T, total basorbed values and absoption profile 
        
        option = CW # Calculate for labda0 only
        option = numerical # Samples over spectrum with central wavelength labda0 and FWHM from -a*FWHM to a*FWHM around labda0. Note that when fill_value is for instance set to nan, going outsited te interpolation limits will raise an error. The spectrum is sampled over steps number of steps.
        option = pulse # Does the same as numerical but then integrating it instead of numerically solving it #...does not work with newest version
        '''
        self.get_pulse(labda0=labda0,FWHM=FWHM)
        if labda0=='Default':
            labda0 = self.labda0
        if FWHM=='Default':
            FWHM = self.FWHM
        
        if option=='pulse':
            R_func = lambda labda: self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)[0]*self.pulse(labda)
            T_func = lambda labda: self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)[1]*self.pulse(labda)
            r_func = lambda labda: self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)[3]*self.pulse(labda)

            start = labda0 - 3*FWHM
            end = labda0 + 3*FWHM
            R, Rerr = scipy.integrate.quad(R_func,start,end)
            T, Terr = scipy.integrate.quad(T_func,start,end)
            r, rerr = scipy.integrate.quad(r_func,start,end)
            ABS = 1.0-T-R

        elif option=='numerical':
            labda_vec = np.linspace(labda0-a*FWHM,labda0+a*FWHM,steps)
            dist = np.array([self.pulse(labda) for labda in labda_vec])
            val = np.array([np.array(self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)) for labda in labda_vec])

            R_val = val[:,0]*dist
            T_val = val[:,1]*dist
            ABS_val = val[:,2]*dist
            r_val = val[:,3]*dist
            t_val = val[:,4]*dist
            M_parts_val = val[:,5]
            Mz_val = val[:,6]
            
            Dlabda = labda_vec[1]-labda_vec[0]
            R = np.nansum(R_val)*Dlabda
            T = np.nansum(T_val)*Dlabda
            r = np.nansum(r_val)*Dlabda
            t = np.nansum(t_val)*Dlabda
            M_parts = M_parts_val
            Mz = Mz_val
            
            abs_func = lambda z: self.get_multiple_abs_wf(z,Mz_val,T_val/dist,d,n,labda_vec)
            
            Rerr = np.nan
            Terr = np.nan
            rerr = np.nan

            ABS = 1.0-T-R

        elif option=='CW':
            R, T, ABS,r ,t, M_parts, Mz = self.calc_multilayer(labda0,n=n,d=d,ang=ang,pol=pol)
            abs_func = lambda z: self.get_absorption(z,Mz,T,d,self.n,labda0)

            Rerr = 0.0
            Terr = 0.0
            rerr = 0.0

        return R, T, ABS, r, Rerr, Terr, rerr, t, M_parts, Mz, abs_func

    def get_pulse(self,labda0='Default',FWHM='Default'):
        '''Distributio  over the spectrum'''

        if labda0=='Default':
            labda0 = self.labda0
        if FWHM=='Default':
            FWHM = self.FWHM
        
        sig = FWHM/(2.0*((2.0*np.log(2))**0.5))

        pulse = lambda labda: ((1.0/(sig*((2.0*np.pi)**0.5)))*np.exp(-0.5*((labda-labda0)**2)/(sig**2)))

        self.pulse = pulse

        return 0
    
#--- ABSORPTION --- #..maybe only for normal incidence
    def get_z_list(self,d_list,d):
        ''' Get z-values for each layer seen from the transmission side as function of position d into the material'''
        d_list = np.array(d_list,dtype='float64')
        z_list = np.nan*d_list
        d_tot = np.nansum(d_list)
        d_new = d_tot-d

        if d<0:
            z_list = d_list
        elif d>d_tot:
            z_list = 0*d_list
        else:
            for i in range(len(z_list)-1,-1,-1):
                if d_new<=np.nansum(d_list[i:]):
                    z_list[i] = d_new - np.nansum(z_list)
                    #d_new = 0.0
                else:
                    z_list[i] = d_list[i]

        return z_list

    def get_abs_function(self,M_all,z,labda,T,n_list):
        '''Get the absorbtion as function fo z

        Note: This may only hold for normal incidence. Source: Klaasjan'''

        n_val = self.get_n_val(labda=labda,n=n_list)
        M_tot = np.array([[1.0,0],[0,1.0]])
        j = len(z)-1
        while j>=0:
            if z[j]!=0:
                M_tot = M_all[j][3](z[j]) @ M_all[j+1][0] @ M_tot
            j = j-1
        av,ab = np.array(M_tot @ np.array([1.0,0]))[0]
        f = av*np.conjugate(av) + ab*np.conjugate(ab) + av*np.conjugate(ab) +ab*np.conjugate(av)
        k0 = (2.0*np.pi)/labda

        try:
            j = list(z>0.0).index(True)
            absorbed = np.real((2.0*k0*np.real(n_val[j+1])*np.imag(n_val[j+1])*f)*(T/n_val[-1]))
        except ValueError:
            absorbed = 0.0

        return absorbed


    def get_absorption(self,d,M_all,T,d_list,n_list,labda):
        '''Gets the absorption function when option=CW'''
        z = self.get_z_list(d_list,d)
        if d>=np.nansum(d_list):
            absorption = 0
        elif d<0:
            absorption = 0
        else:
            absorption = self.get_abs_function(M_all,z,labda,T,n_list)

        return absorption

    def get_multiple_abs_wf(self,z,Mz_val,T_val,d,n,labda_vec):
        '''Gets the absorption function when option=numerical or pulse'''
        val = np.array([self.get_absorption(z,Mz_val[i],T_val[i],d,n,labda_vec[i])*self.pulse(labda_vec[i])*(labda_vec[1]-labda_vec[0]) for i in range(0,len(labda_vec))])
        
        return np.nansum(val)
