import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy.integrate
import pandas as pd
import scipy.interpolate
from typing import Union
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("default") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses

import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning) # Ignores the np.matrix use warning
#warnings.filterwarnings('error')
#
#try:
#    warnings.warn(Warning())
#except Warning:
#    print('Warning was raised as an exception!')

print()
print()
print()

direct_REF_INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))+os.sep+'Refractive_Index'+os.sep

class ML():
    """Class to calculate the reflection, transmission and absorption (profiles) of a light beam through a flat multilayer stack

    Attributes:
        lambda0: central wavelength
        n_dict: Gets all n and k values as a function of wavelength
        n: Selects the n-values per layer from n_dict or imports from n_adjust
        mat: Material name/handle for each layer
        d: List of thichnesses for each layer (except for the first and last which are set to infinity)
        FWHM: The FWHM of the spectrum of the incloming light. This is only included when not CW light is considered
        pos_sub: the position of the substrate
        backscatter: If set to False, the backscattering of the the substrate layer (position pos_sub) is swichted off
        absorb0 (list): list containing information to handle print statments
        fill_value (float): Uses this value for the n+1jk value for lambda outside the interpolation range
    """


    def __init__(self,n_list:list=[],d_list:list=[],labda:float=800.0e-9,FWHM:float=50.0e-9,fill_value:float=np.nan,direct_n:str=direct_REF_INDEX,k_off:list=[],pos_sub:Union[str,int]='Default',n_adjust:dict={},backscatter:bool=False):
        """Initializes the ML object

        Args:
            n_list (list): List of material names in order from top (incoming light side) to bottom
            d_list (list): List of material thicknesses order from top (incoming light side) to bottom. The first and last material thicknesses are omitted because they are considered infinite (len(d_list) = len(n_list)-2)
            labda (float): (Central) wavelength
            FWHM (float): FWHM of the incoming light (omitted when the light is CW)
            fill_value (float): ...to adjust
            direct_n (str): Directory of the refractive-index library
            k_off (list): List of layer indices where the absorption is set off (k set to 0). This is not necessacry of the read in k-value is already zero, however numerically helpfull if k is very small and the thickness large.
            pos_sub (str/int): Layer index of the substrate material. This is only needed if the backscattering from the backside of the substrate has to be set to 0.
            n_adjust (dict): When using own specified refractive index values, one can inlude them in this dictionary. The key is the material name, and the value eather a function (vs. wavelength in meters), or a constact.
            backscatter (bool): If False, set the backscattering from the substrate backsurface off. This can be helpfull wheb considering an unpolished backsurface of the substrate.

        """

        self.labda0 = labda # Set central wavelength
        
        # n and k-values
        self.fill_value = fill_value # Set the values to fill it with ousite the interpolation limits
        self.n_dict = self.get_n(direct=direct_n,k_off=k_off) #Gets all nk values as function of wavelength
        self.n = self.get_n_value(n_list,n_adjust) #Selects the n-values per layers from self.n_dict or imports n_adjust directly
        self.mat = n_list #Material names for each layer
        

        self.d = d_list # Thicknesses for the layers. Omitten first and second infinite layers
        self.FWHM = FWHM # FWHM of spectrum, not important when option='CW'
        self.backscatter=backscatter # If True, backscatter from the substrate is considered, otherwise not
        self.absorb0 = []

        #Position of the substrate
        if type(pos_sub)!=str:
            self.pos_sub = pos_sub
        else:
            self.pos_sub = len(n_list)-2
        print('Substrate is: '+n_list[self.pos_sub])
        if self.backscatter==False:
            print('Will not consider backscattering from substrate')
        
        print('Initialization Done')
        print()

### Refractive index ###

    def get_n(self,direct:str=direct_REF_INDEX,sel:str='model',k_off:list=[]) -> dict:
        """Reads all refractive index values from direct and interpolates those values over the wavelength

        Args:
            direct (str): Directory of the refractive-index library
            sel (str): ...to do
            k_off (list): List of layer indices where the absorption is set off (k set to 0). This is not necessacry of the read in k-value is already zero, however numerically helpfull if k is very small and the thickness large.

        Returns:
            out (dict): # Dictionary with all materials as keys holding the interpolated n,k values

        """
        
        print(direct)
        out = {}
        fn_all = []
        for (dp, dn, fn) in os.walk(direct): # Loop over the refrective index library in direct
            for f in fn:
                ext = f.split('.')[-1].lower()
                if ext=='txt' or ext=='csv':
                    fn_all.append(dp+os.sep+f)
                    opt = dp.split(direct)[-1]
                    if fn_all[-1].split(os.sep)[-1].split('.')[0] in k_off:
                        k_off_sel = True #Set absorption off (necessary for numerical accuracy later)
                    else:
                        k_off_sel = False #Set absorption on
                    n_func, labda, n, k = self.interp_n(fn_all[-1],opt=opt,sel=sel,k_off=k_off_sel) # Interpolates the listed n and k values into a function
                    out[fn_all[-1].split(os.sep)[-1].split('.')[0]] = n_func # Saves the interpolated function in the directory with the material name as key.

        out['Vacuum'] = lambda labda: 1.0 # refractiv index for vacuum

        return out # Dictionary with all materials as keys holding the interpolated n,k values
    
    def interp_n(self,fn:str,opt:str='Site',sel:str='model',k_off:bool=False) -> list:
        """Interpolates the nk values over wavelength

        Args:
            fn (str): Filename of the reported nk values
            opt (str): Eather 'Site' or 'Ellipsometry', which selects the subfolder of the nk-library. 'Site' is found at online (refractiveindex.org) and 'Ellipsometry' are nk-values obtained by ellipsometry. The deviation is made due to the diffrent structures of the generated nk-files.
            sel (str): If equal to 'model', a different column in the nk-file is selected.
            k_off (bool): If True sets the absorption to 0

        Returns:
            list: List containing:
                n_func (function): Interpolated function of n +1jk as function of lambda
                labda (list): Input wavelengt (in meter)
                n (list): Input real part of the refractive index
                k (list): Input imaginairy part of the refractive index

        """
        
        # Read files from nk-library
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
        
        # Obtain a n+1jk function versus wavelength
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
        """Selects the proper n_values from n_adjust (adjusted/self-defined nk-values) or from self.n_dict (nk-library) otherwise

        Args:
            mat (list): List of material names of the stack
            n_adjust (dict): Directlory with material names as keys and a nk-function or value as value. This is to use values different from the one stored in the nk-library

        Output:
            n_out(list): List of nk-fuctions (versus lambda in meter) in the order of mat

        """
        ret = []
        for m in mat:
            if m in n_adjust:
                ret.append(n_adjust[m])
            else:
                ret.append(self.n_dict[m])

        return ret

    def get_n_val(self,labda:Union[str,float]='Default',n:Union[str,dict]='Default') -> list:
        """Calculates proper n_values per layer for lambda=labda

        Args:
            lambda (str/float): Wavelength (in meter)
            n (str/dict): dictionary of n+1jK (refractive index function versus lambda) per key (material)

        Output:
            n_out (list): List of refractive index values at wavelength lambda

        """

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
    def get_Mij(self,n:Union[str,dict],d:list,j:int,labda:Union[str,float]='Default',pol:str='p',ang:float=0.0) -> tuple:
        '''Calculate Mij and Mj matrices for all layers

        Args:
            n (str/dict): Directory of refracive index functions, or 'Default'
            d (list): List of layer thicknesses (without the first and second infinite layers)
            j (int): Index of layer j
            labda (str/float): Wavelength in meters
            pol (str): Polarization, eather 'p' or 's'
            ang (float): Angle of incidence

        Output:
            tuple: Tuple of
                Mjm1j (np.matrix): Transfer-matrix between medium j-1 and j
                Mj (np.matrix): Propagation matrix through material j
                Mjm1j @ Mj (np.matrix): ...to do
                Mjz (np.matrix(z)): Propagation matrix versus thickness z

        '''
        if labda=='Default':
            labda = self.labda0
        
        k0=(2.0*np.pi)/labda
        n_val = self.get_n_val(labda=labda,n=n)
        n0 = n_val[0]
        n1 = n_val[j-1]
        n2 = n_val[j]

        try: 
            n0 = n0(labda)
        except:
            try:
                n1 = n1(labda)
            except:
                try:
                    n2 = n2(labda)
                except:
                    pass

        if 'function' in str(type(n2)):
            n2 = n2(labda)

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
        #Mjm1j=np.array([[(alpha+beta)/2.0, (alpha-beta)/2.0],[(alpha-beta)/2.0, (alpha+beta)/2.0]])

        if j!=len(n)-1:
            dj = d[j-1]
            kd = k0*n2perptilde
            if (np.real(kd)*dj)>100:
                dj = 100 / np.real(kd)
                self.d[j-1] = dj
                if j not in self.absorb0:
                    print(dj)
                    print('Found thick absorbing layer')
                    print('Set thick absorbing layer to: '+str(dj))
                    self.absorb0.append(j)

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
     
    def calc_multilayer(self,labda:float,n:Union[str,dict]='Default',d:Union[str,list]='Default',pol:str='p',ang:float=0.0) -> tuple:
        '''Caluclates R and T for multilayer system at wavelength labda

        Args:
            labda (float): Wavelength in meters
            n (str/dict): Directory of refracive index functions, or 'Default'
            d (list): List of layer thicknesses (without the first and second infinite layers)
            pol (str): Polarization, eather 'p' or 's'
            ang (float): Angle of incidence

        Output:
            tuple: Tuple containing:
                R (float): Intensity of reflection
                T (float): Intensity of transmission
                ABS (float): Intensity of absorption
                r (complex): Reflection (fresnel coefficient)
                t (complex): Transmission (fresnel coefficient)
                M_parts (list): List of all Mjm1j @ Mj matrixes
                M_all (list): List of all [Mjm1j, Mj, Mjm1j @ Mj, Mj]

        '''

        if n=='Default':
            n = self.n
        if d=='Default':
            d = self.d
        
        M = np.array([[1.0,0.0],[0.0,1.0]])
        M_parts = [np.matrix(M)]
        #M_parts = [np.array(M)]
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
        
        theta_start = ang
        theta_end = np.arcsin(np.sin(ang)*(nrealstart/nrealend))
        T = (np.abs((1.0/Ein1))**2)*(nrealend/nrealstart)*(np.cos(theta_end)/np.cos(theta_start))
        t = (1.0/Ein1)*(nrealend/nrealstart)*((np.cos(theta_end)/np.cos(theta_start))**0.5)
        ABS = 1.0-T-R
        

        return R, T, ABS, r, t, M_parts, M_all

### Run calculation ###
    
    def run_backup(self,n:Union[str,dict]='Default',d:Union[list,str]='Default',labda0:Union[str,float]='Default',FWHM:Union[str,float]='Default',option:str='numerical',steps:int=401,ang:float=0.0,pol:str='p',a:float=3) -> tuple:
        ''' Runs the calculation to obtain R, T, total basorbed values and absoption profile 
        
        Args:
            n (str/dict): Directory of refracive index functions, or 'Default' (=self.n)
            d (str/list): List of layer thicknesses (without the first and second infinite layers), if ='Default' then it is equal to self.d
            labda0 (str/float): Wavelength in meters, or 'Default' (=self.labda0)
            FWHM (str/float): FWHM of the beam in meters, or 'Default' (=self.FWHM)
            option (str): Selects beam type
                option = CW # Calculate for labda0 only
                option = numerical # Samples over spectrum with central wavelength labda0 and FWHM from -a*FWHM to a*FWHM around labda0. Note that when fill_value is for instance set to nan, going outsited te interpolation limits will raise an error. The spectrum is sampled over steps number of steps.
                option = pulse # Does the same as numerical but then integrating it instead of numerically solving it #...does not work with newest version
            steps (int): Number of samples between -a*FWHM+lambda and a*FWHM+labda (not workin when option = 'CW')
            ang (float): Angle of incidence
            pol (str): Polarization, eather 'p' or 's'
            a: Selts wavelength sample range (between -a*FWHM+lambda and a*FWHM+labda) (not workin when option = 'CW')

        Output:
            tuple: tuple containing:
                R (float): Intensity of reflection
                T (float): Intensity of transmission
                ABS (float): Intensity of absorption
                r (complex): Reflection (fresnel coefficient)
                Rerr (float): Error of R
                Terr (float): Error of T
                rerr (float): Error of r
                t (complex): Transmission (fresnel coefficient)
                M_parts (list): List of all Mjm1j @ Mj matrixes
                Mz :...to do
                abs_func (float function): Absorption profile versus thickens

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
            val_names = ['R','T','ABS','r','t','M_parts','Mz']
            for name in val_names:
                locals()[name+'_val'] = []

            for labda in labda_vec:
                val = self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)
                for i in range(0,len(val)):
                    locals()[val_names[i]+'_val'].append(val[i])
            
            for i in len(val_names):
                name = val_names[i]+'_val'
                if name[0]!=M:
                    val = locals()[val_names[i]+'_val']
                    locals()[val_names[i]+'_val'] = val*dist

            #val = np.array([np.array(self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)) for labda in labda_vec])
            #R_val = val[:,0]*dist
            #T_val = val[:,1]*dist
            #ABS_val = val[:,2]*dist
            #r_val = val[:,3]*dist
            #t_val = val[:,4]*dist
            #M_parts_val = val[:,5]
            #Mz_val = val[:,6]
            
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
    
    def run(self,n:Union[str,dict]='Default',d:Union[list,str]='Default',labda0:Union[str,float]='Default',FWHM:Union[str,float]='Default',option:str='numerical',steps:int=401,ang:float=0.0,pol:str='p',a:float=3) -> tuple:
        ''' Runs the calculation to obtain R, T, total basorbed values and absoption profile 
        
        Args:
            n (str/dict): Directory of refracive index functions, or 'Default' (=self.n)
            d (str/list): List of layer thicknesses (without the first and second infinite layers), if ='Default' then it is equal to self.d
            labda0 (str/float): Wavelength in meters, or 'Default' (=self.labda0)
            FWHM (str/float): FWHM of the beam in meters, or 'Default' (=self.FWHM)
            option (str): Selects beam type
                option = CW # Calculate for labda0 only
                option = numerical # Samples over spectrum with central wavelength labda0 and FWHM from -a*FWHM to a*FWHM around labda0. Note that when fill_value is for instance set to nan, going outsited te interpolation limits will raise an error. The spectrum is sampled over steps number of steps.
                option = pulse # Does the same as numerical but then integrating it instead of numerically solving it #...does not work with newest version
            steps (int): Number of samples between -a*FWHM+lambda and a*FWHM+labda (not workin when option = 'CW')
            ang (float): Angle of incidence
            pol (str): Polarization, eather 'p' or 's'
            a: Selts wavelength sample range (between -a*FWHM+lambda and a*FWHM+labda) (not workin when option = 'CW')

        Output:
            tuple: tuple containing:
                R (float): Intensity of reflection
                T (float): Intensity of transmission
                ABS (float): Intensity of absorption
                r (complex): Reflection (fresnel coefficient)
                Rerr (float): Error of R
                Terr (float): Error of T
                rerr (float): Error of r
                t (complex): Transmission (fresnel coefficient)
                M_parts (list): List of all Mjm1j @ Mj matrixes
                Mz :...to do
                abs_func (float function): Absorption profile versus thickens

        '''

        self.get_pulse(labda0=labda0,FWHM=FWHM)
        if labda0=='Default':
            labda0 = self.labda0
        if FWHM=='Default':
            FWHM = self.FWHM
        
        if option=='pulse':
            pass
            #R_func = lambda labda: self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)[0]*self.pulse(labda)
            #T_func = lambda labda: self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)[1]*self.pulse(labda)
            #r_func = lambda labda: self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)[3]*self.pulse(labda)

            #start = labda0 - 3*FWHM
            #end = labda0 + 3*FWHM
            #R, Rerr = scipy.integrate.quad(R_func,start,end)
            #T, Terr = scipy.integrate.quad(T_func,start,end)
            #r, rerr = scipy.integrate.quad(r_func,start,end)
            #ABS = 1.0-T-R

        elif option=='numerical':
            labda_vec = np.linspace(labda0-a*FWHM,labda0+a*FWHM,steps)
            dist = np.array([self.pulse(labda) for labda in labda_vec])
            Dlabda = labda_vec[1]-labda_vec[0]
            scale = 1.0/(np.nansum(dist)*Dlabda) # This should be 1 for plroperly sampling over lambda, but compensates for undersampling
            #val_names = ['R','T','ABS','r','t','M_parts','Mz']
            #for name in val_names:
            #    locals()[name+'_val'] = []

            #for labda in labda_vec:
            #    val = self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)
            #    for i in range(0,len(val)):
            #        locals()[val_names[i]+'_val'].append(val[i])
            #
            #for i in len(val_names):
            #    name = val_names[i]+'_val'
            #    if name[0]!=M:
            #        val = locals()[val_names[i]+'_val']
            #        locals()[val_names[i]+'_val'] = val*dist

            #print(R_val)

            ##val = np.array([np.array(self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)) for labda in labda_vec])
            ##R_val = val[:,0]*dist
            ##T_val = val[:,1]*dist
            ##ABS_val = val[:,2]*dist
            ##r_val = val[:,3]*dist
            ##t_val = val[:,4]*dist
            ##M_parts_val = val[:,5]
            ##Mz_val = val[:,6]
            #
            #Dlabda = labda_vec[1]-labda_vec[0]
            #R = np.nansum(R_val)*Dlabda
            #T = np.nansum(T_val)*Dlabda
            #r = np.nansum(r_val)*Dlabda
            #t = np.nansum(t_val)*Dlabda
            #M_parts = M_parts_val
            #Mz = Mz_val
            #
            #abs_func = lambda z: self.get_multiple_abs_wf(z,Mz_val,T_val/dist,d,n,labda_vec)
            #
            #Rerr = np.nan
            #Terr = np.nan
            #rerr = np.nan

            #ABS = 1.0-T-R

        elif option=='CW':
            labda_vec = np.array([labda0])
            dist = np.array([1.0])

            #R, T, ABS,r ,t, M_parts, Mz = self.calc_multilayer(labda0,n=n,d=d,ang=ang,pol=pol)
            #abs_func = lambda z: self.get_absorption(z,Mz,T,d,self.n,labda0)

            #Rerr = 0.0
            #Terr = 0.0
            #rerr = 0.0
        
        
        
        names = ['R', 'T', 'ABS', 'r', 'Rerr', 'Terr', 'rerr', 't', 'M_parts', 'Mz', 'abs_func']
        out_lists = {}
        out = {}
        for name in names:
            out_lists[name] = []
            out[name] = []
        
        for labda in labda_vec:
            R_l, T_l, ABS_l, r_l, Rerr_l, Terr_l, rerr_l, t_l, M_parts_l, Mz_l, abs_func_l = self.run_CW(n=n,d=d,labda=labda,FWHM=FWHM,steps=steps,ang=ang,pol=pol,a=a)
            for name in names:
                out_lists[name].append(locals()[name+'_l'])
        
        if option=='CW':
            for name in names:
                out[name] = out_lists[name][0]

        elif option=='numerical':
            for name in names:
                val = out_lists[name]
                
                if name=='abs_func':
                    #valout = lambda z: 0.0
                    #for i in range(0,len(out[name])):
                        #valout = lambda z: valout(z)+out[name][i](z)*dist*Dlabda
                    valout = None
                    pass
                elif name in ['Mz','M_parts']:
                    valout = val
                else:
                    valout = np.nansum(val*dist)*Dlabda*scale  
                out[name] = valout
            
            out['abs_func'] = lambda z: self.get_multiple_abs_wf(z,out_lists['Mz'],np.array(out_lists['T']),d,n,labda_vec)

        return out['R'], out['T'], out['ABS'], out['r'], out['Rerr'], out['Terr'], out['rerr'], out['t'], out['M_parts'], out['Mz'], out['abs_func']
    

    def run_CW(self,n:Union[str,dict]='Default',d:Union[list,str]='Default',labda:float='Default',FWHM:Union[str,float]='Default',steps:int=401,ang:float=0.0,pol:str='p',a:float=3) -> tuple:
        ''' Runs the calculation to obtain R, T, total basorbed values and absoption profile for one wavelegth
        
        Args:
            n (str/dict): Directory of refracive index functions, or 'Default' (=self.n)
            d (str/list): List of layer thicknesses (without the first and second infinite layers), if ='Default' then it is equal to self.d
            labda0 (str/float): Wavelength in meters, or 'Default' (=self.labda0)
            FWHM (str/float): FWHM of the beam in meters, or 'Default' (=self.FWHM)
            steps (int): Number of samples between -a*FWHM+lambda and a*FWHM+labda (not workin when option = 'CW')
            ang (float): Angle of incidence
            pol (str): Polarization, eather 'p' or 's'
            a: Selts wavelength sample range (between -a*FWHM+lambda and a*FWHM+labda) (not workin when option = 'CW')

        Output:
            tuple: tuple containing:
                R (float): Intensity of reflection
                T (float): Intensity of transmission
                ABS (float): Intensity of absorption
                r (complex): Reflection (fresnel coefficient)
                Rerr (float): Error of R
                Terr (float): Error of T
                rerr (float): Error of r
                t (complex): Transmission (fresnel coefficient)
                M_parts (list): List of all Mjm1j @ Mj matrixes
                Mz :...to do
                abs_func (float function): Absorption profile versus thickens

        '''

        R, T, ABS,r ,t, M_parts, Mz = self.calc_multilayer(labda,n=n,d=d,ang=ang,pol=pol)
        abs_func = lambda z: self.get_absorption(z,Mz,T,d,self.n,labda)

        Rerr = 0.0
        Terr = 0.0
        rerr = 0.0

        return R, T, ABS, r, Rerr, Terr, rerr, t, M_parts, Mz, abs_func

    def get_pulse(self,labda0:Union[str,float]='Default',FWHM:Union[str,float]='Default'):
        '''Distribution  over the spectrum, saves self.pulse as the spectrum

        Args:
            labda0 (str/float): Wavelength in meters, or 'Default' (=self.labda0)
            FWHM (str/float): FWHM of the beam in meters, or 'Default' (=self.FWHM)

        '''

        if labda0=='Default':
            labda0 = self.labda0
        if FWHM=='Default':
            FWHM = self.FWHM
        
        sig = FWHM/(2.0*((2.0*np.log(2))**0.5))

        pulse = lambda labda: ((1.0/(sig*((2.0*np.pi)**0.5)))*np.exp(-0.5*((labda-labda0)**2)/(sig**2)))

        self.pulse = pulse

        return 0
    
#--- ABSORPTION PROFILE --- #..maybe only for normal incidence
    def get_z_list(self,d_list:list,d:float) -> list:
        ''' Get z-values for each layer seen from the transmission side as function of position d into the material

        Args:
            d_list (list): List of layer thicknesses (without the first and second infinite layers)
            d (float); depth-coordinate in the layer

        Output:
            z_list (list): List of passed thicknesses of each layer at d
        '''
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

    def get_abs_function(self,M_all:list,z:float,labda:float,T:float,n_list:Union[str,dict]) -> float:
        '''Get the absorbtion as function of z

        Note: This may only hold for normal incidence. Source: Klaasjan...

        Args:
            M_all (list): List of all [Mjm1j, Mj, Mjm1j @ Mj, Mj]
            z (float):Depth coordinate in the layer
            labda (float): Wavelegth in meters
            T (float): Intesity of transmission (total)
            n_list (str/dict): Dictionary of n+1jK (refractive index function versus lambda) per key (material)

        Output:
            absorption (float): Absorption profile value at z

        '''

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


    def get_absorption(self,d:float,M_all:list,T:float,d_list:list,n_list:Union[str,list],labda:float) -> float:
        '''Gets the absorption function when option=CW...

        Args:
            d (float): Depth into the stack
            M_all (list): List of matrices
            T (float): Intensity of transmission of the stack
            d_list (list): List of layer thicknesses (without the first and second infinite layers)
            n_list (str/dict): Dictionary of n+1jK (refractive index function versus lambda) per key (material)
            labda (float): Wavelength in meters

        Output:
            absorption (float): Absorption profile value at z            

        '''
        z = self.get_z_list(d_list,d)
        if d>=np.nansum(d_list):
            absorption = 0
        elif d<0:
            absorption = 0
        else:
            absorption = self.get_abs_function(M_all,z,labda,T,n_list)

        return absorption

    def get_multiple_abs_wf(self,z:float,Mz_val:list,T_val:list,d:list,n:Union[str,dict],labda_vec:np.array) -> float:
        '''Gets the absorption function when option=numerical or pulse

        Args:
            z (float): Depth into the stack
            Mz_val (list): List of matrices
            T_val (list): List of intensity of transmission of the stack for eveery value in labda_vec
            d (list): List of layer thicknesses (without the first and second infinite layers)
            n (str/dict): Dictionary of n+1jK (refractive index function versus lambda) per key (material)
            labda_vec (list): List of wavelenths in meters

        Output:
            absorption (float): Absorption profile value at z      

        '''
        dist = np.array([self.pulse(labda) for labda in labda_vec])
        Dlabda = labda_vec[1]-labda_vec[0]
        weights = dist*Dlabda
        weights = weights/(np.nansum(weights))

        #val = np.array([self.get_absorption(z,Mz_val[i],T_val[i],d,n,labda_vec[i])*self.pulse(labda_vec[i])*(labda_vec[1]-labda_vec[0]) for i in range(0,len(labda_vec))])
        val = np.array([self.get_absorption(z,Mz_val[i],T_val[i],d,n,labda_vec[i])*weights[i] for i in range(0,len(labda_vec))])
        
        return np.nansum(val)
