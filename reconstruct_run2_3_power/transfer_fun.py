###########################
# This code is using method from Eisenstein & Hu 1998 (Baryonic feature in the matter transfer function).
# to calculate transfer function.
# Firstly made: 11/15/2015
# even using exact s, given same parameters, it doesn't give same shape as the lower panel in fig. 3.
# logout: 8:40pm, 11/16/2015, added the case omega_0 ~ 0.
# Copied from the folder shear_ps/Transfer_function on 08/21/2016
#
import math
import numpy as np
import matplotlib.pyplot as plt

# set cosmological parameters
# data factor.dat using this set of parameters
#case=0 to get data file transfer_fun0.dat for WiggleNowiggle project
'''
#----these parameters match with ones given by Hee-Jong in her code-------------
omega_0 = 0.279                       # total matter ratio
omega_b = 0.0497                      # baryon ratio
h = 0.6747                            # reduced Hubble constant: H_0/(100 km/s/Mpc)
TCMB = 2.727                          # CMB temperature(K)
#-------------------------------------
'''
'''
# default case, which I used for SVD_shear project
# to get transfer function for SVD shear ps, we use Planck 2013 best fit parameters
# to match with the parameter value in test7_nmode.py
#--------------------------------------------
omega_0 = 0.3175                       # total matter ratio
omega_b = 0.0490                      # baryon ratio
h = 0.6711                           # reduced Hubble constant: H_0/(100 km/s/Mpc)
TCMB = 2.727                          # CMB temperature(K)
#-------------------------------------
'''
#-----WMAP7 ML(1.4.2) ---------------------------------------
#omega_0 = 0.27285                       # total matter ratio
#omega_b = 0.04558                      # baryon ratio
#h = 0.702                           # reduced Hubble constant: H_0/(100 km/s/Mpc)
#TCMB = 2.727                          # CMB temperature(K)
#ns = 0.961
#-------------------------------------

#---- Find parameters from Planck 2015 TE+lowP fit Zovnimir linear power spectrum well (guessed it)-------
omega_0 = 0.3075                       # total matter ratio
omega_b = 0.0486                      # baryon ratio
h = 0.6773                           # reduced Hubble constant: H_0/(100 km/s/Mpc)
TCMB = 2.727                          # CMB temperature(K)
ns = 0.965


omgbh2 = omega_b*h*h
omg0h2 = omega_0*h*h

omega_c = omega_0 - omega_b         # cold drak matter ratio
theta_T = TCMB/2.7                  # \Theta_{2.7}

# T(k) = Omega_b/Omega_0 *T_b(k) + Omega_c/Omega_0 * T_c(k), --eq(16)
# in Python, math.log is logrithm of e in mathematics

z_eq = 2.50e4 * omg0h2 /theta_T**4.0                                                          #eq.(2)
k_eq = 7.46e-2* omg0h2/theta_T**2.0                                                           #eq.(3), unit is (Mpc^-1)

z_d_b1 = 0.313*pow(omg0h2, -0.419)*(1.0+0.607* omg0h2**0.674)
z_d_b2 = 0.238 * omg0h2**0.223
z_d = 1291.0* omg0h2**0.251/(1.0+0.659*omg0h2**0.828) * (1.0+z_d_b1* pow(omgbh2, z_d_b2))     #eq.(4)

def R(z):
    return 31.5*omgbh2 / theta_T**4.0 * 1.e3/z                                                #eq.(5)

# sound horizon: s, unit: Mpc, 
##s = 44.5*math.log(9.83/omg0h2)/math.sqrt(1.0+10.0* omgbh2**0.75)  #eq.(26) as the approximation of eq.(6)
s = 2.0/(3.0*k_eq)* math.sqrt(6.0/R(z_eq))* math.log(((1.0+R(z_d))**0.5 + (R(z_d)+R(z_eq))**0.5)/(1.0+R(z_eq)**0.5))  #eq.(6)


def q(k):
    return k * theta_T**2.0 / omg0h2                                                          #eq.(10), k in unit (Mpc^-1)

#---------------T_c part----------------#
# T_c(k) = f*Ttilde_0(k, 1, beta_c)+ (1-f)*Ttilde_0(k, alpha_c, beta_c)
#
a1 = (46.9*omg0h2)**0.670 * (1.0+(32.1* omg0h2)**(-0.532))                                    #
a2 = (12.0*omg0h2)**0.424 * (1.0+(45.0* omg0h2)**(-0.582))                                    #eq.(11)
alpha_c = a1**(-omega_b/omega_0) * a2**(-(omega_b/omega_0)**3.0)


b1 = 0.944/(1.0+(458.0*omg0h2)**(-0.708))
b2 = (0.395* omg0h2)**(-0.0266)
beta_c = 1.0/(1.0+ b1*((omega_c/omega_0)**b2 -1.0))                                           #eq.(12)

def C(k):
    return 14.2/alpha_c + 386.0/(1.0+69.9* q(k)**1.08)                                        #eq.(20)

# actually C(k) function depends on alpha_c
def Ttilde_0(k, alpha_c, beta_c):
    temp1 = math.log(math.e + 1.8*beta_c* q(k))
    temp2 = temp1 + C(k)* q(k)**2.0
    return temp1/temp2                                                                        #eq.(19)

def f(k):
    return 1.0/(1.0+(k*s/5.4)**4.0)                                                           #eq.(18)

def T_c(f, Ttilde_0, k):                                                                                   #eq.(17)
    return f(k)*Ttilde_0(k, 1.0, beta_c) * (1.0-f(k))* Ttilde_0(k, alpha_c, beta_c)


#----------------T_b part--------------#
# T_b(k) is the baryonic part and T_c(k) is the cold dark matter part

k_silk = 1.6*omgbh2**0.52 * omg0h2**0.73 * (1.0+pow(10.4*omg0h2, -0.95))                      #eq.(7), unit: (Mpc^-1)

def G(y):                                                                                     #eq.(15)
    return y*(-6.0*math.sqrt(1.0+y)+ (2.0+3.0*y)*math.log((math.sqrt(1.0+y)+1.0)/(math.sqrt(1.0+y)-1.0)))

alpha_b = 2.07*k_eq * s *pow(1.0+R(z_d), -0.75) * G((1.0+z_eq)/(1.0+z_d))                     #eq.(14)

beta_b = 0.5 + omega_b/omega_0 + (3.0-2.0*omega_b/omega_0)*math.sqrt((17.2*omg0h2)**2.0+1.0)  #eq.(24)
beta_node = 8.41*omg0h2**0.435                                                                #eq.(23)

def stilde(k):
    return s/pow(1.0+(beta_node/(k*s))**3.0, 1.0/3.0)                                         #eq.(22)

def T_b(Ttilde_0, k):
    temp1 = Ttilde_0(k, 1.0, 1.0)/(1.0+(k*s/5.2)**2.0)
    temp2 = alpha_b/(1.0+pow(beta_b/(k*s), 3.0)) * math.exp(-pow(k/k_silk, 1.4))
    j_0 = math.sin(k * stilde(k))/(k * stilde(k))
    return (temp1+temp2)*j_0                                                                  #eq.(21)

def T(T_b, T_c, f, Ttilde_0, k):
    return omega_b/omega_0*T_b(Ttilde_0, k) + omega_c/omega_0*T_c(f, Ttilde_0, k)                   #eq.(16)

#inputf = '../../WiggleNowiggle/CAMB_WMAP7_ML_1.4.2_matterpower.dat'
#inputf = "/Users/ding/Documents/playground/WiggleNowiggle/CAMB_Pk/CAMB_Planck2015__matterpower.dat"
##! the unit of k from CAMB is h Mpc^-1, differs by h from k unit in the paper
#k, Pk = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
#print(k)
k = np.linspace(1.e-4, 100.0, 10000)
# transfer k_CAMB unit from h Mpc^-1 to Mpc^-1
k = k * h

# calculate transfer function
Tr = np.array([T(T_b, T_c, f, Ttilde_0, k[i]) for i in range(len(k))])
# fitted power spectrum = T^2 * k^n, n=1
Pk_fit = Tr**2.0* (k/h)

##----- plot P(k)= T^2(k)*k
'''
# be in mind, when we plot, k unit is h Mpc^-1, it's annoying a little bit!
plt.loglog(k/h, Pk, '--', k, Pk_fit)
plt.show()
plt.close()
'''

'''
##------------plot plot transfer function with wiggles---------------##
# We try to reproduce the results shown in Fig.3. in Eisenstein & Hu (1998)
# absolute value of transfer function versus k
plt.loglog(k/h, abs(Tr))
plt.xlabel('k ($h$ $Mpc^{-1}$)')
plt.ylabel('|T(k)|')
plt.grid(True)
plt.xlim([1.e-3, 1.1])
plt.ylim([1.e-4, 2.0])
text_str = '$\Omega_b/\Omega_0={0}$ \n$\Omega_0={1}$, $h={2}$'.format(omega_b/omega_0, omega_0, h)
plt.text(0.005, 1.e-3, text_str, fontsize=15)
#plt.show()
figname = 'transfer_fun_omega_b_{0}_omega0_{1}_h_{2}.eps'.format(omega_b, omega_0, h)
plt.savefig(figname)
plt.close()
'''

##-----------------------------------------------------##
##-------------zero-baryon approximation--------------##
##
alpha_Gamma = 1.0-0.328* math.log(431.0*omg0h2)*omega_b/omega_0 + 0.38* math.log(22.3*omg0h2)* (omega_b/omega_0)**2.0  #eq.(31)

def Gamma_eff(k):
    return omg0h2*(alpha_Gamma+(1.0-alpha_Gamma)/(1.0+pow(0.43*k*s, 4.0)))                      #eq.(30)*h, h is absoved from eq.(28)

                   
def q_0(k):
    return k* theta_T**2.0/Gamma_eff(k)                                         #eq.(28), k unit is (Mpc^-1), h in denominator moved to eq.(30)

                   
def T_0(q, k):
    L_0 = math.log(2.0*math.e + 1.8*q(k))
    C_0 = 14.2+731.0/(1.0+62.5*q(k))
    return L_0/(L_0+C_0*q(k)**2.0)                                                                  #eq.(29)
                   
def T_b_0(T_0, q, k):
    temp1 = T_0(q, k)/(1.0+(k*s/5.2)**2.0)                                                        #copied from eq.(21)
    temp2 = alpha_b/(1.0+pow(beta_b/(k*s), 3.0)) * math.exp(-pow(k/k_silk, 1.4))
    j_0 = math.sin(k * stilde(k))/(k * stilde(k))
    return (temp1+temp2)*j_0

                   
#in the case Ttilde=T_0, T_c = T_0
Tr_0 = np.array([omega_b/omega_0* T_b_0(T_0, q_0, k[i]) + omega_c/omega_0* T_0(q_0, k[i]) for i in range(len(k))])
T0_norm = np.array([T_0(q_0, k[i]) for i in range(len(k))])

# examine the code with Hee-Jong's code: cos_valuesnew.c, it looks correct(set input parameters and output k correctly)#
#inputf = 'factor.dat'
#hs_k, hs_trans = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0,4), unpack=True) # hs_k in unit h*Mpc^-1 #
#
#plt.loglog(hs_k, hs_trans, k/h, T0_norm, '--')
#plt.show()

#outf = 'transfer_fun.dat' # default case
#outf = 'transfer_fun_WMAP7_ML_1.4.2.dat' # case 0
outf = 'transfer_fun_Planck2015_TT_lowP.dat'
headerline = '# The parameters of TF are omega_0={0}, omega_b={1}, h={2}\n'.format(omega_0,omega_b,h)
with open(outf, 'w') as file1:
    file1.write(headerline)
    file1.write("#k       T_0\n")
    for i in range(len(k)):
        output_string = ""
        output_string += str(k[i]/h) + '  ' + str(T0_norm[i])+"\n"
        file1.write(output_string)
file1.close()



plt.loglog(k/h, T0_norm, '--', label='$T_0(k)$')
plt.loglog(k/h, Tr_0, label='$T(k)$')
leg = plt.legend(loc='upper right')
leg.set_frame_on(False)
plt.xlabel('k ($h$ $Mpc^{-1}$)')
plt.ylabel('T(k)')
plt.xlim([1.e-3, 1.0])
plt.ylim([1.e-3, 1.2])
plt.grid(True)
text_str = '$\Omega_b/\Omega_0={0}$, $\Omega_0={1}$, $h={2}$'.format(omega_b/omega_0, omega_0, h)
plt.text(5.e-3, 2.e-3, text_str, fontsize=15)
#figname = 'transfer_fun_omega_b_{0}_omega0_{1}_h_{2}.eps'.format(omega_b, omega_0, h)
#plt.savefig(figname)
plt.show()




                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                
 



