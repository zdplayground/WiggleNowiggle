#! ~/vert/bin/python
# Copied the code from multiplot_mean_individual.py, rename the file to multiplot_bscale_mean.py, 06/27/2016.
# 1. Plot fitted alpha_perp, alpha_para, Sigma_xy, Sigma_z, b_0, b_scale for the mean power spectra in terms of bias.
# 2. Simplify the code by removing some defined functions and routines.
#
from __future__ import print_function
import numpy as np
from scipy import interpolate
import time
import os
import matplotlib.pyplot as plt

sim_run = 'run2_3'
N_dset = 20
N_mu = 100
N_params = 4
rec_id = 0                 # 0: no reconstruction; 1: with reconstruction

sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = ['NW', 'WG']
sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space

sub_sim_space = ['real', 'rsd']
title_space = ['in real space', 'in redshift space']
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']
Pk_type = ['obs', 'true']

mcut_Npar_list = [[37, 149, 516, 1524, 3830],
                  [35, 123, 374, 962, 2105],
                  [34, 103, 290, 681, 1390]]
N_masscut = [5, 5, 5]
# The bias_array value corresponds to the iterms in mcut_Npar_list. If Npar of mass cut point is larger, then bias is larger too.
bias_array = np.array([[0.90, 1.09, 1.40, 1.81, 2.33],
                       [1.21, 1.52, 1.96, 2.53, 3.20],
                       [1.48, 1.88, 2.41, 3.07, 3.87]])
Sigsm2_z1 = [2*31, 2*38, 2*49, 2*65, 2*86]  # From Hee-Jong's note

# The mean bias without masscut from lower to higher redshift. Get it from run_3_bias_1D_Pk.dat in the folder run3_1D_bias.
bias_0 = np.array([1.0, 1.0, 1.0])
omega_m0 = 0.3075
omega_l = 0.6925

idir = '/Users/ding/Documents/playground/WiggleNowiggle/HS_pre_rec_run2_3/'

inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

Pk_wovers = Pk_wiggle/Pk_smooth
tck_Pk_wovers = interpolate.splrep(k_smooth, Pk_wovers)
tck_Pk_sm = interpolate.splrep(k_smooth, Pk_smooth)


#----- select a certain \mu bin -------#
indices_p1, indices_p2, indices_p3 = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
ifile = idir+'{}_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kaver.wig_minus_now_mean_fof_a_0.5000_mcut34.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id])
print(ifile)
k_sorted, mu_sorted = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(0, 1), unpack=True)
print(k_sorted, mu_sorted)

for i in xrange(len(mu_sorted)):
    if mu_sorted[i]>=0.0 and mu_sorted[i]<0.1:
        indices_p1 = np.append(indices_p1, i)
    elif mu_sorted[i]>=0.4 and mu_sorted[i]<0.5:
        indices_p2 = np.append(indices_p2, i)
    elif mu_sorted[i]>=0.9 and mu_sorted[i]<1.0:
        indices_p3 = np.append(indices_p3, i)
print(i)
print(sum(map(np.size, [indices_p1, indices_p2, indices_p3])))

# define a plot function showing the paramter alpha_perp and alpha_para from fof with diffent biases as well as from subsample
def plot_alpha_fofsub_bias(bias_array, y1, y2, bias_0, sub_y1, sub_y2, N_masscut, space_id, sim_space, title_space, odir, figname):
    ylim_list = np.array([[-0.2, 1.4], [-0.4, 0.4]])
    n0, n1, n2 = N_masscut[0], N_masscut[1], N_masscut[2]
    print(n0, n1, n2)
    print(bias_array[0][:], y1[0:n0, 0])
    plt.errorbar(bias_array[0][:], (y1[0:n0, 0]-1.0)*100, yerr=[y1[0:n0, 2]*100, y1[0:n0, 1]*100], fmt='k*-')
    plt.errorbar(bias_array[1][:], (y1[n0 : n0+n1, 0]-1.0)*100, yerr=[y1[n0 : n0+n1, 2]*100, y1[n0 : n0+n1, 1]*100], fmt='ro-')
    plt.errorbar(bias_array[2][:], (y1[n0+n1 : n0+n1+n2, 0]-1.0)*100, yerr=[y1[n0+n1 : n0+n1+n2, 2]*100, y1[n0+n1 : n0+n1+n2, 1]*100], fmt='bv-')

    plt.errorbar(bias_array[0][:], (y2[0:n0, 0]-1.0)*100, yerr=[y2[0:n0, 2]*100, y2[0:n0, 1]*100], fmt='k*--')
    plt.errorbar(bias_array[1][:], (y2[n0 : n0+n1, 0]-1.0)*100, yerr=[y2[n0 : n0+n1, 2]*100, y2[n0 : n0+n1, 1]*100], fmt='ro--')
    plt.errorbar(bias_array[2][:], (y2[n0+n1 : n0+n1+n2, 0]-1.0)*100, yerr=[y2[n0+n1 : n0+n1+n2, 2]*100, y2[n0+n1 : n0+n1+n2, 1]*100], fmt='bv--')

    plt.errorbar(bias_0[0], (sub_y1[0, 0]-1.0)*100, yerr=[[sub_y1[0, 2]*100], [sub_y1[0, 1]*100]], fmt='ks')
    plt.errorbar(bias_0[1], (sub_y1[1, 0]-1.0)*100, yerr=[[sub_y1[1, 2]*100], [sub_y1[1, 1]*100]], fmt='rs')
    plt.errorbar(bias_0[2], (sub_y1[2, 0]-1.0)*100, yerr=[[sub_y1[2, 2]*100], [sub_y1[2, 1]*100]], fmt='bs')

    plt.errorbar(bias_0[0], (sub_y2[0, 0]-1.0)*100, yerr=[[sub_y2[0, 2]*100], [sub_y2[0, 1]*100]], fmt='kd')
    plt.errorbar(bias_0[1], (sub_y2[1, 0]-1.0)*100, yerr=[[sub_y2[1, 2]*100], [sub_y2[1, 1]*100]], fmt='rd')
    plt.errorbar(bias_0[2], (sub_y2[2, 0]-1.0)*100, yerr=[[sub_y2[2, 2]*100], [sub_y2[2, 1]*100]], fmt='bd')
    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$\alpha-1.0$ (%)', fontsize=24)
    textline = r"solid lines: FoF $\alpha_{\bot}$"+"\n"+r"dashed lines: FoF $\alpha_{\|}$"\
                "\n"+r"square dots: Subsample $\alpha_{\bot}$"+"\n"+r"diamond dots: Subsample $\alpha_{\|}$"+"\n"+\
               r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(0.6, 0.9, textline)
    plt.xlim([0.5, 4.2])
    plt.ylim([ylim_list[rec_id, 0], ylim_list[rec_id, 1]])
    plt.title(r"Fitted $\alpha_{\bot}$ and $\alpha_{\|}$ %s" %(title_space[space_id]), fontsize=16)

    plt.savefig(odir+figname)
    plt.show()
    plt.close()

# similar with the function plot_alpha_fofsub_bias, but for parameter Sigma_xy, Sigma_z
def plot_Sigma_fofsub_bias(bias_array, y1, y2, bias_0, sub_y1, sub_y2, N_masscut, space_id, sim_space, title_space, sim_a, omega_m0, omega_l, odir, figname):
    ylim_list = np.array([[0., 12.0], [0., 12.0]])
    n0, n1, n2 = N_masscut[:]
    plt.errorbar(bias_array[0][:], y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*-')
    plt.errorbar(bias_array[1][:], y1[n0 : n0+n1, 0], yerr=[y1[n0 : n0+n1, 2], y1[n0 : n0+n1, 1]], fmt='ro-')
    plt.errorbar(bias_array[2][:], y1[n0+n1 : n0+n1+n2, 0], yerr=[y1[n0+n1 : n0+n1+n2, 2], y1[n0+n1 : n0+n1+n2, 1]], fmt='bv-')
    
    plt.errorbar(bias_array[0][:], y2[0 : n0, 0], yerr=[y2[0 : n0, 2], y2[0 : n0, 1]], fmt='k*--')
    plt.errorbar(bias_array[1][:], y2[n0 : n0+n1, 0], yerr=[y2[n0 : n0+n1, 2], y2[n0 : n0+n1, 1]], fmt='ro--')
    plt.errorbar(bias_array[2][:], y2[n0+n1 : n0+n1+n2, 0], yerr=[y2[n0+n1 : n0+n1+n2, 2], y2[n0+n1 : n0+n1+n2, 1]], fmt='bv--')
    
    plt.errorbar(bias_0[0], sub_y1[0, 0], yerr=[[sub_y1[0, 2]], [sub_y1[0, 1]]], fmt='ks')
    plt.errorbar(bias_0[1], sub_y1[1, 0], yerr=[[sub_y1[1, 2]], [sub_y1[1, 1]]], fmt='rs')
    plt.errorbar(bias_0[2], sub_y1[2, 0], yerr=[[sub_y1[2, 2]], [sub_y1[2, 1]]], fmt='bs')
    
    plt.errorbar(bias_0[0], sub_y2[0, 0], yerr=[[sub_y2[0, 2]], [sub_y2[0, 1]]], fmt='kd')
    plt.errorbar(bias_0[1], sub_y2[1, 0], yerr=[[sub_y2[1, 2]], [sub_y2[1, 1]]], fmt='rd')
    plt.errorbar(bias_0[2], sub_y2[2, 0], yerr=[[sub_y2[2, 2]], [sub_y2[2, 1]]], fmt='bd')

    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$\Sigma$ $[Mpc/h]$', fontsize=24)
    textline = r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0" + "\n"+\
               r"solid lines: FoF $\Sigma_{xy}$"+"\n"+r"dashed lines: FoF $\Sigma_{z}$""\n"+\
               r"square dots: Subsample $\Sigma_{xy}$"+"\n"+r"diamond dots: Subsample $\Sigma_{z}$"
    
    if space_id == 1:
        c1 = 1.0+f_value(omega_m0, omega_l, float(sim_a[0]))
        c2 = 1.0+f_value(omega_m0, omega_l, float(sim_a[1]))
        c3 = 1.0+f_value(omega_m0, omega_l, float(sim_a[2]))
        plt.errorbar(bias_array[0][:], c1*y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*:')
        plt.errorbar(bias_array[1][:], c2*y1[n0 : n0+n1, 0], yerr=[y1[n0 : n0+n1, 2], y1[n0 : n0+n1, 1]], fmt='ro:')
        plt.errorbar(bias_array[2][:], c3*y1[n0+n1 : n0+n1+n2, 0], yerr=[y1[n0+n1 : n0+n1+n2, 2], y1[n0+n1 : n0+n1+n2, 1]], fmt='bv:')
        textline += "\n"+r"dotted lines: $(1+\Omega_m^{0.6}(z))\Sigma_{xy}$"
    plt.text(2.5, 5.5, textline)
    plt.xlim([0.5, 4.2])
    plt.ylim([ylim_list[rec_id, 0], ylim_list[rec_id, 1]]) # with four parameters
    #plt.ylim([3.0, 14.0])   # add with 3 additional parameters
    plt.title(r"Fitted $\Sigma_{xy}$ and $\Sigma_{z}$ %s" %(title_space[space_id]), fontsize=16)

    plt.savefig(odir+figname)
    plt.show()
    plt.close()

# Compare b_0 from the simulation with that from the calculation in low k region
def plot_b_0_fofsub(bias_array, y1, sub_bias_0, sub_y1, N_masscut, space_id, sim_space, title_space, odir, figname):
    n0, n1, n2 = N_masscut[:]
    plt.errorbar(bias_array[0][:], y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*-')
    plt.errorbar(bias_array[1][:], y1[n0 : n0+n1, 0], yerr=[y1[n0 : n0+n1, 2], y1[n0 : n0+n1, 1]], fmt='ro-')
    plt.errorbar(bias_array[2][:], y1[n0+n1 : n0+n1+n2, 0], yerr=[y1[n0+n1 : n0+n1+n2, 2], y1[n0+n1 : n0+n1+n2, 1]], fmt='bv-')
    
    plt.errorbar(bias_0[0], sub_y1[0, 0], yerr=[[sub_y1[0, 2]], [sub_y1[0, 1]]], fmt='ks')
    plt.errorbar(bias_0[1], sub_y1[1, 0], yerr=[[sub_y1[1, 2]], [sub_y1[1, 1]]], fmt='rs')
    plt.errorbar(bias_0[2], sub_y1[2, 0], yerr=[[sub_y1[2, 2]], [sub_y1[2, 1]]], fmt='bs')
    diag_x = np.linspace(0, 4.0, 100)
    plt.plot(diag_x, diag_x, '--')
    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$b_0$', fontsize=24)
    textline = r"solid lines: FoF"\
               "\n"+r"square dots: Subsample"+ "\n"\
                r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(0.5, 2.5, textline, fontsize=16)
    plt.savefig(odir + figname)
    plt.show()
    plt.close()

# Plot b_scale from the fitting code mcmc_fit_wig-now_b_scale.py
def plot_b_scale_fofsub(bias_array, y1, bias_0, sub_y1, N_masscut, space_id, sim_space, title_space, odir, figname):
    n0, n1, n2 = N_masscut[:]
    plt.errorbar(bias_array[0][:], y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*-')
    plt.errorbar(bias_array[1][:], y1[n0 : n0+n1, 0], yerr=[y1[n0 : n0+n1, 2], y1[n0 : n0+n1, 1]], fmt='ro-')
    plt.errorbar(bias_array[2][:], y1[n0+n1 : n0+n1+n2, 0], yerr=[y1[n0+n1 : n0+n1+n2, 2], y1[n0+n1 : n0+n1+n2, 1]], fmt='bv-')
    
    plt.errorbar(bias_0[0], sub_y1[0, 0], yerr=[[sub_y1[0, 2]], [sub_y1[0, 1]]], fmt='ks')
    plt.errorbar(bias_0[1], sub_y1[1, 0], yerr=[[sub_y1[1, 2]], [sub_y1[1, 1]]], fmt='rs')
    plt.errorbar(bias_0[2], sub_y1[2, 0], yerr=[[sub_y1[2, 2]], [sub_y1[2, 1]]], fmt='bs')
    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$b_{\mathtt{scale}}$', fontsize=24)
    textline = r"solid lines: FoF"\
               "\n"+r"square dots: Subsample"+ "\n"\
               r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(2.4, 25.0, textline, fontsize=16)
    plt.savefig(odir + figname)
    plt.show()
    plt.close()


# plot P(k, \mu) in terms of k
def plot_P_kmu(k, P_kmu, odir, space_id, z_id, mcut_id, filename):
    plt.plot(k, P_kmu, '+')
    plt.xscale("log")
    plt.xlim([0.004, 0.3])
    plt.ylim([-0.08, 0.08])
    plt.xlabel(r"$k$ $[h/Mpc]$")
    plt.ylabel(r"$(P_{wig}/P_{now})_{mean}$")
    plt.title("$z={}$, bias=${}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], title_space[space_id]))
    figname = odir+ "Pk_"+filename + ".pdf"
    plt.savefig(figname)
    #plt.show()
    plt.close()


# calculate f~\omega_m(z)^0.6
def f_value(omega_m0, omega_l, a):
    omega_ma = omega_m0/(omega_m0 + a**3.0 * omega_l)
    return omega_ma**0.6

# In the 4 parameters case, get the fitted power spectrum from the fitting model
def fit_obs(theta, k_p, mu_p, tck_Pk_wovers):
    alpha_1, alpha_2, sigma_xy, sigma_z = theta
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    Pk_t = interpolate.splev(k_t, tck_Pk_wovers, der=0)
    Pk_model = (Pk_t-1.0)*np.exp(k_t*k_t *(mu_t*mu_t*(sigma_xy+sigma_z)*(sigma_xy-sigma_z)-sigma_xy**2.0)/2.0) #it has deleted 1.0
    return Pk_model

# With the mode-coupling constant(totally 5 parameters), get the fitted power spectrum from the fitting model
def fit_obs_mc(theta, k_p, mu_p, tck_Pk_wovers, tck_Pk_sm):
    alpha_1, alpha_2, sigma_xy, sigma_z, const = theta
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    kt_para2 = (k_t*mu_t)**2.0
    
    Pk_t = interpolate.splev(k_t, tck_Pk_wovers, der=0)
    Pk_sm = interpolate.splev(k_t, tck_Pk_sm, der=0)
    # here Pk_t represents Pk_wig/Pk_now in theory
    Pk_fittrue = (Pk_t-1.0)*np.exp((kt_para2 * (sigma_xy+sigma_z)*(sigma_xy-sigma_z) - (k_t*sigma_xy)**2.0)/2.0)
    Pk_fitobs = Pk_fittrue/(1.0 + const/Pk_sm)
    return Pk_fitobs


# overplot fitted power spectrum with the observed one
##def overplot_fit_obs(k_p, k_smooth, Pk_obs, Pk_fit, Pk_wovers, var_Pk_wnw_mean, space_id, z_id, reduced_chi2, odir, fname, mu_boundary=None): # for Subsample P(k, mu) plotting
def overplot_fit_obs(k_p, k_smooth, Pk_obs, Pk_fit, Pk_wovers, var_Pk_wnw_mean, space_id, z_id, mcut_id, reduced_chi2, odir, fname, mu_boundary=None):
    plt.figure(figsize=(13, 8))
    plt.errorbar(k_p, Pk_obs, yerr=var_Pk_wnw_mean**0.5, fmt='.', lw=1.5, label='obs')
    plt.plot(k_p, Pk_fit, 'r-', lw=1.5, label='fit')
    plt.plot(k_smooth, Pk_wovers-1.0, '-', lw=1.0, label='theory')
    plt.xscale("log")
    plt.xlabel('$k$ [$h/Mpc$]', fontsize=28)
    plt.xticks([1.e-1, 2.e-1, 3.e-1], [0.1, 0.2, 0.3], fontsize=24)
    plt.yticks(fontsize=24)
    ##plt.ylabel(r'$(P_{\mathtt{wig}}/P_{\mathtt{now}})_{\mathtt{mean}}-1.0$', fontsize=28)
    plt.ylabel(r'$P_{\mathtt{wig}}-P_{\mathtt{now}})/(G^2 b^2 P_{\mathtt{sm}})$', fontsize=28)  # only for (P_wig-P_now)/(G^2 b^2 P_sm) case
    plt.grid(True)
    plt.legend(loc='best', fontsize=24)
    plt.xlim([0.0042, 0.32])
    plt.ylim([-0.1, 0.1])
    ##plt.title(r"$z={0}$, ${1}<\mu<{2:.1f}$, $\Delta\chi^2={3:.2f}$ {4}".format(sim_z[z_id], mu_boundary[0], mu_boundary[1], reduced_chi2, title_space[space_id]), fontsize=24) # for subsample case
    plt.title(r"$z={0}$, bias={1}, ${2}<\mu<{3:.1f}$, $\Delta\chi^2={4:.2f}$ {5}".format(sim_z[z_id], bias_array[z_id][mcut_id], mu_boundary[0], mu_boundary[1], reduced_chi2, title_space[space_id]), fontsize=24)
    figname = odir + "fit_Pk_"+fname+".pdf"
    plt.savefig(figname)
    #plt.show()
    plt.close()


# define a function to show the reduced chi^2 from three different fitting models.
def plot_reduced_chi2(bias_array, y1, bias_0, sub_y1, odir, figname):
    plt.plot(bias_array[0][:], y1[0, 0, :], 'k*-.', lw=2.0) # line style shows the fitting model.
    plt.plot(bias_array[1][:], y1[0, 1, :], 'ro-.', lw=2.0)
    plt.plot(bias_array[2][:], y1[0, 2, :], 'bv-.', lw=2.0)
    
    plt.plot(bias_array[0][:], y1[1, 0, :], 'k*--')
    plt.plot(bias_array[1][:], y1[1, 1, :], 'ro--')
    plt.plot(bias_array[2][:], y1[1, 2, :], 'bv--')
    
    plt.plot(bias_array[0][:], y1[2, 0, :], 'k*-')
    plt.plot(bias_array[1][:], y1[2, 1, :], 'ro-')
    plt.plot(bias_array[2][:], y1[2, 2, :], 'bv-')
    
    plt.plot(bias_0[0], sub_y1[0, 0], 'k*', bias_0[0], sub_y1[0, 1], 'r*', bias_0[0], sub_y1[0, 2], 'b*')
    plt.plot(bias_0[0], sub_y1[1, 0], 'kp', bias_0[0], sub_y1[1, 1], 'rp', bias_0[0], sub_y1[1, 2], 'bp')
    plt.plot(bias_0[0], sub_y1[2, 0], 'kh', bias_0[0], sub_y1[2, 1], 'rh', bias_0[0], sub_y1[2, 2], 'bh',)
    plt.xlabel('bias', fontsize=24)
    plt.ylim([1.05, 1.30])
    plt.ylabel(r'$\Delta \chi^2$', fontsize=24)
    textline = r"blue: $z=1.0$"+"\n"+r"red: $z=0.6$"+"\n"+r"black: $z=0.0$"
    plt.text(3.0, 1.17, textline, fontsize=20)
    plt.savefig(odir + figname)
    #plt.show()
    plt.close()
#---- part 3: compare the error bars from two cases, one is the one sigma error of fitted mean power spectrum, the other is the square root of the sample variance
#--
def compare_sigma_errs():
    fof_idir_mean = './{}_{}_params_fitted_mean_dset/'.format(sim_run, rec_dirs[rec_id])
    fof_idir_avgind = './{}_{}_params_fitted_individual_avg_sigmaerror/'.format(sim_run, rec_dirs[rec_id])
    sub_idir_mean = '../run_2_3_2D_power/2d_Pk_mu_{}/params_fitted_ksorted_mu_shotnoise/'.format(N_mu)
    sub_idir_avgind = '../run_2_3_2D_power/2d_Pk_mu_{}/params_fitted_individual_avg_sigmaerror/'.format(N_mu)
    
    odir = './{}_figs_barn_avg_individual/'.format(rec_dirs[rec_id])
    
    for space_id in xrange(2):
        sigerr_mean_matrix = np.empty((0, 4), dtype='float64') # sigma error mean for parameters from fitted mean P(k), first column is for alpha_perp, the second is for alpha_para
        sigerr_avgind_matrix = np.empty((0, 4), dtype='float64') # sigma error average individual for parameters from average fitted P(k) individually
        sub_sigerr_m_matrix = np.empty((0, 4), dtype='float64')
        sub_sigerr_a_matrix = np.empty((0, 4), dtype='float64')
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                fof_ifile_mean = fof_idir_mean + '{}kave{}.wnw_mean_fof_a_{}_mcut{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                sigerr_mean = np.loadtxt(fof_ifile_mean, dtype='float64', comments='#', usecols=(1,))
                sigerr_mean_matrix = np.vstack((sigerr_mean_matrix, sigerr_mean))
                
                fof_ifile_avgind = fof_idir_avgind + '{}kave{}.wnw_fof_a_{}_mcut{}_params_avg_sigma.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                sigerr_avgind = np.loadtxt(fof_ifile_avgind, dtype='float64', comments='#', usecols=(1,))
                sigerr_avgind_matrix = np.vstack((sigerr_avgind_matrix, sigerr_avgind))
            
            sub_ifile_mean = sub_idir_mean + 'sub_{}_{}_params.dat'.format(sim_a[z_id], sub_sim_space[space_id])
            sub_sigerr_m = np.loadtxt(sub_ifile_mean, dtype='float64', comments='#', usecols=(1,))
            sub_sigerr_m_matrix = np.vstack((sub_sigerr_m_matrix, sub_sigerr_m))
            
            sub_ifile_avgind = sub_idir_avgind + 'sub_a_{}_{}_params_avg_sigma.dat'.format(sim_a[z_id], sub_sim_space[space_id])
            sub_sigerr_a = np.loadtxt(sub_ifile_avgind, dtype='float64', comments='#', usecols=(1,))
            sub_sigerr_a_matrix = np.vstack((sub_sigerr_a_matrix, sub_sigerr_a))
        
        
        figname = '{}_fof_sub_params_mu_{}_alpha_errdiff{}.pdf'.format(rec_dirs[rec_id], N_mu, sim_space[space_id])
        plot_params_sigerr_diff(bias_array, sigerr_mean_matrix, sigerr_avgind_matrix, bias_0, sub_sigerr_m_matrix, sub_sigerr_a_matrix,
                                N_masscut, space_id, sim_space, title_space, odir, figname)

#----- part 1: plot parametrs fitted for the mean P(k)
#
def read_params(inputf):
    N_skip_header = 1
    N_skip_footer = 1
    data_m = np.genfromtxt(inputf, dtype=np.float, comments=None, skip_header=N_skip_header, skip_footer=N_skip_footer, usecols=(1, 2, 3))
    return data_m

def show_params_fitted():
    # 1 represents true as a free parameter, 0 is false. Parameters are alpha_perp, alpha_para, \Sigam_xy, \Sigma_z, b_0, b_scale
    #initial_pvalue = np.array([1, 1, 0, 0, 1, 1])
    #initial_pvalue = np.array([1, 1, 1, 1, 1, 0])
    initial_pvalue = np.array([1, 1, 1, 1, 1, 1])
    indices_str = ''.join(map(str, initial_pvalue))

    #fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_dset_broad_prior/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    #odir = idir+'{}_figs_barn_mean_wnw_diff_bscale/'.format(rec_dirs[rec_id])
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale_broad_prior/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    
    for space_id in xrange(1):
        alpha_1m, alpha_2m = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        sub_alpha_1m, sub_alpha_2m = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        Sigma_xym, Sigma_zm = np.empty((0,3),dtype='float'), np.empty((0,3), dtype='float')
        sub_Sigma_xym, sub_Sigma_zm = np.empty((0,3),dtype='float'), np.empty((0,3), dtype='float')
        b_0m, b_scalem = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        sub_b_0m, sub_b_scalem = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                
                fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], indices_str)
                #print(fof_ifile)
                #read parameter data file
                params_err = read_params(fof_ifile)
                #print(params_err)
                alpha_1m = np.vstack((alpha_1m, params_err[0, :]))
                alpha_2m = np.vstack((alpha_2m, params_err[1, :]))
#                b_0m = np.vstack((b_0m, params_err[2, :]))
#                b_scalem = np.vstack((b_scalem, params_err[3, :]))
                Sigma_xym = np.vstack((Sigma_xym, params_err[2, :]))
                Sigma_zm = np.vstack((Sigma_zm, params_err[3, :]))
                b_0m = np.vstack((b_0m, params_err[4, :]))
                b_scalem = np.vstack((b_scalem, params_err[5, :]))


            sub_ifile = sub_idir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], indices_str)
            print(sub_ifile)
            params_err = read_params(sub_ifile)
            sub_alpha_1m = np.vstack((sub_alpha_1m, params_err[0, :]))
            sub_alpha_2m = np.vstack((sub_alpha_2m, params_err[1, :]))
#            sub_b_0m = np.vstack((sub_b_0m, params_err[2, :]))
#            sub_b_scalem = np.vstack((sub_b_scalem, params_err[3, :]))
#            print(sub_b_0m)
            sub_Sigma_xym = np.vstack((sub_Sigma_xym, params_err[2, :]))
            sub_Sigma_zm = np.vstack((sub_Sigma_zm, params_err[3, :]))
            sub_b_0m = np.vstack((sub_b_0m, params_err[4, :]))
            sub_b_scalem = np.vstack((sub_b_scalem, params_err[5, :]))

        figname_alpha = '{}kave{}_wig-now_mean_fof_sub_alpha_params{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_Sigma = '{}kave{}_wig-now_mean_fof_sub_Sigma_params{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_0 = '{}kave{}_wig-now_mean_fof_sub_b0_params{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_scale = '{}kave{}_wig-now_mean_fof_sub_bscale_params{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)

    
        plot_alpha_fofsub_bias(bias_array, alpha_1m, alpha_2m, bias_0, sub_alpha_1m, sub_alpha_2m, N_masscut, space_id, sim_space, title_space, odir, figname_alpha)
        plot_Sigma_fofsub_bias(bias_array, Sigma_xym, Sigma_zm, bias_0, sub_Sigma_xym, sub_Sigma_zm, N_masscut, space_id, sim_space, title_space, sim_a, omega_m0, omega_l, odir, figname_Sigma)
        plot_b_0_fofsub(bias_array, b_0m, bias_0, sub_b_0m, N_masscut, space_id, sim_space, title_space, odir, figname_b_0)
        plot_b_scale_fofsub(bias_array, b_scalem, bias_0, sub_b_scalem, N_masscut, space_id, sim_space, title_space, odir, figname_b_scale)


# show reduced \chi^2 from different numbers of fitting parameters
def show_reduced_chi2():
    pindices_list = ['110011', '111110', '111111']
    N_skip_header_list = [5, 6, 7]
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale/'.format(rec_dirs[rec_id])
    chi2_matrix = np.zeros((3,3,5))
    sub_chi2_matrix = np.zeros((3,3))
    for space_id in xrange(1):
        for indices_case in xrange(3):
            for z_id in xrange(3):
                for mcut_id in xrange(N_masscut[z_id]):
                    fof_ifile = fof_idir +'{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], pindices_list[indices_case])
                    reduced_chi2 = np.genfromtxt(fof_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                    print(reduced_chi2)
                    chi2_matrix[indices_case][z_id][mcut_id] = reduced_chi2
                sub_ifile = sub_idir +'{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], pindices_list[indices_case])
                sub_reduced_chi2 = np.genfromtxt(sub_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                sub_chi2_matrix[indices_case][z_id] = sub_reduced_chi2
    print(chi2_matrix)
    figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2.pdf'.format(rec_fprefix[rec_id], sim_space[space_id])
    plot_reduced_chi2(bias_array, chi2_matrix, bias_0, sub_chi2_matrix, odir, figname_chi2)

# show FoF P(k, mu) observed and fitted from the model
def show_fof_P_kmu():
    ##fof_idir = './{}_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(sim_run, rec_dirs[rec_id])
    fof_idir = './{}_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(sim_run, rec_dirs[rec_id])
    ##params_idir = './{}_{}_params_fitted_mean_dset/'.format(sim_run, rec_dirs[rec_id])
    ##params_idir = './{}_{}_params_fitted_mean_dset_fofPkobs_subPktrue/'.format(sim_run, rec_dirs[rec_id])
    ##params_idir = './params_true_mc_fitted_mean_ksorted_mu_masscut/' # parameters fitted from the model with mode-coupling constant, fit P(k,mu) without shot noise
    ##params_idir = './params_obs_mc_fitted_mean_ksorted_mu_masscut/'  # with mc constant, fit P(k,mu) with shot noise
    params_idir = './params_ALL_wnw_diff_fitted_mean_dset/' # parameter fitted for (P_wig-P_now)/(G^2b^2P_sm), FoF P(k, mu) with shot noise included.
    ##Cov_Pk_idir = './{}_Cov_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(sim_run, rec_dirs[rec_id])
    Cov_Pk_idir = './{}_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(sim_run, rec_dirs[rec_id])
    ##odir = './{}_figs_barn_mean/'.format(rec_dirs[rec_id])
    odir = './{}_figs_fofPkobs_subPktrue/'.format(rec_dirs[rec_id])
    ##odir = './{}_figs_barn_mean_mc/'.format(rec_dirs[rec_id])
    indices_p = indices_p1
    k_p = np.array(k_sorted[indices_p])
    mu_p = np.array(mu_sorted[indices_p])
    mu_boundary = np.round([min(mu_p), max(mu_p)], decimals=1)
    print(mu_boundary)
    n_fitbin = np.size(mu_p)
    for space_id in xrange(1):
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                ##fname = '{}kave{}.wnw_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id]) # for mean P(k, \mu)
                fname = '{}kave{}.wnw_diff_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id]) # wnw diff power spectrum
                fof_ifile = fof_idir +fname + '.dat'
                print(fof_ifile)
                k, mu, P_kmu_all = np.loadtxt(fof_ifile, dtype='f8', comments='#', unpack=True)
                ##P_kmu_obs = P_kmu_all[indices_p] - 1.0
                P_kmu_obs = np.array(P_kmu_all[indices_p])  # for Pk wnw difference, no need to use "-1.0" term
                #plot_P_kmu(k, P_kmu_obs, odir, space_id, z_id, mcut_id, fname)

                ifile_params = params_idir + fname + '_params.dat'
                params_err = np.loadtxt(ifile_params, dtype='f4', comments='#', usecols=(0,))
                P_kmu_fit = fit_obs(params_err, k_p, mu_p, tck_Pk_wovers)
                ##P_kmu_fit = fit_obs_mc(params_err, k_p, mu_p, tck_Pk_wovers, tck_Pk_sm)
                ifile_cov_Pk_obs = Cov_Pk_idir + fname + '.dat'
                print(ifile_cov_Pk_obs)
                Cov_Pk_wnw = np.loadtxt(ifile_cov_Pk_obs, dtype='f4', comments='#')
                var_Pk_wnw_mean = np.diag(Cov_Pk_wnw)[indices_p]/N_dset
                
                chi_square = sum((P_kmu_fit-P_kmu_obs)**2.0/var_Pk_wnw_mean)
                reduced_chi2 = chi_square/float(n_fitbin-N_params)
                print('Reduced chi^2: ', reduced_chi2)
                figname = fname+"_fofPkobs"
                ##figname = fname+"_fofPktrue"
                ##figname = "mc_"+fname+"_fofPkobs"
                ##figname = "mc_"+fname+"_fofPktrue"
                overplot_fit_obs(k_p, k_smooth, P_kmu_obs, P_kmu_fit, Pk_wovers, var_Pk_wnw_mean, space_id, z_id, mcut_id, reduced_chi2, odir, figname, mu_boundary)

# show Subsample P(k, mu) observed and fitted from the model
def show_Subsample_P_kmu():
    sub_idir = './{}_sub_Pk_2d_wnw_mean_{}_ksorted_mu/'.format(sim_run, rec_dirs[rec_id])
    params_idir = './{}_sub_{}_params_fitted_mean_dset/'.format(sim_run, rec_dirs[rec_id])
    Cov_Pk_idir = './{}_sub_Cov_Pk_2d_wnw_mean_{}_ksorted_mu/'.format(sim_run, rec_dirs[rec_id])
    odir = './{}_figs_fofPkobs_subPktrue/'.format(rec_dirs[rec_id])
    indices_p = indices_p1
    k_p = np.array(k_sorted[indices_p])
    mu_p = np.array(mu_sorted[indices_p])
    mu_boundary = np.round([min(mu_p), max(mu_p)], decimals=1)
    print(mu_boundary)
    n_fitbin = np.size(mu_p)
    for space_id in xrange(1):
        for z_id in xrange(3):
            fname = '{}kave{}.wnw_mean_sub_a_{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id]) # for mean P(k, \mu)
            sub_ifile = sub_idir +fname + '.dat'
            print(sub_ifile)
            k, mu, P_kmu_obs = np.loadtxt(sub_ifile, dtype='f8', comments='#', unpack=True)
            P_kmu_obs = P_kmu_obs[indices_p] - 1.0
            #plot_P_kmu(k, P_kmu_obs, odir, space_id, z_id, mcut_id, fname)
            
            ifile_params = params_idir + fname + '_params.dat'
            params_err = np.loadtxt(ifile_params, dtype='f4', comments='#', usecols=(0,))
            P_kmu_fit = fit_obs(params_err, k_p, mu_p, tck_Pk_wovers)
            ifile_cov_Pk_obs = Cov_Pk_idir + fname + '.dat'
            print(ifile_cov_Pk_obs)
            Cov_Pk_wnw = np.loadtxt(ifile_cov_Pk_obs, dtype='f4', comments='#')
            var_Pk_wnw_mean = np.diag(Cov_Pk_wnw)[indices_p]/N_dset
            
            chi_square = sum((P_kmu_fit-P_kmu_obs)**2.0/var_Pk_wnw_mean)
            reduced_chi2 = chi_square/float(n_fitbin-N_params)
            print('Reduced chi^2: ', reduced_chi2)
            figname = fname+"_subPktrue"
            overplot_fit_obs(k_p, k_smooth, P_kmu_obs, P_kmu_fit, Pk_wovers, var_Pk_wnw_mean, space_id, z_id, reduced_chi2, odir, figname, mu_boundary)


def main():
    t0 = time.clock()
    show_params_fitted()
    #show_reduced_chi2()
    #compare_sigma_errs()
    #show_fof_P_kmu()
    #show_Subsample_P_kmu()
    t1 = time.clock()
    print(t1-t0)

# to call the main() function to begin the program.
if __name__ == '__main__':
    main()
