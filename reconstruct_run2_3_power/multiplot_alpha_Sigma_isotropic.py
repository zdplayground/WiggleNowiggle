#! ~/vert/bin/python
# Copied the code from multiplot_bscale_mean.py, 08/20/2016.
# 1. Plot alpha, Sigma, b_0, b_scale in terms of bias fitted for the mean power spectra by the isotropic model, i.e.,
# P_wig - P_now = G^2 b^2 (P_lin - P_sm) C^2_G (1+b_scale *k*k)
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
rec_id = 1                 # 0: no reconstruction; 1: with reconstruction

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
#bias_0 = np.array([0.882, 1.197, 1.481])
bias_0 = np.array([1.0, 1.0, 1.0])
omega_m0 = 0.3075
omega_l = 0.6925


idir = '/Users/ding/Documents/playground/WiggleNowiggle/reconstruct_run2_3_power/'

inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

Pk_wovers = Pk_wiggle/Pk_smooth
tck_Pk_wovers = interpolate.splrep(k_smooth, Pk_wovers)
tck_Pk_sm = interpolate.splrep(k_smooth, Pk_smooth)


#----- select a certain \mu bin -------#
indices_p1, indices_p2, indices_p3 = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
ifile = idir+'{}_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/{}kaver.wnw_mean_fof_a_0.5000_mcut34.dat'.format(sim_run, rec_dirs[1], rec_fprefix[1])
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
def plot_alpha_fofsub_bias(bias_array, y1, bias_0, sub_y1, N_masscut, space_id, sim_space, title_space, odir, figname):
    ylim_list = np.array([[-0.2, 1.4], [-0.4, 0.4]])
    n0, n1, n2 = N_masscut[0], N_masscut[1], N_masscut[2]
    print(n0, n1, n2)
    print(bias_array[0][:], y1[0:n0, 0])
    plt.errorbar(bias_array[0][:], (y1[0:n0, 0]-1.0)*100, yerr=[y1[0:n0, 2]*100, y1[0:n0, 1]*100], fmt='k*-')
    plt.errorbar(bias_array[1][:], (y1[n0 : n0+n1, 0]-1.0)*100, yerr=[y1[n0 : n0+n1, 2]*100, y1[n0 : n0+n1, 1]*100], fmt='ro-')
    plt.errorbar(bias_array[2][:], (y1[n0+n1 : n0+n1+n2, 0]-1.0)*100, yerr=[y1[n0+n1 : n0+n1+n2, 2]*100, y1[n0+n1 : n0+n1+n2, 1]*100], fmt='bv-')

    plt.errorbar(bias_0[0], (sub_y1[0, 0]-1.0)*100, yerr=[[sub_y1[0, 2]*100], [sub_y1[0, 1]*100]], fmt='ks')
    plt.errorbar(bias_0[1], (sub_y1[1, 0]-1.0)*100, yerr=[[sub_y1[1, 2]*100], [sub_y1[1, 1]*100]], fmt='rs')
    plt.errorbar(bias_0[2], (sub_y1[2, 0]-1.0)*100, yerr=[[sub_y1[2, 2]*100], [sub_y1[2, 1]*100]], fmt='bs')

    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$\alpha-1.0$ (%)', fontsize=24)
    textline = r"solid lines: FoF $\alpha$"+"\n"+\
               r"square dots: Subsample $\alpha$"+\
               r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(0.6, 0.9, textline)
    plt.xlim([0.5, 4.2])
    plt.ylim([ylim_list[rec_id, 0], ylim_list[rec_id, 1]])
    plt.title(r"Fitted $\alpha$ %s" %(title_space[space_id]), fontsize=16)

    plt.savefig(odir+figname)
    plt.show()
    plt.close()

# similar with the function plot_alpha_fofsub_bias, but for parameter Sigma_xy, Sigma_z
def plot_Sigma_fofsub_bias(bias_array, y1, bias_0, sub_y1, N_masscut, space_id, sim_space, title_space, sim_a, omega_m0, omega_l, odir, figname):
    ylim_list = np.array([[3.0, 12.0], [0., 12.0]])
    n0, n1, n2 = N_masscut[:]
    plt.errorbar(bias_array[0][:], y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*-')
    plt.errorbar(bias_array[1][:], y1[n0 : n0+n1, 0], yerr=[y1[n0 : n0+n1, 2], y1[n0 : n0+n1, 1]], fmt='ro-')
    plt.errorbar(bias_array[2][:], y1[n0+n1 : n0+n1+n2, 0], yerr=[y1[n0+n1 : n0+n1+n2, 2], y1[n0+n1 : n0+n1+n2, 1]], fmt='bv-')
    
    plt.errorbar(bias_0[0], sub_y1[0, 0], yerr=[[sub_y1[0, 2]], [sub_y1[0, 1]]], fmt='ks')
    plt.errorbar(bias_0[1], sub_y1[1, 0], yerr=[[sub_y1[1, 2]], [sub_y1[1, 1]]], fmt='rs')
    plt.errorbar(bias_0[2], sub_y1[2, 0], yerr=[[sub_y1[2, 2]], [sub_y1[2, 1]]], fmt='bs')

    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$\Sigma$ $[Mpc/h]$', fontsize=24)
    textline = r"solid lines: FoF $\Sigma$"+"\n"+\
               r"square dots: Subsample $\Sigma$"+"\n"+\
               r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    
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
    plt.title(r"Fitted $\Sigma$ %s" %(title_space[space_id]), fontsize=16)

    plt.savefig(odir+figname)
    plt.show()
    plt.close()

# Compare b_0 from the simulation with that from the calculation in low k region
def plot_b_0_fofsub(bias_array, y1, sub_bias_0, sub_y1, odir, figname):
    n0, n1, n2 = N_masscut[:]
    plt.errorbar(bias_array[0][:], y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*-')
    plt.errorbar(bias_array[1][:], y1[n0 : n0+n1, 0], yerr=[y1[n0 : n0+n1, 2], y1[n0 : n0+n1, 1]], fmt='ro-')
    plt.errorbar(bias_array[2][:], y1[n0+n1 : n0+n1+n2, 0], yerr=[y1[n0+n1 : n0+n1+n2, 2], y1[n0+n1 : n0+n1+n2, 1]], fmt='bv-')
    
    plt.errorbar(bias_0[0], sub_y1[0, 0], yerr=[[sub_y1[0, 2]], [sub_y1[0, 1]]], fmt='ks')
    plt.errorbar(bias_0[1], sub_y1[1, 0], yerr=[[sub_y1[1, 2]], [sub_y1[1, 1]]], fmt='rs')
    plt.errorbar(bias_0[2], sub_y1[2, 0], yerr=[[sub_y1[2, 2]], [sub_y1[2, 1]]], fmt='bs')
    diag_x = np.linspace(0, 4.5, 100)
    plt.plot(diag_x, diag_x, '--')
    plt.xlabel('bias', fontsize=24)
    plt.ylabel(r'$b_0$', fontsize=24)
    plt.xlim([0., 4.5])
    plt.ylim([0., 4.5])
    textline = r"solid lines: FoF"\
               "\n"+r"square dots: Subsample"+ "\n"\
                r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(0.5, 2.5, textline, fontsize=16)
    plt.savefig(odir + figname)
    plt.show()
    plt.close()

# Plot b_scale from the fitting code mcmc_fit_wig-now_b_scale.py
def plot_b_scale_fofsub(bias_array, y1, bias_0, sub_y1, odir, figname):
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
    plt.text(.7, -5.0, textline, fontsize=16)
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
#
#    plt.plot(bias_array[0][:], y1[2, 0, :], 'k*-')
#    plt.plot(bias_array[1][:], y1[2, 1, :], 'ro-')
#    plt.plot(bias_array[2][:], y1[2, 2, :], 'bv-')

    plt.plot(bias_0[0], sub_y1[0, 0], 'ks', bias_0[0], sub_y1[0, 1], 'rs', bias_0[0], sub_y1[0, 2], 'bs')
    plt.plot(bias_0[0], sub_y1[1, 0], 'kp', bias_0[0], sub_y1[1, 1], 'rp', bias_0[0], sub_y1[1, 2], 'bp')
#    plt.plot(bias_0[0], sub_y1[2, 0], 'kh', bias_0[0], sub_y1[2, 1], 'rh', bias_0[0], sub_y1[2, 2], 'bh')
    plt.xlabel('bias', fontsize=24)
    plt.ylim([1.05, 1.30])
    plt.ylabel(r'$\Delta \chi^2$', fontsize=24)
    textline = r"blue: $z=1.0$"+"\n"+r"red: $z=0.6$"+"\n"+r"black: $z=0.0$"
    plt.text(3.0, 1.17, textline, fontsize=20)
    plt.savefig(odir + figname)
    plt.show()
    plt.close()

#----- part 1: plot parametrs fitted for the mean P(k)
#
def read_params(inputf):
    N_skip_header = 1
    N_skip_footer = 1
    data_m = np.genfromtxt(inputf, dtype=np.float, comments=None, skip_header=N_skip_header, skip_footer=N_skip_footer, usecols=(1, 2, 3))
    return data_m

def show_params_fitted():
    # 1 represents true as a free parameter, 0 is false. Total parameters are alpha_perp, alpha_para, \Sigam_xy, \Sigma_z, b_0, b_scale
    params_indices = np.array([0, 1, 0, 1, 1, 1])
    #params_indices = np.array([0, 1, 0, 0, 1, 1])
    indices_str = ''.join(map(str, params_indices))

    #fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    
    for space_id in xrange(1):
        alpha_m = np.empty((0,3), dtype='float')
        sub_alpha_m = np.empty((0,3), dtype='float')
        Sigma_m = np.empty((0,3), dtype='float')
        sub_Sigma_m = np.empty((0,3), dtype='float')
        b_0m, b_scalem = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        sub_b_0m, sub_b_scalem = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], indices_str)
                print(fof_ifile)
                ##params_err = np.loadtxt(fof_ifile, dtype=float, comments='#')
                params_err = read_params(fof_ifile)
                alpha_m = np.vstack((alpha_m, params_err[0, :]))
                Sigma_m = np.vstack((Sigma_m, params_err[1, :]))
                b_0m = np.vstack((b_0m, params_err[2, :]))
                b_scalem = np.vstack((b_scalem, params_err[3, :]))
#                b_0m = np.vstack((b_0m, params_err[1, :]))
#                b_scalem = np.vstack((b_scalem, params_err[2, :]))


            sub_ifile = sub_idir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], indices_str)
            print(sub_ifile)
            
            ##params_err = np.loadtxt(sub_ifile, dtype='float64', comments='#')
            params_err = read_params(sub_ifile)
            sub_alpha_m = np.vstack((sub_alpha_m, params_err[0, :]))
            sub_Sigma_m = np.vstack((sub_Sigma_m, params_err[1, :]))
            sub_b_0m = np.vstack((sub_b_0m, params_err[2, :]))
            sub_b_scalem = np.vstack((sub_b_scalem, params_err[3, :]))
#            sub_b_0m = np.vstack((sub_b_0m, params_err[1, :]))
#            sub_b_scalem = np.vstack((sub_b_scalem, params_err[2, :]))


        figname_alpha = '{}kave{}_wig-now_mean_fof_sub_alpha_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_Sigma = '{}kave{}_wig-now_mean_fof_sub_Sigma_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_0 = '{}kave{}_wig-now_mean_fof_sub_b0_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_scale = '{}kave{}_wig-now_mean_fof_sub_bscale_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)

    
        plot_alpha_fofsub_bias(bias_array, alpha_m, bias_0, sub_alpha_m, N_masscut, space_id, sim_space, title_space, odir, figname_alpha)
#        plot_Sigma_fofsub_bias(bias_array, Sigma_m, bias_0, sub_Sigma_m, N_masscut, space_id, sim_space, title_space, sim_a, omega_m0, omega_l, odir, figname_Sigma)
        plot_b_0_fofsub(bias_array, b_0m, bias_0, sub_b_0m, odir, figname_b_0)
        plot_b_scale_fofsub(bias_array, b_scalem, bias_0, sub_b_scalem, odir, figname_b_scale)

# define a function to show parameters alpha and b_scale fitted from the case \Sigma is obtained from theoretical equation given by Zvonimir.
def show_params_fitted_Sigma_Theory():
    # 1 represents true as a free parameter, 0 represents a fixed parameter. Parameters are alpha_perp, alpha_para, \Sigam_xy, \Sigma_z, b_0, b_scale
    ##params_indices = np.array([0, 1, 0, 0, 1, 1])
    params_indices = np.array([0, 0, 0, 0, 1, 1])
    indices_str = ''.join(map(str, params_indices))
    
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_Theory_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale_Sigma_Theory/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    
    for space_id in xrange(1):
        alpha_m = np.empty((0,3), dtype='float')
        sub_alpha_m = np.empty((0,3), dtype='float')
        Sigma_m = np.empty((0,3), dtype='float')
        sub_Sigma_m = np.empty((0,3), dtype='float')
        b_0m, b_scalem = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        sub_b_0m, sub_b_scalem = np.empty((0,3), dtype='float'), np.empty((0,3), dtype='float')
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                
                fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], indices_str)
                sub_ifile = sub_idir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], indices_str)
                params_err = read_params(fof_ifile)
#                alpha_m = np.vstack((alpha_m, params_err[0, :]))
#                b_0m = np.vstack((b_0m, params_err[1, :]))
#                b_scalem = np.vstack((b_scalem, params_err[2, :]))
                b_0m = np.vstack((b_0m, params_err[0, :]))
                b_scalem = np.vstack((b_scalem, params_err[1, :]))
            print(sub_ifile)
            
            params_err = read_params(sub_ifile)
#            sub_alpha_m = np.vstack((sub_alpha_m, params_err[0, :]))
#            sub_b_0m = np.vstack((sub_b_0m, params_err[1, :]))
#            sub_b_scalem = np.vstack((sub_b_scalem, params_err[2, :]))

            sub_b_0m = np.vstack((sub_b_0m, params_err[0, :]))
            sub_b_scalem = np.vstack((sub_b_scalem, params_err[1, :]))
        
        figname_alpha = '{}kave{}_wig-now_mean_fof_sub_alpha_params{}_isotropic_Sigma_Theory.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_0 = '{}kave{}_wig-now_mean_fof_sub_b0_params{}_isotropic_Sigma_Theory.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_scale = '{}kave{}_wig-now_mean_fof_sub_bscale_params{}_isotropic_Sigma_Theory.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        
#        plot_alpha_fofsub_bias(bias_array, alpha_m, bias_0, sub_alpha_m, N_masscut, space_id, sim_space, title_space, odir, figname_alpha)
        plot_b_0_fofsub(bias_array, b_0m, bias_0, sub_b_0m, odir, figname_b_0)
        plot_b_scale_fofsub(bias_array, b_scalem, bias_0, sub_b_scalem, odir, figname_b_scale)

# show reduced \chi^2 from different numbers of fitting parameters
def show_reduced_chi2():
    pindices_list = ['010111']
    #pindices_list = ['010011']
    N_skip_header_list = [5]
#    N_skip_header_list = [4]
    #fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale/'.format(rec_dirs[rec_id])
    chi2_matrix = np.zeros((1,3,5))
    sub_chi2_matrix = np.zeros((1,3))
    for space_id in xrange(1):
        for indices_case in xrange(1):
            for z_id in xrange(3):
                for mcut_id in xrange(N_masscut[z_id]):
                    fof_ifile = fof_idir +'{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], pindices_list[indices_case])
                    reduced_chi2 = np.genfromtxt(fof_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                    print(reduced_chi2)
                    chi2_matrix[indices_case][z_id][mcut_id] = reduced_chi2
                sub_ifile = sub_idir +'{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], pindices_list[indices_case])
                sub_reduced_chi2 = np.genfromtxt(sub_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                sub_chi2_matrix[indices_case][z_id] = sub_reduced_chi2
    print(chi2_matrix)
    figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], pindices_list[indices_case])
    plot_reduced_chi2(bias_array, chi2_matrix, bias_0, sub_chi2_matrix, odir, figname_chi2)

# show reduced \chi^2 from different numbers of fitting parameters
def show_reduced_chi2_Sigma_Theory():
    #Sigma_array = [7.836, 5.749, 4.774]
    pindices_list = ['000011', '010011']
    N_skip_header_list = [3, 4]
    #pindices_list = ['000011']
    #N_skip_header_list = [3]
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_Theory_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale_Sigma_Theory/'.format(rec_dirs[rec_id])
    chi2_matrix = np.zeros((2,3,5))
    sub_chi2_matrix = np.zeros((2,3))
    for space_id in xrange(1):
        for indices_case in xrange(2):
            for z_id in xrange(3):
                for mcut_id in xrange(N_masscut[z_id]):
                    fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], pindices_list[indices_case])
                    reduced_chi2 = np.genfromtxt(fof_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                    print(reduced_chi2)
                    chi2_matrix[indices_case][z_id][mcut_id] = reduced_chi2
                sub_ifile = sub_idir +'{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], pindices_list[indices_case])
                sub_reduced_chi2 = np.genfromtxt(sub_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                sub_chi2_matrix[indices_case][z_id] = sub_reduced_chi2
    print(chi2_matrix)
    ##figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], pindices_list[indices_case])
    ##figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2_params{}_isotropic_Sigma.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], pindices_list[indices_case])
    figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2_params{}_{}_isotropic_Sigma.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], pindices_list[0], pindices_list[1])
    plot_reduced_chi2(bias_array, chi2_matrix, bias_0, sub_chi2_matrix, odir, figname_chi2)

def main():
    t0 = time.clock()
    #show_params_fitted()
    #show_reduced_chi2()
    #show_params_fitted_Sigma_Theory()
    show_reduced_chi2_Sigma_Theory()
    t1 = time.clock()
    print(t1-t0)

# to call the main() function to begin the program.
if __name__ == '__main__':
    main()
