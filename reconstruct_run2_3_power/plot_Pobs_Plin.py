#!/usr/bin/python
# made on 07/08/2016, plot P_obs/P_lin with mode coupling constant term
# The constant A fitted is the
from __future__ import print_function
import numpy as np
from scipy import interpolate
from scipy import special
import time
import math
import os
import matplotlib.pyplot as plt

Omega_m = 0.3075
sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = [['N1', 'W1'], ['N3', 'W3']] # NW, WG for run2; N3, W3 for run3.
sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space
space_title = ['in real space', 'in redshift space']
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']
wig_types = ['now', 'wig']
Pk_type = ['obs', 'true']

mcut_Npar_list = [[37, 149, 516, 1524, 3830],
                  [35, 123, 374, 962, 2105],
                  [34, 103, 290, 681, 1390]]
N_masscut = np.size(mcut_Npar_list, axis=1)

bias_list = np.array([[0.90, 1.09, 1.40, 1.81, 2.33],
                      [1.21, 1.52, 1.96, 2.53, 3.20],
                      [1.48, 1.88, 2.41, 3.07, 3.87]])

# define growth factor G(z)
def growth_factor(z, Omega_m):
    a = 1.0/(1.0+z)
    v = (1.0+z)*(Omega_m/(1.0-Omega_m))**(1.0/3.0)
    phi = math.acos((v+1.0-3.0**0.5)/(v+1.0+3.0**0.5))
    m = (math.sin(75.0/180.0* math.pi))**2.0
    part1c = 3.0**0.25 * (1.0+ v**3.0)**0.5
    # first elliptic integral
    F_elliptic = special.ellipkinc(phi, m)
    # second elliptic integral
    Se_elliptic = special.ellipeinc(phi, m)
    part1 = part1c * ( Se_elliptic - 1.0/(3.0+3.0**0.5)*F_elliptic)
    part2 = (1.0 - (3.0**0.5 + 1.0)*v*v)/(v+1.0+3.0**0.5)
    d_1 = 5.0/3.0*v*(part1 + part2)
    # if a goes to 0, use d_11, when z=1100, d_1 is close to d_11
    #    d_11 = 1.0 - 2.0/11.0/v**3.0 + 16.0/187.0/v**6.0
    return a*d_1
G_0 = growth_factor(0.0, Omega_m) # G_0 at z=0, normalization factor

idir0 = '/Users/ding/Documents/playground/WiggleNowiggle/reconstruct_run2_3_power/'

inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

tck_Pk_smooth = interpolate.splrep(k_smooth, Pk_smooth, k=3)   # k=3, use cubic spline to fit
print(interpolate.splev(k_smooth, tck_Pk_smooth, der=0))
inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

Pk_wovers = Pk_wiggle/Pk_smooth

# define a function plotting P(k) from simulation and from theory
def plot_Psim_Pthe(k_obs, v1, v2, v3, rec_id, z_id, mcut_id, space_id, Pt_id, odir):
    plt.plot(k_obs, v1, 'b+', label=r"$\frac{P_{now}}{G^2b^2 P_{linnow}}$")
    plt.plot(k_obs, v2, 'g*', label=r"$\frac{P_{now}-P_{shot}}{G^2b^2 P_{linnow}}$")
    plt.plot(k_obs, v3, 'r-', label=r"$\frac{P_{now}-A}{G^2b^2 P_{linnow}}$")
    plt.legend(loc="upper left", frameon=False)
    plt.title("z={}, bias={} {}".format(sim_z[z_id], bias_list[z_id][mcut_id], space_title[space_id]))
    plt.xlabel(r"$k$ $[h/Mpc]$", fontsize=18)
    plt.ylabel(r"$P_{simulation}/P_{theory}$", fontsize=18)
    figname = odir+"fof_Psim_Pthe_{}_{}kave{}_{}_a_{}_mcut{}.pdf".format(Pk_type[Pt_id], rec_fprefix[rec_id], sim_space[space_id], wig_types[wtype_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
    plt.savefig(figname)
    plt.show()
    plt.close()

def plot_sub_Pk_wnw_diff(k_obs, Pk_wnw_diff, Pk_wnw_ratio, rec_id, z_id, space_id, odir):
    plt.plot(k_obs, Pk_wnw_diff, 'b-',label=r"$(P_{obsw}-P_{obsnw})/b^2G^2P_{linnw}$")
    plt.plot(k_obs, Pk_wnw_ratio, 'r-.', label=r"$P_{obsw}/P_{obsnw}-1$")
    plt.legend(loc='best', fontsize=14, frameon=False)
    plt.xlabel(r"$k$ $[h/Mpc]$", fontsize=18)
    plt.title("Subsample z={} {}".format(sim_z[z_id], space_title[space_id]), fontsize=18)
    plt.ylim([-0.08, 0.08])
    plt.savefig("sub_Pk_wnw_diff_{}kave{}_a_{}.pdf".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id]))
    plt.show()
    plt.close()

#-- copy the function from ../../reconstruct_run_2_3_power/plot_Pobs_Plin.py; plotting the fitted const parameter
def plot_mc_const(bias_array, mc_const_array, mc_const_uperrs, mc_const_downerrs, shotnoise_array, odir):
    plt.errorbar(bias_array[0, :], mc_const_array[0][0][:], yerr=[mc_const_downerrs[0][0][:], mc_const_uperrs[0][0][:]], fmt='k*-', label=r'$z=0$')
    plt.errorbar(bias_array[1, :], mc_const_array[0][1][:], yerr=[mc_const_downerrs[0][1][:], mc_const_uperrs[0][1][:]], fmt='ro-', label=r'$z=0.6')
    plt.errorbar(bias_array[2, :], mc_const_array[0][2][:], yerr=[mc_const_downerrs[0][2][:], mc_const_uperrs[0][2][:]], fmt='bv-', label=r'$z=1.0')
    
    plt.errorbar(bias_array[0, :], mc_const_array[1][0][:]+shotnoise_array[0][:], yerr=[mc_const_downerrs[1][0][:], mc_const_uperrs[1][0][:]], fmt='k*--')
    plt.errorbar(bias_array[1, :], mc_const_array[1][1][:]+shotnoise_array[1][:], yerr=[mc_const_downerrs[1][1][:], mc_const_uperrs[1][1][:]], fmt='ro--')
    plt.errorbar(bias_array[2, :], mc_const_array[1][2][:]+shotnoise_array[2][:], yerr=[mc_const_downerrs[1][2][:], mc_const_uperrs[1][2][:]], fmt='bv--')
    plt.yscale("log")
    plt.ylim([1, 1.e5])
    textline = "black: z=0; red: z=0.6; blue: z=1.0\n"
    textline = textline + r"solid lines: fit $\hat{P}_{\mathtt{wig}}/\hat{P}_{\mathtt{now}}$;"+"\n"
    ##textline = textline + r"dashed lines: fit $(\hat{P}_{\mathtt{wig}}-1/n)/(\hat{P}_{\mathtt{now}}-1/n)$"
    textline = textline + r"dashed lines: fit $(\hat{P}_{\mathtt{wig}}-1/n)/(\hat{P}_{\mathtt{now}}-1/n)$"
    plt.text(0.8, 15000, textline, fontsize=14)
    plt.grid(True)
    plt.xlabel('bias', fontsize=24)
    ##plt.ylabel(r'$A$ $[Mpc^3/h^3]$', fontsize=14)
    plt.ylabel(r'$A,$ $A+1/n$ $[Mpc^3/h^3]$', fontsize=24)
    figname = odir+"fof_const_mc.pdf"
    plt.savefig(figname)
    plt.show()
    plt.close()


#wtype_id = 0
Pt_id = 1      # 0: fit observed P(k); 1: fit true P(k)
params_fit_dir = 'params_{}_mc_fitted_mean_ksorted_mu_masscut/'.format(Pk_type[Pt_id])
odir = 'figs_Pk_sim_theory/'
if not os.path.exists(odir):
    os.makedirs(odir)

#---- plot Pk from theory and observation ----
def show_Pk_obs_the():
    rec_id = 1
    Pk_idir = 'run2_3_fof_Pk_obs_true_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    shotnoise_array = np.zeros((3, N_masscut))
    for space_id in xrange(1):
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut):
                Pk_filename = '{}kave{}.{}_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_types[0], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                inputf = idir0 + Pk_idir + Pk_filename
                print(inputf)
                k_obs, Pk_now_obs, Pk_now_true = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0, 2, 3), unpack=True)
                #print('now shot noise: ', Pk_now_obs-Pk_now_true)
                Pk_sm_obsk = interpolate.splev(k_obs, tck_Pk_smooth, der=0)    # interpolate theoretical linear P(k) at points k_obs
                # read Pk_wig files
                Pk_filename = '{}kave{}.{}_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_types[1], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                inputf = idir0 + Pk_idir + Pk_filename
                print(inputf)
                k_obs, Pk_wig_obs, Pk_wig_true = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0, 2, 3), unpack=True)
                #print('wig shot noise: ', Pk_wig_obs-Pk_wig_true)
                shotnoise_array[z_id][mcut_id] = Pk_wig_obs[0]-Pk_wig_true[0]
                
                bias = bias_list[z_id][mcut_id]
                z = float(sim_z[z_id])
                print("z=", z, "G: ", growth_factor(z, Omega_m)/G_0)
                G2b2 = (growth_factor(z, Omega_m)/G_0*bias)**2.0
                Pk_ggrow = Pk_sm_obsk * G2b2
                Pk_wnw_diff = (Pk_wig_true - Pk_now_true)/Pk_ggrow
                
                params_fit_filename = '{}kave{}.wnw_mean_fof_a_{}_mcut{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                params_fit_file = idir0 + params_fit_dir + params_fit_filename
                Pshot_fit = np.genfromtxt(params_fit_file, dtype=float, comments='#', skip_header=5, usecols=(0,))
                Pshot_fit = Pshot_fit * G2b2            # the mode coupling constant contains G^2b^2 in the denominator, so here times the factor back.
                print(Pshot_fit)
                
                v1 = Pk_now_obs/Pk_ggrow
                v2 = Pk_now_true/Pk_ggrow
                ##v3 = (Pk_now_obs- Pshot_fit)/Pk_ggrow
                v3 = (Pk_now_true - Pshot_fit)/Pk_ggrow   # for Pt_id=1
#                plot_Psim_Pthe(k_obs, v1, v2, v3, rec_id, z_id, mcut_id, space_id, Pt_id, odir)
#                plot_Pk_wnw_diff(k_obs, Pk_wnw_diff, rec_id, z_id, mcut_id, space_id, odir)
        ofile = Pk_idir+'{}kave{}_shotnoise_mean_wig_mcut.out'.format(rec_fprefix[rec_id], sim_space[space_id])
        header_line = 'Row(z: 0, 0.6, 1.0)   Column(mcut: low to high mass cuts)'
        np.savetxt(ofile, shotnoise_array, fmt='%.7e', newline='\n', comments='#', header = header_line)

#---- plot fitted parameter A (mc_const) in both cases, fitting the observed and true P(k) from simulations.
def show_mc_const():
    rec_id = 1
    mc_const_array = np.zeros((2,3,5))
    mc_const_uperrs = np.zeros((2,3,5))
    mc_const_downerrs = np.zeros((2,3,5))
    for Pt_id in xrange(2):
        for z_id in xrange(3):
            z = float(sim_z[z_id])
            print("z=", z, "G: ", growth_factor(z, Omega_m)/G_0)
            for mcut_id in xrange(N_masscut):
                bias = bias_list[z_id][mcut_id]
                G2b2 = (growth_factor(z, Omega_m)/G_0*bias)**2.0
                for space_id in xrange(1):
                    params_fit_dir = 'params_{}_mc_fitted_mean_ksorted_mu_masscut/'.format(Pk_type[Pt_id])
                    params_fit_filename = '{}kave{}.wnw_mean_fof_a_{}_mcut{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                    params_fit_file = idir0 + params_fit_dir + params_fit_filename
                    mc_const, mc_const_upsigma, mc_const_downsigma = np.genfromtxt(params_fit_file, dtype=float, comments='#', skip_header=5, unpack=True)
                    mc_const_array[Pt_id][z_id][mcut_id] = mc_const * G2b2  # A' from fitting equals A/(G^2b^2) in the fitting model. We want to get A.
                    mc_const_uperrs[Pt_id][z_id][mcut_id] = mc_const_upsigma * G2b2
                    mc_const_downerrs[Pt_id][z_id][mcut_id] = mc_const_downsigma * G2b2
    print(mc_const_array)
    shotnoise_idir = 'run2_3_fof_Pk_obs_true_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    ifile = shotnoise_idir + '{}kave{}_shotnoise_mean_wig_mcut.out'.format(rec_fprefix[rec_id], sim_space[space_id])
    shotnoise_array = np.loadtxt(ifile, dtype='f4', comments='#')
    #print(shotnoise_array)
    plot_mc_const(bias_list, mc_const_array, mc_const_uperrs, mc_const_downerrs, shotnoise_array, odir)

#------
def show_sub_Pk_obs_the():
    rec_id = 1
    Pk_idir = 'run2_3_sub_Pk_obs_true_2d_wnw_mean_{}_ksorted_mu/'.format(rec_dirs[rec_id])
    bias = 1.0
    for z_id in xrange(3):
        z = float(sim_z[z_id])
        print("z=", z, "G: ", growth_factor(z, Omega_m)/G_0)
        G2b2 = (growth_factor(z, Omega_m)/G_0*bias)**2.0
        for space_id in xrange(1):
            Pk_filename = '{}kave{}.{}_mean_fof_a_{}ga.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_types[0], sim_a[z_id])
            inputf = idir0 + Pk_idir + Pk_filename
            print(inputf)
            k_obs, Pk_now_obs, Pk_now_true = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0, 2, 3), unpack=True)
            Pk_sm_obsk = interpolate.splev(k_obs, tck_Pk_smooth, der=0)    # interpolate theoretical linear P(k) at points k_obs
            # read Pk_wig files
            Pk_filename = '{}kave{}.{}_mean_fof_a_{}ga.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_types[1], sim_a[z_id])
            inputf = idir0 + Pk_idir + Pk_filename
            #print(inputf)
            k_obs, Pk_wig_obs, Pk_wig_true = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0, 2, 3), unpack=True)
            Pk_ggrow = Pk_sm_obsk * G2b2
            Pk_wnw_diff = (Pk_wig_true - Pk_now_true)/Pk_ggrow
            Pk_wnw_ratio = Pk_wig_true/Pk_now_true - 1.0
            #plot_sub_Pk_wnw_diff(k_obs, Pk_wnw_diff, Pk_wnw_ratio, rec_id, z_id, space_id, odir)

def main():
#    show_Pk_obs_the()
    show_mc_const()
#    show_sub_Pk_obs_the()

if __name__ == '__main__':
    main()
