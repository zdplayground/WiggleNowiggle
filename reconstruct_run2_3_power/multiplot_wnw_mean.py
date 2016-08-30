#!/usr/bin/env python
# Copy the same code from the folder reconstruct_run3_power on 07/14/2016, plot mean 1D P(k) of run2 and run3 simulation post reconstruction.
# 1. In Hee-Jong's 1D spherically averaged power spectrum, k_obs is the up boundary of each k bin, modified on 07/01/2016
# 2. Add overplot of run2 and run3 P(k) in redshift space. 07/04/2016
# 3. Rename it from multiplot_mean.py to multiplot_wnw_mean.py, showing it's for plotting parameters or power spectrum of P_wig/P_now. 08/22/2016.
# 4. Show reduced \chi^2 for P_wig/P_now with different cases, including the additional constant A or not. 08/22/2016.
#
from __future__ import print_function
import numpy as np
from scipy import interpolate
import math
import time
import os
import matplotlib.pyplot as plt
import growth_fun
from growth_fun import growth_factor

Omega_m = 0.3075
G_0 = growth_factor(0.0, Omega_m) # G_0 at z=0, normalization factor

sim_run = 'run2_3'
N_dset = 20
N_skip_header = 11

sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = ['NW', 'WG']
sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space
L_box = 1380.0             # box size in the simulation, unit Mpc/h
delta_k = 2.0*np.pi /L_box

sub_sim_space = ['real', 'rsd']
title_space = ['in real space', 'in redshift space']
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']
rec_title = ['Before reconstuction', 'After reconstruction']

mcut_Npar_list = [[37, 149, 516, 1524, 3830],
                  [35, 123, 374, 962, 2105],
                  [34, 103, 290, 681, 1390]]
N_masscut = [5, 5, 5]
# The bias_array value corresponds to the iterms in mcut_Npar_list. If Npar of mass cut point is larger, then bias is larger too.
bias_array = np.array([[0.90, 1.09, 1.40, 1.81, 2.33],
                       [1.21, 1.52, 1.96, 2.53, 3.20],
                       [1.48, 1.88, 2.41, 3.07, 3.87]])

inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
tck_Pk_smooth = interpolate.splrep(k_smooth, Pk_smooth, k=3)   # k=3, use cubic spline to fit

inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

Pk_wovers = Pk_wiggle/Pk_smooth

#----- select a certain \mu bin -------#
ifile = '/Users/ding/Documents/playground/WiggleNowiggle/DESI_Yu/Analysisz0/z0mcut37/ALL/Rkaver.N10_fof_1.0000ga.dat'
data_m = np.genfromtxt(ifile, dtype='f8', comments='#', skip_header=N_skip_header)
num_kbin = np.size(data_m, axis=0)
kk = data_m[:, 0]

indices_sort = [i[0] for i in sorted(enumerate(kk), key=lambda x: x[1])]

for i in xrange(num_kbin):
    if kk[indices_sort[i]] > 0.3:
        break
print(indices_sort[i-1])
ind_up = i
ind_low = 0
n_fitbin = i
print("# of k bin: ", n_fitbin)
indices_p = indices_sort[ind_low: ind_up]

k_sorted, mu_sorted, Nmode_sorted = kk[indices_p], data_m[indices_p, 1], data_m[indices_p, 3]
print(k_sorted, mu_sorted, Nmode_sorted)
indices_p1, indices_p2, indices_p3 = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
for i in xrange(len(mu_sorted)):
    if mu_sorted[i]>=0.0 and mu_sorted[i]<0.1:
        indices_p1 = np.append(indices_p1, i)
    elif mu_sorted[i]>=0.4 and mu_sorted[i]<0.5:
        indices_p2 = np.append(indices_p2, i)
    elif mu_sorted[i]>=0.9 and mu_sorted[i]<1.0:
        indices_p3 = np.append(indices_p3, i)
print(i)
print(sum(map(np.size, [indices_p1, indices_p2, indices_p3])))

# over plot P(k)wig and P(k)now in terms of k
def plot_Pk_wigandnow(k_obs, Pk_now_obs, Pk_wig_obs, odir, space_id, z_id, mcut_id, filename):
    plt.plot(k_obs, Pk_now_obs, '--', label='no-wiggle')
    plt.plot(k_obs, Pk_wig_obs, '-', label='wiggle')
    plt.legend(loc="upper right")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([0.004, 0.3])
    plt.ylim([1.e3, 1.e6])
    plt.xlabel(r"$k$ $[h/Mpc]$")
    plt.ylabel(r"$P(k)_{obs}$")
    plt.title("$z={}$, bias=${}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], title_space[space_id]))
    figname = odir+ "Pk_wigandnow_"+filename + ".pdf"
    plt.savefig(figname)
    #plt.show()
    plt.close()

# plot P(k)_wig/P_now in terms of k
def plot_Pk_wnw(k_obs, Pk_wnw_nn_obs, Pk_wnw_wn_obs, k_true, Pk_true, odir, space_id, z_id, mcut_id, filename):
    plt.figure(figsize=(10., 6.5))
    plt.plot(k_obs, Pk_wnw_nn_obs, '+-', label='obs no shot noise')
    plt.plot(k_obs, Pk_wnw_wn_obs, '>-.', label='obs with shot noise')
    plt.plot(k_true, Pk_true, '.--', label='Theoretical')
    plt.legend(loc="upper left")
    plt.xscale("log")
    plt.xlim([0.004, 0.3])
    plt.ylim([-0.08, 0.08])
    plt.xlabel(r"$k$ $[h/Mpc]$", fontsize=24)
    plt.ylabel(r"$(P_{\mathrm{wig}}/P_{\mathrm{now}})_{\mathrm{mean}}$", fontsize=24)
    plt.title("$z={}$, bias=${}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], title_space[space_id]),fontsize=20)
    figname = odir+ "Pk_"+filename + ".pdf"
    plt.savefig(figname)
    #plt.show()
    plt.close()

# overplot P(k)_wnw from run2 and run3
#def plot_Pk_wnw_run2_3(k_obs, Pk_r2, Pk_r3, odir, space_id, z_id, mcut_id, filename):
#    plt.plot(k_obs, Pk_r2, '+-', label='run2')
#    plt.plot(k_obs, Pk_r3, '.--', label='run3')
#    plt.legend(loc="upper left")
#    plt.xscale("log")
#    plt.xlim([0.004, 0.3])
#    plt.ylim([-0.08, 0.08])
#    plt.xlabel(r"$k$ $[h/Mpc]$")
#    plt.ylabel(r"$(P_{wig}/P_{now})_{mean}$")
#    plt.title("$z={}$, bias=${}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], title_space[space_id]))
#    figname = odir+ "run2_3_Pk_"+filename + ".pdf"
#    #plt.savefig(figname)
#    plt.show()
#    plt.close()

# overplot P(k) from run2 and run3
def plot_Pk_run2_3(k_obs, Pk_r2, Pk_r3, odir, space_id, z_id, mcut_id, filename):
    plt.plot(k_obs, Pk_r2, '+-', label='run2')
    plt.plot(k_obs, Pk_r3, '.--', label='run3')
    plt.legend(loc="upper left")
    plt.xscale("log")
    plt.xlim([0.004, 0.3])
    #plt.ylim([-0.08, 0.08])
    plt.xlabel(r"$k$ $[h/Mpc]$")
    plt.ylabel(r"$(P(k)_{mean}$")
    plt.title("$z={}$, bias=${}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], title_space[space_id]))
    figname = odir+ "run2_3_Pk_"+filename + ".pdf"
    plt.savefig(figname)
    #plt.show()
    plt.close()

# over plot P(k) from FoF and subsample at one z with various biases
def plot_Pk_fof_sub(k_obs, fof_Pk_mwig_obs, sub_Pk_wig_obs, rec_id, z_id, space_id, odir):
    plt.loglog(k_obs, fof_Pk_mwig_obs[0, :], label='b={}'.format(bias_array[z_id][0]))
    plt.loglog(k_obs, fof_Pk_mwig_obs[1, :], label='b={}'.format(bias_array[z_id][1]))
    plt.loglog(k_obs, fof_Pk_mwig_obs[2, :], label='b={}'.format(bias_array[z_id][2]))
    plt.loglog(k_obs, fof_Pk_mwig_obs[3, :], label='b={}'.format(bias_array[z_id][3]))
    plt.loglog(k_obs, fof_Pk_mwig_obs[4, :], label='b={}'.format(bias_array[z_id][4]))
    plt.loglog(k_obs, sub_Pk_wig_obs, 'k--', label='Subsample')
    plt.ylim([10, 1.e6])
    plt.legend(loc='lower left', fontsize=18)
    plt.xlabel(r'$k$ $[h/Mpc]$', fontsize=18)
    plt.ylabel(r'$P(k)$ $[Mpc^3/h^3]$')
    plt.title('{} $z={}$ {}'.format(rec_title[rec_id], sim_z[z_id], title_space[space_id]))
    figname = '{}pk{}_fof_sub_z_{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], sim_z[z_id])
    plt.savefig(odir+figname)
    #plt.show()
    plt.close()

def show_Pk_run2n3_mean():
    rec_id = 0      # 0: no reconstruction; 1: with reconstruction
    fof_idir = './{}_Pk_1d_obs_true_mean_{}_ksorted_masscut/'.format(sim_run, rec_dirs[rec_id])
    fof_wig_ifile = fof_idir + '{}pk{}.wig_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[0], sim_a[0], mcut_Npar_list[0][0])
    k_obs, Pk_wig_obs = np.loadtxt(fof_wig_ifile, dtype='f8', comments='#', usecols=(0, 1), unpack=True)
    k_obs = k_obs - delta_k/2.0        # k_obs in data file is the up boundary of each k bin
    
    odir = './{}_1d_figs_Pk_mean/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    for space_id in xrange(2):
        for z_id in xrange(3):
            fof_Pk_mwig_obs = np.array([], dtype=np.float64).reshape(0, len(k_obs))
            for mcut_id in xrange(N_masscut[z_id]):
                wnw_fname = '{}pk{}.wnw_obs_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id]) # for mean P(k, \mu)_wnw
                now_fname = '{}pk{}.now_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                wig_fname = '{}pk{}.wig_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                fof_wnw_ifile = fof_idir +wnw_fname + '.dat'
                fof_now_ifile = fof_idir +now_fname + '.dat'
                fof_wig_ifile = fof_idir +wig_fname + '.dat'
                print(fof_wig_ifile)
                k_obsPk_wnw_wn_obs, Pk_wnw_nn_obs = np.loadtxt(fof_wnw_ifile, dtype='f8', comments='#', usecols=(1,2), unpack=True)
                Pk_wnw_wn_obs = Pk_wnw_wn_obs - 1.0
                Pk_wnw_nn_obs = Pk_wnw_nn_obs - 1.0
                Pk_wig_obs = np.loadtxt(fof_wig_ifile, dtype='f8', comments='#', usecols=(1,))
                fof_Pk_mwig_obs = np.vstack([fof_Pk_mwig_obs, Pk_wig_obs])
                plot_Pk_wnw(k_obs, Pk_wnw_nn_obs, Pk_wnw_wn_obs, k_smooth, Pk_wovers-1.0, odir, space_id, z_id, mcut_id, wnw_fname)
        
            sub_wig_ifile = fof_idir + '{}pk{}.wig_mean_sub_a_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            sub_Pk_wig_obs = np.loadtxt(sub_wig_ifile, dtype='f8', comments='#', usecols=(2,))
#            plot_Pk_fof_sub(k_obs, fof_Pk_mwig_obs, sub_Pk_wig_obs, rec_id, z_id, space_id, odir)


                
#                fname = '{}pk{}.mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
#                plot_Pk_wigandnow(k_obs, Pk_now_obs, Pk_wig_obs, odir, space_id, z_id, mcut_id, fname)


def show_Pk_run2_3():
    run2_wnw_idir = '../../reconstruct_run2_power/run2_1D_wrongAnalysis/run2_Pk_1d_wnw_mean_{}_masscut/'.format(rec_dirs[rec_id])
    run2_wig_idir = '../../reconstruct_run2_power/run2_1D_wrongAnalysis/run2_Pk_1d_wig_mean_{}_masscut/'.format(rec_dirs[rec_id])
    run2_now_idir = '../../reconstruct_run2_power/run2_1D_wrongAnalysis/run2_Pk_1d_now_mean_{}_masscut/'.format(rec_dirs[rec_id])
    run3_wnw_idir = './run3_Pk_1d_wnw_mean_{}_masscut/'.format(rec_dirs[rec_id])
    run3_wig_idir = './run3_Pk_1d_wig_mean_{}_masscut/'.format(rec_dirs[rec_id])
    run3_now_idir = './run3_Pk_1d_now_mean_{}_masscut/'.format(rec_dirs[rec_id])
    odir = './run2_3_figs_barn_mean/'
    if not os.path.exists(odir):
        os.makedirs(odir)
    for space_id in xrange(1, 2):
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                wnw_fname = '{}pk{}.wnw_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                wig_fname = '{}pk{}.wig_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                now_fname = '{}pk{}.now_mean_fof_a_{}_mcut{}'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                
                run2_wnw_ifile = run2_wnw_idir + wnw_fname + '.dat'
                run3_wnw_ifile = run3_wnw_idir + wnw_fname + '.dat'
                
                run2_wig_ifile = run2_wig_idir + wig_fname + '.dat'
                run3_wig_ifile = run3_wig_idir + wig_fname + '.dat'
                run2_now_ifile = run2_now_idir + now_fname + '.dat'
                run3_now_ifile = run3_now_idir + now_fname + '.dat'
                
                print(run2_wig_ifile)
                k_obs, Pk_wnw_r2 = np.loadtxt(run2_wnw_ifile, dtype='f8', comments='#', unpack=True)
                k_obs, Pk_wnw_r3 = np.loadtxt(run3_wnw_ifile, dtype='f8', comments='#', unpack=True)
                k_obs = k_obs - delta_k/2.0
                Pk_wnw_r2 = Pk_wnw_r2 - 1.0
                Pk_wnw_r3 = Pk_wnw_r3 - 1.0
#                plot_Pk_wnw_run2_3(k_obs, Pk_wnw_r2, Pk_wnw_r3, odir, space_id, z_id, mcut_id, wnw_fname)

                k_r2, Pk_wig_r2 = np.loadtxt(run2_wig_ifile, dtype='f8', comments='#', unpack=True)
                k_r3, Pk_wig_r3 = np.loadtxt(run3_wig_ifile, dtype='f8', comments='#', unpack=True)
                plot_Pk_run2_3(k_obs, Pk_wig_r2, Pk_wig_r3, odir, space_id, z_id, mcut_id, wig_fname)

                k_r2, Pk_now_r2 = np.loadtxt(run2_now_ifile, dtype='f8', comments='#', unpack=True)
                k_r3, Pk_now_r3 = np.loadtxt(run3_now_ifile, dtype='f8', comments='#', unpack=True)
                plot_Pk_run2_3(k_obs, Pk_now_r2, Pk_now_r3, odir, space_id, z_id, mcut_id, now_fname)

# plot 1 \sigma of P_wig/Pnow
# plot sigma_Pkmean and compare it with the standard error 2.e-4
def plot_sigma_Pk_wnw_mean(kk_bin, sigma_Pk_wnw, sigma_Pk_wnw_diff, Nmode_bin, mu_boundary, rec_id, space_id, z_id, mcut_id, odir):
    plt.loglog(kk_bin, sigma_Pk_wnw, label=r"$\sigma_{P_{\mathrm{wig}}/P_{\mathrm{now}}}$")
    plt.loglog(kk_bin, sigma_Pk_wnw_diff, label=r"$\sigma_{(P_{\mathrm{wig}}-P_{\mathrm{now}})/(G^2b^2 P_{\mathrm{sm}})}$")
    #plt.loglog(kk_bin, np.ones(len(kk_bin))*2.e-4, 'k--')
    plt.grid(True)
    plt.xlim([0.0, 0.32])
    plt.xlabel(r'$k$ ($h/Mpc$)', fontsize=24)
    plt.ylabel(r'$\sigma_{\mathrm{mean}}$', fontsize=24)
    plt.title(r"$z={}$, bias={}, ${}<\mu<{}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], mu_boundary[0], mu_boundary[1], title_space[space_id]))
    plt.loglog(kk_bin, (2.0/Nmode_bin)**0.5, 'k-.', label=r'$1/\sqrt{N_\mathrm{modes}/2}$')
    plt.legend(loc='upper right', frameon=False)
    figname = odir + 'sigma_Pk_{}kave{}.wnw_mean_fof_a_{}_mcut{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
    #plt.savefig(figname)
    plt.show()
    plt.close()

# plot the relative error, e.g., \sigma_x/x
def plot_sigma_P_over_P(k_p, relative_sigma_Pk_wnw, relative_sigma_Pk_wnw_diff, Nmode_bin, mu_boundary, rec_id, space_id, z_id, mcut_id, odir):
    plt.figure(figsize=(10,8))
    plt.loglog(k_p, relative_sigma_Pk_wnw, label=r"$\sigma_{(P_{\mathrm{wig}}/P_{\mathrm{now}})}/(P_{\mathrm{wig}}/P_{\mathrm{now}})$")
    plt.loglog(k_p, relative_sigma_Pk_wnw_diff, label=r"$\sigma_{(P_{\mathrm{wig}}-P_{\mathrm{now}})}/(P_{\mathrm{wig}}-P_{\mathrm{now}})$")
    #plt.loglog(kk_bin, np.ones(len(kk_bin))*2.e-4, 'k--')
    plt.grid(True)
    plt.xlim([0.0, 0.32])
    plt.ylim([1.e-4, 1.e3])
    plt.xlabel(r'$k$ ($h/Mpc$)', fontsize=24)
    plt.ylabel(r"$\sigma_{(P_{\mathrm{wig}}/P_{\mathrm{now}})}/(P_{\mathrm{wig}}/P_{\mathrm{now}})$, $\sigma_{(P_{\mathrm{wig}}-P_{\mathrm{now}})}/(P_{\mathrm{wig}}-P_{\mathrm{now}})$", fontsize=24)
    plt.title(r"$z={}$, bias={}, ${}<\mu<{}$ {}".format(sim_z[z_id], bias_array[z_id][mcut_id], mu_boundary[0], mu_boundary[1], title_space[space_id]), fontsize=24)
    plt.loglog(k_p, (2.0/Nmode_bin)**0.5, 'k-.', label=r'$1/\sqrt{N_\mathrm{modes}/2}$')
    plt.legend(loc='upper right', frameon=False)
    figname = odir + 'relative_sigma_{}kave{}.wnw_mean_fof_a_{}_mcut{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
    #plt.savefig(figname)
    plt.show()
    plt.close()

def show_sigma_Pwnw():
    rec_id = 1
    indices_pn = indices_p1
    k_p = np.array(k_sorted[indices_pn])
    Pk_sm_obsk = interpolate.splev(k_p, tck_Pk_smooth, der=0)    # interpolate theoretical linear P(k) at points k_p
    
    mu_p = np.array(mu_sorted[indices_pn])
    Nmode_p = np.array(Nmode_sorted[indices_pn])
    Nmode_p = Nmode_p * N_dset                                   # count the number of data sets (N_dset)
    mu_boundary = np.round([min(mu_p), max(mu_p)], decimals=1)
    print(mu_boundary)
    n_fitbin = np.size(mu_p)
    print(n_fitbin)
    idir_Pk = ['./run2_3_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id]), './run2_3_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])]
    idir_covPk = ['./run2_3_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id]), './run2_3_Cov_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])]
    odir = './{}_figs_fofPkobs_subPktrue/'.format(rec_dirs[rec_id])
    #odir = './{}_figs_barn_mean/'.format(rec_dirs[rec_id])
    for space_id in xrange(1):
        for z_id in xrange(3):
            for mcut_id in xrange(4,5):#(N_masscut[z_id]):
                bias = bias_array[z_id][mcut_id]
                z = float(sim_z[z_id])
                print("z=", z, "G: ", growth_factor(z, Omega_m)/G_0)
                G2b2 = (growth_factor(z, Omega_m)/G_0*bias)**2.0
                
                inputf = idir_Pk[1] + '{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Pk_wnw = np.loadtxt(inputf, dtype='f4', comments='#', usecols=(2,))
                inputf = idir_covPk[1] + '{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_a = np.loadtxt(inputf, dtype='f4', comments='#')
                sigma_Pk_a = (np.diag(Cov_Pk_a)[indices_pn]) **0.5
                sigma_Pk_wnw = sigma_Pk_a/math.sqrt(float(N_dset))
                ##sigma_Pk_wnw = sigma_Pk_a/math.sqrt(float(N_dset))* G2b2 * Pk_sm_obsk  # get sigma_(Pwig/Pnow)*P_{Lin,w}
                relative_sigma_Pk_wnw = sigma_Pk_wnw/Pk_wnw[indices_pn]              # get the relative error
                print(sigma_Pk_wnw.shape)
                

                
                inputf = idir_Pk[0] + '{}kave{}.wnw_diff_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Pk_wnw_diff = np.loadtxt(inputf, dtype='f4', comments='#', usecols=(2,))
                Pk_wig_minus_now = Pk_wnw_diff[indices_pn] * G2b2* Pk_sm_obsk                          # get back P_wig-P_now
                
                inputf = idir_covPk[0] + '{}kave{}.wnw_diff_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_a = np.loadtxt(inputf, dtype='f4', comments='#')
                sigma_Pk_a = (np.diag(Cov_Pk_a)[indices_pn]) **0.5
                sigma_Pk_wnw_diff = sigma_Pk_a/math.sqrt(float(N_dset))
                ##sigma_Pk_wnw_diff = sigma_Pk_wnw_diff * G2b2 * Pk_sm_obsk              # get back sigma_Pk_wnw_diff
                
                ##relative_sigma_Pk_wnw_diff = sigma_Pk_wnw_diff/np.abs(Pk_wnw_diff[indices_pn]) # get the relative error
                relative_sigma_Pk_wnw_diff = sigma_Pk_wnw_diff/(abs(Pk_wig_minus_now)+1.0) # get the "relative" error, add 1.0 in the denominator
                #print sigma_Pk_a.shape
                plot_sigma_Pk_wnw_mean(k_p, sigma_Pk_wnw, sigma_Pk_wnw_diff, Nmode_p, mu_boundary, rec_id, space_id, z_id, mcut_id, odir)
                #plot_sigma_P_over_P(k_p, relative_sigma_Pk_wnw, relative_sigma_Pk_wnw_diff, Nmode_p, mu_boundary, rec_id, space_id, z_id, mcut_id, odir)

# This subroutine is based on the show_sigma_Pwnw(), test the result obtained from the above.
def show_sigma_Pwig_minus_now():
    rec_id = 1
    indices_pn = indices_p1
    k_p = np.array(k_sorted[indices_pn])
    Pk_sm_obsk = interpolate.splev(k_p, tck_Pk_smooth, der=0)    # interpolate theoretical linear P(k) at points k_p
    
    mu_p = np.array(mu_sorted[indices_pn])
    Nmode_p = np.array(Nmode_sorted[indices_pn])
    Nmode_p = Nmode_p * N_dset                                   # count the number of data sets (N_dset)
    mu_boundary = np.round([min(mu_p), max(mu_p)], decimals=1)
    print(mu_boundary)
    n_fitbin = np.size(mu_p)
    print(n_fitbin)
    idir_Pk = './run2_3_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    idir_covPk = './run2_3_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    odir = './{}_figs_fofPkobs_subPktrue/'.format(rec_dirs[rec_id])
    
    for space_id in xrange(1):
        for z_id in xrange(3):
            for mcut_id in xrange(N_masscut[z_id]):
                inputf = idir_Pk + '{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Pk_wnw = np.loadtxt(inputf, dtype='f4', comments='#', usecols=(2,))
                inputf = idir_covPk + '{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_a = np.loadtxt(inputf, dtype='f4', comments='#')
                sigma_Pk_a = (np.diag(Cov_Pk_a)[indices_pn]) **0.5
                sigma_Pk_wnw = sigma_Pk_a/math.sqrt(float(N_dset))
                relative_sigma_Pk_wnw = sigma_Pk_wnw/Pk_wnw[indices_pn]              # get the relative error
                print(sigma_Pk_wnw.shape)
                
                inputf = idir_Pk + '{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Pk_wnw_diff = np.loadtxt(inputf, dtype='f4', comments='#', usecols=(2,))
                Pk_wig_minus_now = Pk_wnw_diff[indices_pn]
                inputf = idir_covPk + '{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_a = np.loadtxt(inputf, dtype='f4', comments='#')
                sigma_Pk_a = (np.diag(Cov_Pk_a)[indices_pn]) **0.5
                sigma_Pk_wnw_diff = sigma_Pk_a/math.sqrt(float(N_dset))

                relative_sigma_Pk_wnw_diff = sigma_Pk_wnw_diff/(abs(Pk_wig_minus_now)+1.0) # get the "relative" error, add 1.0 in the denominator
                
                plot_sigma_P_over_P(k_p, relative_sigma_Pk_wnw, relative_sigma_Pk_wnw_diff, Nmode_p, mu_boundary, rec_id, space_id, z_id, mcut_id, odir)

# plot correlation coefficient of the covariance matrix of P(k) by setting mu closed to 0.0
def plot_coeff_Cov_Pk_wvsnow(kk_bin, R, N_fitbin, rec_id, z_id, mu_boundary, space_id, mcut_id, odir, figname):
    plt.pcolor(R, cmap='Greys')
    plt.colorbar()
    num_tickers = 5
    d_xy = N_fitbin/num_tickers
    xy_locs = np.arange(0, N_fitbin, d_xy)
    xy_labels = np.array([round(kk_bin[i],2) for i in xrange(0, N_fitbin, d_xy)])
    
    plt.axis([kk_bin.min(), kk_bin.max(), kk_bin.min(), kk_bin.max()])
    plt.xlabel('$k$ ($h/Mpc$)', fontsize=24)
    plt.ylabel('$k$ ($h/Mpc$)', fontsize=24)
    plt.xticks(xy_locs, xy_labels)
    plt.yticks(xy_locs, xy_labels)
    plt.title(r'$z={}$, bias=${}$, ${}<\mu<{}$ {}'.format(sim_z[z_id], bias_array[z_id][mcut_id], mu_boundary[0], mu_boundary[1], title_space[space_id]), fontsize=20)

    plt.savefig(figname)
    #plt.show()
    plt.close()

def show_coeff_Cov_Pk_wvsnow():
    rec_id = 1
    ##idir = './run2_3_Cov_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    idir = './run2_3_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    ##odir = './{}_figs_barn_mean/'.format(rec_dirs[rec_id])
    odir = './{}_figs_fofPkobs_subPktrue/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    #indices_pn = indices_p1 # 0.0<mu<0.1, 251 modes
    #indices_pn = indices_p2 # 0.4<mu<0.5, 443 modes
    indices_pn = indices_p3 # 0.9<mu<1.0, 579 modes
    N_fitbin = len(indices_pn)
    k_p = np.array(k_sorted[indices_pn])
    mu_p = np.array(mu_sorted[indices_pn])
    mu_boundary = np.round([min(mu_p), max(mu_p)], decimals=1)
    id_m = np.array([], dtype=int).reshape(0, N_fitbin)
    for i in xrange(N_fitbin):
        id_m = np.vstack([id_m, indices_pn])
    id_col = np.ravel(id_m)
    id_row = np.ravel(id_m, order='F')
    id_rc = (id_row, id_col)
    print(id_rc)

    for z_id in xrange(3):
        for space_id in xrange(1):
            for mcut_id in xrange(N_masscut[z_id]):
                inputf = idir+'{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_all = np.loadtxt(inputf, dtype='f4', comments='#')
                Cov_Pk_part = (Cov_Pk_all[id_rc]).reshape(N_fitbin, N_fitbin)
                print(Cov_Pk_part.shape)
                R = np.corrcoef(Cov_Pk_part)
                ##figname = odir+'coeff_Cov_Pk_fof_{}kave{}.wnw_mu_{}_a_{}_mcut{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], mu_boundary[1], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                figname = odir+'coeff_Cov_Pk_fof_obs_{}kave{}.wnw_mu_{}_a_{}_mcut{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], mu_boundary[1], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                plot_coeff_Cov_Pk_wvsnow(k_p, R, N_fitbin, rec_id, z_id, mu_boundary, space_id, mcut_id, odir, figname)

# show reduced \chi^2 for fitting P_wig/P_now with different cases.
'''
def show_reduced_chi2():
    rec_id = 1
    idir = "/Users/ding/Documents/playground/WiggleNowiggle/reconstruct_run2_3_power/"
    fof_idir = idir+'run2_3_ALL_params_fitted_mean_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale/'.format(rec_dirs[rec_id])
    chi2_matrix = np.zeros((1,3,5))
    sub_chi2_matrix = np.zeros((1,3))
    for space_id in xrange(1):
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
'''

def main():
    t0 = time.clock()
    #show_Pk_run2n3_mean()
    show_sigma_Pwnw()
    #show_sigma_Pwig_minus_now()
    #show_coeff_Cov_Pk_wvsnow()

    t1 = time.clock()
    print(t1-t0)

# to call the main() function to begin the program.
if __name__ == '__main__':
    main()