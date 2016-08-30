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
#bias_0 = np.array([0.882, 1.197, 1.481])
bias_0 = np.array([1.0, 1.0, 1.0])
omega_m0 = 0.3075
omega_l = 0.6925
fix_Sigma = [8.336, 6.116, 5.079]  # obtained using P_{Lin,w}


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
def plot_alpha_fofsub_bias(bias_array, y1, bias_0, sub_y1, N_masscut, space_id, sim_space, title_space, odir, figname):
    ylim_list = np.array([[-0.2, 1.4], [-0.4, 0.4]])
    textline_loc = np.array([[], [0.6, 0.9]])
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
               r"square dots: Subsample $\alpha$"+"\n"+\
               r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(0.6, 0.9, textline, fontsize=16)
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

    plt.axhline(fix_Sigma[0], ls='--', color='k')
    plt.axhline(fix_Sigma[1], ls='--', color='r')
    plt.axhline(fix_Sigma[2], ls='--', color='b')
    
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
    plt.text(2.4, 7.5, textline, fontsize=16)
    plt.xlim([0.5, 4.2])
    #plt.ylim([ylim_list[rec_id, 0], ylim_list[rec_id, 1]]) # with four parameters
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
    plt.ylim([0., 4.5])
    plt.xlim([0., 4.5])
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
    ##plt.errorbar(bias_array[0][:], y1[0 : n0, 0], yerr=[y1[0 : n0, 2], y1[0 : n0, 1]], fmt='k*--')
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
    plt.text(2.5, 3.0, textline, fontsize=16)
    plt.savefig(odir + figname)
    plt.show()
    plt.close()


# define a function to show the reduced chi^2 from three different fitting models.
def plot_reduced_chi2(bias_array, y1, bias_0, sub_y1, odir, figname):
    plt.plot(bias_array[0][:], y1[0, 0, :], 'k*-.', lw=2.0) # line style shows the fitting model.
    ##plt.plot(bias_array[0][:], y1[0, 0, :], 'k*--', lw=2.0)
    plt.plot(bias_array[1][:], y1[0, 1, :], 'ro-.', lw=2.0)
    plt.plot(bias_array[2][:], y1[0, 2, :], 'bv-.', lw=2.0)

    plt.plot(bias_array[0][:], y1[1, 0, :], 'k*--')
    plt.plot(bias_array[1][:], y1[1, 1, :], 'ro--')
    plt.plot(bias_array[2][:], y1[1, 2, :], 'bv--')
#
#    plt.plot(bias_array[0][:], y1[2, 0, :], 'k*-')
#    plt.plot(bias_array[1][:], y1[2, 1, :], 'ro-')
#    plt.plot(bias_array[2][:], y1[2, 2, :], 'bv-')

    plt.plot(bias_0[0], sub_y1[0, 0], 'ks', bias_0[0], sub_y1[0, 1], 'rs', bias_0[0], sub_y1[0, 2], 'bs')  # D means diamond shape
    plt.plot(bias_0[0], sub_y1[1, 0], 'kp', bias_0[0], sub_y1[1, 1], 'rp', bias_0[0], sub_y1[1, 2], 'bp')
#    plt.plot(bias_0[0], sub_y1[2, 0], 'kh', bias_0[0], sub_y1[2, 1], 'rh', bias_0[0], sub_y1[2, 2], 'bh')
    plt.xlabel('bias', fontsize=24)
    plt.ylim([1.05, 1.30])
    plt.ylabel(r'$\Delta \chi^2$', fontsize=24)
    textline = r"blue: $z=1.0$"+"\n"+r"red: $z=0.6$"+"\n"+r"black: $z=0.0$"
    plt.text(3.0, 1.20, textline, fontsize=20)
    plt.savefig(odir + figname)
    plt.show()
    plt.close()


# Show the scatter plot of alpha and b_scale
def scatter_alpha_bscale(alpha_m, b_scalem, sub_alpha_m, sub_b_scalem, odir, figname_ab):
    print(alpha_m.shape, b_scalem.shape)
    plt.scatter(alpha_m[0:5], b_scalem[0:5], c='k')
    plt.scatter(alpha_m[5:10], b_scalem[5:10], c='r')
    plt.scatter(alpha_m[10:15], b_scalem[10:15], c='b')
    plt.scatter(sub_alpha_m[0], sub_b_scalem[0], c='k', marker=u's')
    plt.scatter(sub_alpha_m[1], sub_b_scalem[1], c='r', marker=u's')
    plt.scatter(sub_alpha_m[2], sub_b_scalem[2], c='b', marker=u's')
    plt.xlabel(r"$\alpha$", fontsize=24)
    plt.ylabel(r"$b_{\mathtt{scale}}$", fontsize=24)
    textline = r"blue: a=0.5"+"\n"+"red: a=0.625"+"\n"+"black: a=1.0"
    plt.text(1.01, 3.0, textline, fontsize=16)
    plt.savefig(odir+figname_ab)
    #plt.show()
    plt.close()


#----- part 1: plot parametrs fitted for the mean P(k)
#
def read_params(inputf):
    N_skip_header = 1
    N_skip_footer = 1
    data_m = np.genfromtxt(inputf, dtype=np.float, comments=None, skip_header=N_skip_header, skip_footer=N_skip_footer, usecols=(1, 2, 3))
    return data_m

# define a function to show parameters fitted under the condition alpha and Sigma are symmetric.
def show_params_fitted():
    sigma_id = 0                                         # sigma_id denotes the vaule of \Sigma picked for the case z=0.
    Sigmaz_array = [7.336, 7.836, 8.336, 8.836, 9.336]   # at z=0, ranging -1 to +1 on the base of 8.336 which is obtained using P_{Lin,w}
    # 1 represents true as a free parameter, 0 represents a fixed parameter. Parameters are alpha_perp, alpha_para, \Sigam_xy, \Sigma_z, b_0, b_scale
    #params_indices = np.array([0, 1, 0, 1, 1, 1])
    params_indices = np.array([0, 1, 0, 0, 1, 1])
    indices_str = ''.join(map(str, params_indices))
    
    ##fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_dset/'.format(rec_dirs[rec_id])
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_Theory_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    ##odir = idir+'{}_figs_barn_mean_wnw_diff_bscale/'.format(rec_dirs[rec_id])
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
                if z_id == 0:
                    fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], indices_str, Sigmaz_array[sigma_id])
                    sub_ifile = sub_idir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], indices_str, Sigmaz_array[sigma_id])
                else:
                    fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], indices_str)
                    sub_ifile = sub_idir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], indices_str)
                print(fof_ifile)

                params_err = read_params(fof_ifile)
                alpha_m = np.vstack((alpha_m, params_err[0, :]))
#                Sigma_m = np.vstack((Sigma_m, params_err[1, :]))
#                b_0m = np.vstack((b_0m, params_err[2, :]))
#                b_scalem = np.vstack((b_scalem, params_err[3, :]))
                b_0m = np.vstack((b_0m, params_err[1, :]))
                b_scalem = np.vstack((b_scalem, params_err[2, :]))

            print(sub_ifile)
            
            ##params_err = np.loadtxt(sub_ifile, dtype='float64', comments='#')
            params_err = read_params(sub_ifile)
            sub_alpha_m = np.vstack((sub_alpha_m, params_err[0, :]))
#            sub_Sigma_m = np.vstack((sub_Sigma_m, params_err[1, :]))
#            sub_b_0m = np.vstack((sub_b_0m, params_err[2, :]))
#            sub_b_scalem = np.vstack((sub_b_scalem, params_err[3, :]))
            sub_b_0m = np.vstack((sub_b_0m, params_err[1, :]))
            sub_b_scalem = np.vstack((sub_b_scalem, params_err[2, :]))

        figname_alpha = '{}kave{}_wig-now_mean_fof_sub_alpha_params{}_isotropic_Sigmaz_{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str, Sigma_array[z_id])
        figname_b_0 = '{}kave{}_wig-now_mean_fof_sub_b0_params{}_isotropic_Sigmaz_{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str, Sigma_array[z_id])
        figname_b_scale = '{}kave{}_wig-now_mean_fof_sub_bscale_params{}_isotropic_Sigmaz_{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str, Sigma_array[z_id])


        plot_alpha_fofsub_bias(bias_array, alpha_m, bias_0, sub_alpha_m, N_masscut, space_id, sim_space, title_space, odir, figname_alpha)
#        plot_Sigma_fofsub_bias(bias_array, Sigma_m, bias_0, sub_Sigma_m, N_masscut, space_id, sim_space, title_space, sim_a, omega_m0, omega_l, odir, figname_Sigma)
        plot_b_0_fofsub(bias_array, b_0m, bias_0, sub_b_0m, odir, figname_b_0)
        plot_b_scale_fofsub(bias_array, b_scalem, bias_0, sub_b_scalem, odir, figname_b_scale)

# define a function to show parameters alpha and b_scale fitted from the case \Sigma is obtained from theoretical equation given by Zvonimir.
def show_params_fitted_Sigma_Theory():
    Sigma_array = [7.836, 5.749, 4.774]
    # 1 represents true as a free parameter, 0 represents a fixed parameter. Parameters are alpha_perp, alpha_para, \Sigam_xy, \Sigma_z, b_0, b_scale
    params_indices = np.array([0, 1, 0, 0, 1, 1])
    ##params_indices = np.array([0, 0, 0, 0, 1, 1])
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

                fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], indices_str, Sigma_array[z_id])
                sub_ifile = sub_idir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], indices_str, Sigma_array[z_id])
                params_err = read_params(fof_ifile)
                alpha_m = np.vstack((alpha_m, params_err[0, :]))
                b_0m = np.vstack((b_0m, params_err[1, :]))
                b_scalem = np.vstack((b_scalem, params_err[2, :]))
#                b_0m = np.vstack((b_0m, params_err[0, :]))
#                b_scalem = np.vstack((b_scalem, params_err[1, :]))
            print(sub_ifile)
            
            params_err = read_params(sub_ifile)
            sub_alpha_m = np.vstack((sub_alpha_m, params_err[0, :]))
            sub_b_0m = np.vstack((sub_b_0m, params_err[1, :]))
            sub_b_scalem = np.vstack((sub_b_scalem, params_err[2, :]))

#            sub_b_0m = np.vstack((sub_b_0m, params_err[0, :]))
#            sub_b_scalem = np.vstack((sub_b_scalem, params_err[1, :]))

        figname_alpha = '{}kave{}_wig-now_mean_fof_sub_alpha_params{}_isotropic_Sigma_Theory.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_0 = '{}kave{}_wig-now_mean_fof_sub_b0_params{}_isotropic_Sigma_Theory.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_b_scale = '{}kave{}_wig-now_mean_fof_sub_bscale_params{}_isotropic_Sigma_Theory.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)
        figname_ab = '{}kave{}_wig-now_mean_fof_sub_params{}_isotropic_alpha_Sigma_scatter.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], indices_str)

#        plot_alpha_fofsub_bias(bias_array, alpha_m, bias_0, sub_alpha_m, N_masscut, space_id, sim_space, title_space, odir, figname_alpha)
#        plot_b_0_fofsub(bias_array, b_0m, bias_0, sub_b_0m, odir, figname_b_0)
#        plot_b_scale_fofsub(bias_array, b_scalem, bias_0, sub_b_scalem, odir, figname_b_scale)
#print("alpha_m, b_scalem: ", alpha_m, b_scalem)
        scatter_alpha_bscale(alpha_m[:, 0], b_scalem[:, 0], sub_alpha_m[:, 0], sub_b_scalem[:, 0], odir, figname_ab)

# show reduced \chi^2 from different numbers of fitting parameters
def show_reduced_chi2():
    #pindices_list = ['010111']
    pindices_list = ['010011']
    #N_skip_header_list = [5]
    N_skip_header_list = [4]
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
                    if z_id == 0:
                        fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], pindices_list[indices_case], Sigmaz_array[1])
                    else:
                        fof_ifile = fof_idir +'{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], pindices_list[indices_case])
                    reduced_chi2 = np.genfromtxt(fof_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                    print(reduced_chi2)
                    chi2_matrix[indices_case][z_id][mcut_id] = reduced_chi2
                sub_ifile = sub_idir +'{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], pindices_list[indices_case])
                sub_reduced_chi2 = np.genfromtxt(sub_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                sub_chi2_matrix[indices_case][z_id] = sub_reduced_chi2
    print(chi2_matrix)
    ##figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2_params{}_isotropic.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], pindices_list[indices_case])
    figname_chi2 = '{}kave{}_wig-now_mean_fof_sub_reduced_chi2_params{}_isotropic_Sigmaz_{}.pdf'.format(rec_fprefix[rec_id], sim_space[space_id], pindices_list[indices_case], Sigmaz_array[sigma_id])
    plot_reduced_chi2(bias_array, chi2_matrix, bias_0, sub_chi2_matrix, odir, figname_chi2)

# show reduced \chi^2 from different numbers of fitting parameters
def show_reduced_chi2_Sigma_Theory():
    Sigma_array = [7.836, 5.749, 4.774]
    pindices_list = ['000011', '010011']
    N_skip_header_list = [3, 4]
##    pindices_list = ['000011']
##    N_skip_header_list = [3]
    fof_idir = idir+'params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_Theory_dset/'.format(rec_dirs[rec_id])
    sub_idir = fof_idir
    odir = idir+'{}_figs_barn_mean_wnw_diff_bscale_Sigma_Theory/'.format(rec_dirs[rec_id])
    chi2_matrix = np.zeros((2,3,5))
    sub_chi2_matrix = np.zeros((2,3))
    for space_id in xrange(1):
        for indices_case in xrange(2):
            for z_id in xrange(3):
                for mcut_id in xrange(N_masscut[z_id]):
                    fof_ifile = fof_idir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], pindices_list[indices_case], Sigma_array[z_id])
                    reduced_chi2 = np.genfromtxt(fof_ifile, dtype=np.float, comments=None, skip_header=N_skip_header_list[indices_case], usecols=(1,))
                    print(reduced_chi2)
                    chi2_matrix[indices_case][z_id][mcut_id] = reduced_chi2
                sub_ifile = sub_idir +'{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], pindices_list[indices_case], Sigma_array[z_id])
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
    #show_params_fitted_Sigma_Theory()
    #show_reduced_chi2()
    show_reduced_chi2_Sigma_Theory()
    t1 = time.clock()
    print(t1-t0)

# to call the main() function to begin the program.
if __name__ == '__main__':
    main()
