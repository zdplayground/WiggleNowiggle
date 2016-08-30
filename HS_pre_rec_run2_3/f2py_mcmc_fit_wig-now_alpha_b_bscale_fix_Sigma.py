# Copied the code from mcmc_fit_wig-now_b_bscale_alpha_Sigma.py, 08/21/2016;
# Modify it to fix \alpha_xy = \alpha_z, and \Sigma_xy=\Sigma_z=\Sigma from calculation.
# b^2 (1+b_scale *k^2)G^2 (Plin - Psm) C_G^2, where both b and b_scale are the fitting parameters.
# 1. The fitting k range is from the k_min of the data file to about 0.3 h/Mpc.
# 2. Set the N_skip_footer and reading file process according to the data file.
# 3. Fit 20 data sets with mass cut separately. From MCMC results, the error of fitted \alpha is about 0.0001 assigning 50 walkers
#    each with 4000 steps. And the error of \Sigma is about 0.01.
# 4. Made the prior range broader and tested the fitting result is consistent with that from narrow prior range, 08/16/2016.
# 5. Write some functions with fortran. Following Zvonimir's suggestion, use theoretical \Sigma calculate from \Sigam^2/2 =
#    1/3 \int d^3 q P_lin/q^2 (1 - j_0[ R_bao * q] ) ~ 30 (Mpc/h)^2. Store data files in the folder odir params_DD_wig-now_b_bscale_fitted_mean_alpha_Sigma_Theory_dset/, 08/25/2016.
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import print_function
import csv
import emcee
#import corner
import time
import numpy as np
import os
import scipy.optimize as op
from scipy import interpolate
from growth_fun import growth_factor
from lnprob_module import match_params, cal_pk_model, lnprior
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator


# input Pk_obs = \hat{P}_wig - \hat{P}_now
def lnlike(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf):
    alpha_1, alpha_2, sigma_xy, sigma_z, b_0, b_scale = match_params(theta, params_indices, fix_params)
    # set alpha_1=alpha_2 and sigma_xy = sigma_z
    alpha_1 = alpha_2
    #sigma_xy = sigma_z
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    Pk_linw = interpolate.splev(k_t, tck_Pk_linw, der=0)
    Pk_sm = interpolate.splev(k_t, tck_Pk_sm, der=0)

    Pk_model = cal_pk_model(Pk_linw, Pk_sm, k_t, mu_t, sigma_xy, sigma_z, b_0, b_scale, norm_gf)
    diff = Pk_model - Pk_obs
    return -0.5* np.sum(diff**2.0 *ivar)

def lnprob(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf):
    lp = lnprior(theta, params_indices, fix_params)
    if (lp < -1.e20):
        return -np.inf
    return lp + lnlike(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)


# Gelman&Rubin convergence criterion, referenced from Florian's code RSDfit_challenge_hex_steps_fc_hoppper.py
def gelman_rubin_convergence(sample_chains, num_chains, num_steps, num_params):
    
    # Calculate Gelman & Rubin diagnostic
    # 1. Remove the first half of the current chains
    # 2. Calculate the within chain and between chain variances
    # 3. estimate your variance from the within chain and between chain variance
    # 4. Calculate the potential scale reduction parameter
    
    withinchainmean = np.mean(sample_chains, axis=1) # shape=(num_chains, num_params)
    withinchainvar = np.var(sample_chains, axis=1)   # shape=(num_chains, num_params)
    meanall = np.mean(withinchainmean, axis=0)       # shape=(1, num_params)
    W = np.mean(withinchainvar, axis=0)
    B = np.zeros(num_params)
    #print withinchainmean, withinchainvar, meanall, W, B
    for i in xrange(num_params):
        for j in xrange(num_chains):
            B[i] = B[i] + num_steps*(meanall[i] - withinchainmean[j][i])**2.0/(num_chains-1.)
    estvar = (1. - 1./num_steps)*W + B/num_steps
    scalereduction = np.sqrt(estvar/W)
    return scalereduction


# write parameters fitted in files
def write_params(filename, params_mcmc, params_name, reduced_chi2):
    header_line = '# The fitted parameters alpha_1, alpha_2, sigma_xy, sigma_z, b_0, b_scale (by row) with their upward and downward one sigma error, and the reduced \chi^2.\n'
    with open(filename, 'w') as fwriter:
        fwriter.write(header_line)
        for i in xrange(len(params_name)):
            fwriter.write("#{0}: {1:.7f} {2:.7f} {3:.7f}\n".format(params_name[i], params_mcmc[i][0], params_mcmc[i][1], params_mcmc[i][2]))
        fwriter.write("#reduced_chi2: {0:.7f}".format(reduced_chi2))


# MCMC routine
def mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name):
    
    result = op.minimize(chi2, theta, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf))
    theta_optimize = result["x"]
    print("Optimized parameters: ", theta_optimize) # only output parameters which are free to change
    
    # Set up the sampler.
    # sometimes sigma_xy or sigma_z obtained from optimize routine is negative, we should use abs() passing to the emcee routine.
    pos = [theta_optimize + params_T*np.random.uniform(-1.0,1.0,N_params) for i in xrange(N_walkers)]
    #print(pos, np.shape(pos))
    
    sampler = emcee.EnsembleSampler(N_walkers, N_params, lnprob, a=2.5, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf))
    print(type(sampler))
    
    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    sampler.reset()
    
    sampler.run_mcmc(pos, N_walkersteps)  # the chain member has the shape: (N_walkers, nlinks, dim)
    print("Done.")
    print("Autocorrelation time: ", sampler.acor) # show the autocorrelation time
    #print(sampler.get_autocorr_time(window=50, fast=False))  # not sure about the function of window
    print ("Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))
    
    burnin = N_walkersteps/2
    sample_chains = sampler.chain[:, burnin:, :]
    scalereduction = gelman_rubin_convergence(sample_chains, N_walkers, np.size(sample_chains, axis=1), N_params)
    print "Scalereduction: ", scalereduction
    
    #samples = sampler.chain[:, burnin:, :].reshape((-1, N_params))
    samples = sampler.chain[:, 1000:, :].reshape((-1, N_params))
    # Compute the quantiles.
    theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.865, 50, 84.135], axis=0)))
    print(theta_mcmc)
    print("MCMC result: ")
    for i in xrange(len(theta)):
        print("{0}={1[0]}+{1[1]}-{1[2]}".format(params_name[i], theta_mcmc[i]))
    
    del sampler
    return np.array(theta_mcmc)



#################################################################################################
######################################--------main code---------#################################
#################################################################################################
# simulation run name
sim_run = 'run2_3'
N_dataset = 20
N_mu_bin = 100
#N_skip_header = 11
#N_skip_footer = 31977
Omega_m = 0.3075
G_0 = growth_factor(0.0, Omega_m) # G_0 at z=0, normalization factor
N_walkers = 40
N_walkersteps = 5000

rec_id = 0
#Sigma_0 = 8.3364           # the value of \Sigma_xy and \Sigma_z, unit Mpc/h, calculated using P_{Lin,w}
Sigma_0 = 7.8364            # ~7.8 Mpc/h, the theoretical value

sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = ['NW', 'WG']
sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']

mcut_Npar_list = [[37, 149, 516, 1524, 3830],
                  [35, 123, 374, 962, 2105],
                  [34, 103, 290, 681, 1390]]
N_masscut = np.size(mcut_Npar_list, axis=1)

inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
tck_Pk_sm = interpolate.splrep(k_smooth, Pk_smooth)

inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
tck_Pk_linw = interpolate.splrep(k_wiggle, Pk_wiggle)


# firstly, read one file and get k bins we want for the fitting range
dir0='/Users/ding/Documents/playground/WiggleNowiggle/reconstruct_run2_3_power/run2_3_Pk_obs_2d_wnw_mean_ALL_ksorted_mu_masscut/'
inputf = dir0 +'Rkaver.wnw_diff_mean_fof_a_1.0000_mcut37.dat'
k_p, mu_p = np.loadtxt(inputf, dtype='f8', comments='#', delimiter=' ', usecols=(0,1), unpack=True)
print(k_p, mu_p)
N_fitbin = len(k_p)
print('# of (k, mu) bins: ', N_fitbin)
Volume = 1380.0**3.0   # the volume of simulation box


##odir = './params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_dset/'.format(rec_dirs[rec_id])
odir = './params_{}_wig-now_b_bscale_fitted_mean_alpha_Sigma_Theory_dset/'.format(rec_dirs[rec_id])
if not os.path.exists(odir):
    os.makedirs(odir)
#------- fit the FoF and DM Subsample mean P(k, mu)_wig - P_now by G^2 b^2(P_linw - P_sm)*C_G^2 with isotropic constraints -------#
def fit_fof_sub_mean():
    for z_id in xrange(1,3):
        G_z = growth_factor(float(sim_z[z_id]), Omega_m)/G_0
        Sigma_z = Sigma_0 * G_z
        
#        Sigma_z = Sigma_z - 1.0
#        for count in xrange(5):

        print("Sigma_z: ", Sigma_z)
        sigma_xy, sigma_z = Sigma_z, Sigma_z
        alpha_1, alpha_2, b_0, b_scale = 1.0, 1.0, 1.0, 0.0
        all_params = alpha_1, alpha_2, sigma_xy, sigma_z, b_0, b_scale
        all_names = "alpha_1", "alpha_2", "sigma_xy", "sigma_z", "b_0", "b_scale"
        all_temperature = 0.01, 0.01, 0.1, 0.1, 0.1, 1.0

        params_indices = [0, 1, 0, 0, 1, 1] # 0: parameter fixed, 1: parameter free.
        fix_params = np.array([], dtype=np.float)
        theta = np.array([], dtype=np.float)
        params_T = np.array([], dtype=np.float)
        params_name = []
        N_params = 0
        count = 0
        for i in params_indices:
            if i == 1:
                fix_params = np.append(fix_params, 0.)
                theta = np.append(theta, all_params[count])
                params_T = np.append(params_T, all_temperature[count])
                params_name.append(all_names[count])
                N_params += 1
            else:
                fix_params = np.append(fix_params, all_params[count])
            count += 1
        print(theta, params_name, N_params)
        print("fixed params: ", fix_params)

        norm_gf = growth_factor(float(sim_z[z_id]), Omega_m)/G_0
        np.random.seed()
        for space_id in xrange(1):
            for mcut_id in xrange(N_masscut):
                ifile_Pk = './{}_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Pk_wnw_diff_obs = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.
                
                ifile_Cov_Pk = './{}_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
                ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                                   # the mean sigma error
                
                params_mcmc = mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name)

                chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf)
                reduced_chi2 = chi_square/(N_fitbin-N_params)
                print('Reduced chi2: ', reduced_chi2)
                # output parameters into a file
                ##ofile_params = odir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], ''.join(map(str, params_indices)))
                ofile_params = odir + '{}kave{}.wig-now_b_bscale_mean_fof_a_{}_mcut{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], ''.join(map(str, params_indices)), round(Sigma_z, 3))
                write_params(ofile_params, params_mcmc, params_name, reduced_chi2)

            # Fit for DM power spectrum
            ifile_Pk = './{}_sub_Pk_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wig_minus_now_mean_sub_a_{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            Pk_wnw_diff_true = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.
            
            ifile_Cov_Pk = './{}_sub_Cov_Pk_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wig_minus_now_mean_sub_a_{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
            ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                              # the mean sigma error
            
            params_mcmc = mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name)
            chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf)
            reduced_chi2 = chi_square/(N_fitbin-N_params)
            print('Reduced chi2: ', reduced_chi2)
            # output parameters into a file
            ##ofile_params = odir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)))
            ofile_params = odir + '{}kave{}.wig-now_b_bscale_mean_sub_a_{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)), round(Sigma_z, 3))
            write_params(ofile_params, params_mcmc, params_name, reduced_chi2)

#            Sigma_z = Sigma_z+0.5


def main():
    t0 = time.clock()
    fit_fof_sub_mean()
    t1 = time.clock()
    print(t1-t0)

if __name__ == '__main__':
    main()


