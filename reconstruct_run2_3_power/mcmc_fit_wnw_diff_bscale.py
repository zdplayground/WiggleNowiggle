# Copied the code from mcmc_fit_wnw_diff.py, 08/05/2016;
# Modify it correspondingly to fit the mean 2D (P_wig-P_now)/G2b^2P with 100 \mu bins and mass cut application.
# 1. The fitting k range is from the k_min of the data file to about 0.3 h/Mpc.
# 2. Set the N_skip_footer and reading file process according to the data file.
# 3. Fit 20 data sets with mass cut separately. From MCMC results, the error of fitted \alpha is about 0.0001 assigning 50 walker
# each with 4000 steps. And the error of \Sigma is about 0.01.
# 4. On the base of the model in mcmc_fit_wnw_diff.py, we add the term (1+b_scale * k^2), where b_scale is a fitting parameter.
# 5. Modify the prior range for b_scale, i.e. -100.0<b_scale<100.0; 08/11/2016.
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
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator


# Define the probability function as likelihood * prior.
# theta is the parameter array
def lnprior(theta):
# alpha_1: alpha perpendicular, alpha_2: alpha parallel, mu_p: mu prime, two sigma components: sigma_xy, sigma_z
    alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = theta
    if 0.9<alpha_1<1.1 and 0.9<alpha_2<1.1 and 0.0<=sigma_xy<15.0 and 0.0<=sigma_z<15.0 and -100.0<b_scale<100.0:
        return 0.0
    return -np.inf

def lnlike(theta, k_p, mu_p, Pk_obs, tck_Pk_true, ivar):
    alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = theta
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    Pk_t = interpolate.splev(k_t, tck_Pk_true, der=0)
    # here Pk_t represents Pk_wig/Pk_now in theory
    Pk_model = (Pk_t-1.0)*np.exp(k_t*k_t *(mu_t*mu_t*(sigma_xy+sigma_z)*(sigma_xy-sigma_z)-sigma_xy**2.0)/2.0) * (1.0+b_scale * k_t**2.0)
    diff = Pk_model - Pk_obs
    return -0.5* np.sum(diff**2.0 *ivar)

def lnprob(theta, k_p, mu_p, Pk_obs, tck_Pk_true, ivar):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k_p, mu_p, Pk_obs, tck_Pk_true, ivar)

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
def write_params(filename, alpha_1, alpha_2, sigma_xy, sigma_z, b_scale, reduced_chi2):
    data_m = np.array([alpha_1, alpha_2, sigma_xy, sigma_z, b_scale])
    header_line = ' The fitted parameters alpha_1, alpha_2, sigma_xy, sigma_z, b_scale (by row), with their upward and downward one sigma error.\n'
    header_line += ' The reduced \chi^2={0:.3f}'.format(reduced_chi2)
    np.savetxt(filename, data_m, fmt='%.7f', header=header_line, comments='#')


# MCMC routine
def mcmc_routine(N_params, N_walkers, N_walkersteps, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, name_id, space_id):
    
    alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = 0.9, 1.1, 10.0, 10.0, 0.0
    
    result = op.minimize(chi2, [alpha_1, alpha_2, sigma_xy, sigma_z, b_scale], args=(k_range, mu_range, Pk_wnow_obs, tck_Pk_true, ivar_Pk_wnow))
    alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = result["x"]
    print(alpha_1, alpha_2, sigma_xy, sigma_z, b_scale)

    # Set up the sampler.
    ndim, nwalkers = N_params, N_walkers
    # sometimes sigma_xy or sigma_z obtained from optimize routine is negative, we should use abs() passing to the emcee routine.
    pos = [[np.random.uniform(0.9,1.1), np.random.uniform(0.9,1.1), np.random.uniform(0.1, 15.0), np.random.uniform(0.1, 15.0), np.random.uniform(-1.0, 1.0)] for i in xrange(nwalkers)]
    print(np.shape(pos))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=3.0, args=(k_range, mu_range, Pk_wnow_obs, tck_Pk_true, ivar_Pk_wnow))
    print(type(sampler))
    
    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    sampler.reset()
    
    walkersteps = N_walkersteps
    sampler.run_mcmc(pos, walkersteps)  # the chain member has the shape: (nwalkers, nlinks, dim)
    print("Done.")
    print("Autocorrelation time: ", sampler.acor)
    print(sampler.get_autocorr_time(window=50, fast=False))  # not sure about the function of window
    print ("Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))
    
    sample_chains = sampler.chain[:, walkersteps/2:, :]
    scalereduction = gelman_rubin_convergence(sample_chains, nwalkers, np.size(sample_chains, axis=1), ndim)
    print "Scalereduction: ", scalereduction
    
    burnin = walkersteps/2
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    # Compute the quantiles.
    alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                              zip(*np.percentile(samples, [15.865, 50, 84.135], axis=0)))
                                              
      
    print("""MCMC result:
          alpha_1={0[0]}+{0[1]}-{0[2]}
          alpha_2={1[0]}+{1[1]}-{1[2]}
          sigma_xy={2[0]}+{2[1]}-{2[2]}
          sigma_z={3[0]}+{3[1]}-{3[2]}
          bscale={4[0]}+{4[1]}-{4[2]}""".format(alpha_1, alpha_2, sigma_xy, sigma_z, b_scale))
      
    #plot_walkers(sampler, alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], name_id, ascale_id, space_id)
    #plot_triangle(samples, alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], name_id, ascale_id, space_id, mass_id)
    del sampler
    return alpha_1, alpha_2, sigma_xy, sigma_z, b_scale



#################################################################################################
######################################--------main code---------#################################
#################################################################################################
# simulation run name
sim_run = 'run2_3'
N_dataset = 20
N_mu_bin = 100
N_skip_header = 11
N_skip_footer = 31977


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

inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

Pk_wovers = Pk_wiggle/Pk_smooth
tck_Pk_true = interpolate.splrep(k_smooth, Pk_wovers)


# firstly, read one file and get k bins we want for the fitting range
dir0='./run2_3_Pk_obs_2d_wnw_mean_ALL_ksorted_mu_masscut/'

inputf = dir0 +'Rkaver.wnw_diff_mean_fof_a_1.0000_mcut37.dat'
k_p, mu_p = np.loadtxt(inputf, dtype='f8', comments='#', delimiter=' ', usecols=(0,1), unpack=True)
print(k_p, mu_p)
N_fitbin = len(k_p)
print('# of (k, mu) bins: ', N_fitbin)
Volume = 1380.0**3.0   # the volume of simulation box

N_params = 5
N_walkers = 60
N_walkersteps = 10000
#------- fit FoF mean P(k, mu) -------#
def fit_fof_mean():
    name_id = 0
    for rec_id in xrange(1):
        odir = './params_{}_wnw_diff_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
        if not os.path.exists(odir):
            os.makedirs(odir)
        for mcut_id in xrange(N_masscut):
            np.random.seed()
            for z_id in xrange(1):
                for space_id in xrange(1):
                    ifile_Pk = './{}_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wnw_diff_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                    Pk_wnw_diff_obs = np.loadtxt(ifile_Pk, dtype='f8', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.
                    
                    ifile_Cov_Pk = './{}_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wnw_diff_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                    Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f8', comments='#')
                    ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                              # the mean sigma error
                    
                    alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = mcmc_routine(N_params, N_walkers, N_walkersteps, k_p, mu_p, Pk_wnw_diff_obs, ivar_Pk_wnow, name_id, space_id)
                    chi_square = chi2([alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], b_scale[0]], k_p, mu_p, Pk_wnw_diff_obs, tck_Pk_true, ivar_Pk_wnow)
                    reduced_chi2 = chi_square/(N_fitbin-N_params)
                    print('Reduced chi2: ', reduced_chi2)
                    # output parameters into a file
                    ofile_params = odir + '{}kave{}.wnw_diff_bscale_mean_fof_a_{}_mcut{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                    write_params(ofile_params, alpha_1, alpha_2, sigma_xy, sigma_z, b_scale, reduced_chi2)


#------- fit Subsample mean P(k, mu) -------#
def fit_sub_mean():
    name_id = 0
    for rec_id in xrange(1):
        odir = './params_{}_wnw_diff_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
        if not os.path.exists(odir):
            os.makedirs(odir)
        np.random.seed()
        for z_id in xrange(3):
            for space_id in xrange(1):
                ifile_Pk = './{}_sub_Pk_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wnw_diff_mean_a_{}ga.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
                Pk_wnw_diff_true = np.loadtxt(ifile_Pk, dtype='f8', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.
                
                ifile_Cov_Pk = './{}_sub_Cov_Pk_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wnw_diff_mean_a_{}ga.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
                Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f8', comments='#')
                ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                              # the mean sigma error
                
                alpha_1, alpha_2, sigma_xy, sigma_z, b_scale = mcmc_routine(N_params, N_walkers, N_walkersteps, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, name_id, space_id)
                chi_square = chi2([alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], b_scale[0]], k_p, mu_p, Pk_wnw_diff_true, tck_Pk_true, ivar_Pk_wnow)
                reduced_chi2 = chi_square/(N_fitbin-N_params)
                print('Reduced chi2: ', reduced_chi2)
                # output parameters into a file
                ofile_params = odir + '{}kave{}.wnw_diff_bscale_mean_sub_a_{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
                write_params(ofile_params, alpha_1, alpha_2, sigma_xy, sigma_z, b_scale, reduced_chi2)

def main():
    t0 = time.clock()
    fit_fof_mean()
    fit_sub_mean()
    t1 = time.clock()
    print(t1-t0)

if __name__ == '__main__':
    main()


