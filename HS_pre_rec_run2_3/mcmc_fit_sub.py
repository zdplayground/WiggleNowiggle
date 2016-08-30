# Copied the code from the folder reconstruct_run2_3_power, 07/16/2016.
# Modify it correspondingly to fit the mean 2D P_wig/P_now with 100 \mu bins in subsample simulations.
# 1. The fitting k range is from the k_min of the data file to about 0.3 h/Mpc.
# 2. Set the N_skip_footer and reading file process according to the data file.
# 3. Use P(k, \mu) data files with shot noise subtracted.
# 4. Fit 20 data sets with mass cut separately. From MCMC results, the error of fitted \alpha is about 0.0001 assigning 50 walker
# each with 4000 steps. And the error of \Sigma is about 0.01.
#
# Question: The same mass cut for wiggled and no-wiggle power spectrum gives the different numbers of halos selected. Therefore, the
# shot noise is different. The difference is around the range(0.00001, 0.0006). Does the error propagate largely to Sigma's? 
#
# (6. After reconstruction, the fitted \Sigma seems to go to 0. So the prior should be changed, including negative value as well. 06/28/2016)
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
    alpha_1, alpha_2, sigma_xy, sigma_z = theta
    if 0.9<alpha_1<1.1 and 0.9<alpha_2<1.1 and 0.0<sigma_xy<15.0 and 0.0<sigma_z<15.0:
        return 0.0
    return -np.inf

def lnlike(theta, k_p, mu_p, Pk_obs, tck_Pk_true, ivar):
    alpha_1, alpha_2, sigma_xy, sigma_z = theta
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    Pk_t = interpolate.splev(k_t, tck_Pk_true, der=0)
    # here Pk_t represents Pk_wig/Pk_now in theory
    Pk_model = (Pk_t-1.0)*np.exp(k_t*k_t *(mu_t*mu_t*(sigma_xy+sigma_z)*(sigma_xy-sigma_z)-sigma_xy**2.0)/2.0)
    diff = Pk_model - (Pk_obs-1.0)
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
def write_params(filename, alpha_1, alpha_2, sigma_xy, sigma_z):
    data_m = np.array([alpha_1, alpha_2, sigma_xy, sigma_z])
    header_line=' The fitted parameters alpha_1, alpha_2, sigma_xy, sigma_z (by row), and their upward and downward one sigma error'
    np.savetxt(filename, data_m, fmt='%.7f', header=header_line, comments='#')


# MCMC routine
def mcmc_routine(k_range, mu_range, Pk_wnw_obs, ivar_Pk_wnow, ofile_params, space_id):
    
#    alpha_1, alpha_2, sigma_xy, sigma_z = 0.9, 1.1, 10.0, 10.0
#    
#    result = op.minimize(chi2, [alpha_1, alpha_2, sigma_xy, sigma_z], args=(k_range, mu_range, Pk_wnw_obs, tck_Pk_true, ivar_Pk_wnow))
#    alpha_1, alpha_2, sigma_xy, sigma_z = result["x"]
#    print(alpha_1, alpha_2, sigma_xy, sigma_z)

    # Set up the sampler.
    ndim, nwalkers = 4, 48
    # sometimes sigma_xy or sigma_z obtained from optimize routine is negative, we should use abs() passing to the emcee routine.
    pos = [[np.random.uniform(0.9,1.1), np.random.uniform(0.9,1.1), np.random.uniform(0.1, 15.0), np.random.uniform(0.1, 15.0)] for i in xrange(nwalkers)]
    print(np.shape(pos))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=3.0, args=(k_range, mu_range, Pk_wnw_obs, tck_Pk_true, ivar_Pk_wnow))
    print(type(sampler))
    
    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    sampler.reset()
    
    walkersteps = 4000
    sampler.run_mcmc(pos, walkersteps)  # the chain member has the shape: (nwalkers, nlinks, dim)
    print("Done.")
    print("Autocorrelation time: ", sampler.acor)
    print(sampler.get_autocorr_time(window=50, fast=False))  # not sure about the function of window
    print ("Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))
    
    sample_chains = sampler.chain[:, walkersteps/2:, :]
    scalereduction = gelman_rubin_convergence(sample_chains, nwalkers, np.size(sample_chains, axis=1), ndim)
    print "Scalereduction: ", scalereduction
    
    burnin = 2000
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    # Compute the quantiles.
    alpha_1, alpha_2, sigma_xy, sigma_z = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                              zip(*np.percentile(samples, [15.865, 50, 84.135], axis=0)))
                                              
    # output parameters into a file
    write_params(ofile_params, alpha_1, alpha_2, sigma_xy, sigma_z)
      
    print("""MCMC result:
          alpha_1={0[0]}+{0[1]}-{0[2]}
          alpha_2={1[0]}+{1[1]}-{1[2]}
          sigma_xy={2[0]}+{2[1]}-{2[2]}
          sigma_z={3[0]}+{3[1]}-{3[2]}""".format(alpha_1, alpha_2, sigma_xy, sigma_z))
      
    #plot_walkers(sampler, alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], name_id, ascale_id, space_id)
    #plot_triangle(samples, alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], name_id, ascale_id, space_id, mass_id)
    del sampler
    return alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0]



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
dir0='./run2_3_sub_Pk_obs_true_2d_wnw_mean_DD_ksorted_mu/'

inputf = dir0 +'kaver.wnw_mean_sub_a_0.5000.dat'
k_p, mu_p = np.loadtxt(inputf, dtype='f8', comments='#', delimiter=' ', usecols=(0,1), unpack=True)
print(k_p, mu_p)
N_fitbin = len(k_p)
Volume = 1380.0**3.0   # the volume of simulation box


#-------------------------------------------------------------- fit FoF P_wig/P_now with mass cut ----------------------------------------------------
#def fit_individual():    # could take reference from mcmc_fit_fofmasscut.py


def fit_mean():
    rec_id = 0
    odir = './{}_params_fitted_mean_dset_fofPkobs_subPktrue/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    np.random.seed()
    for z_id in xrange(3):
        for space_id in xrange(1):
            ifile_Pk = './{}_sub_Pk_obs_true_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wnw_mean_sub_a_{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            # the variabel name seems confusing, Pk_wnw_obs means Pk from simulation's power spectrum with shot noise subtracted.
            Pk_wnw_obs = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(3,)) # be careful that there are k, \mu, P(k, \mu)_obs, (k, \mu)_true columns.
            
            ifile_Cov_Pk = './{}_Cov_sub_Pk_true_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wnw_mean_sub_a_{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
            ivar_Pk_wnw = N_dataset/np.diag(Cov_Pk_wnw)                              # the mean sigma error
            ofile_params = odir + '{}kave{}.wnw_mean_sub_a_{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            alpha_1, alpha_2, sigma_xy, sigma_z = mcmc_routine(k_p, mu_p, Pk_wnw_obs, ivar_Pk_wnw, ofile_params, space_id)
            chi_square = chi2([alpha_1, alpha_2, sigma_xy, sigma_z], k_p, mu_p, Pk_wnw_obs, tck_Pk_true, ivar_Pk_wnw)
            print(chi_square)
            print('Reduced chi2: ', chi_square/(N_fitbin-4), '\n')

def main():
    t0 = time.clock()
    #fit_individual()
    fit_mean()
    t1 = time.clock()
    print(t1-t0)

if __name__ == '__main__':
    main()


