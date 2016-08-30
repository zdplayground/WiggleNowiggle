#---------------------------------------------------------------------------------------------------
# Add shot noise term in the fitting model and use the observed P_wig/P_now fitted by the model, 07/07/2016.
# It deals with run2 and run3 data sets with 100 \mu bins and mass cut.
# 1. The minimum k of the fitting k range is the k_min of the data file.
# 2. Set the N_skip_footer and reading file process according to the data file.
# 3. Use P(k, \mu) data files with shot noise included.
# 4. Rename the code from mcmc_fit_Pkobs.py to mcmc_fit_Pkmc.py, Pkmc means Pk has mode coupling constant A. Also try
#    using the fitting model to fit the true P_wig/P_now and compare the parameters with those of the observed P_wnw, 07/09/2016.
#
# Note on 05/19/2016, using 50 walkers each with 4000 steps gives the precision 0.0001 to \alpha and
# about (0.01 - 0.1) to \Sigma
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

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
# add const: counting shot noise, the prior range is based on the maximum value of shot noise obtained from data files.
    alpha_1, alpha_2, sigma_xy, sigma_z, const = theta
    if 0.9<alpha_1<1.1 and 0.9<alpha_2<1.1 and 0.0 <=sigma_xy <=100. and 0.0<=sigma_z<100.0:
        return 0.0
    return -np.inf


# be aware Pk_obs here is the observed P_wig/P_now -1.0
def lnlike(theta, k_p, mu_p, Pk_obs, tck_Pk_wovers, tck_Pk_sm, ivar):
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

    diff = Pk_fitobs - Pk_obs
    return -0.5* np.dot(diff**2.0, ivar)

def lnprob(theta, k_p, mu_p, Pk_obs, tck_Pk_wovers, tck_Pk_sm, ivar):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k_p, mu_p, Pk_obs, tck_Pk_wovers, tck_Pk_sm, ivar)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)


# Gelman&Rubin convergence cretia, referenced from Florian's code RSDfit_challenge_hex_steps_fc_hoppper.py
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


def write_params(filename, alpha_1, alpha_2, sigma_xy, sigma_z, const):
    data_m = np.array([alpha_1, alpha_2, sigma_xy, sigma_z, const])
    header_line=' The fitted parameters alpha_1, alpha_2, sigma_xy, sigma_z, const (by row), and their upward and downward one sigma error'
    #header_line += ' with \chi^2 = {0:.3f}.'.format(chi2_value)
    np.savetxt(filename, data_m, fmt='%.7f', header=header_line, comments='#')

def mcmc_routine(k_p, mu_p, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_wovers, tck_Pk_sm, z_id, space_id):
    
    alpha_1, alpha_2, sigma_xy, sigma_z, const = 0.9, 1.1, 10.0, 10.0, 0.0
    
    result = op.minimize(chi2, [alpha_1, alpha_2, sigma_xy, sigma_z, const], args=(k_p, mu_p, Pk_wnow_obs, tck_Pk_wovers, tck_Pk_sm, ivar_Pk_wnow))
    alpha_1, alpha_2, sigma_xy, sigma_z, const = result["x"]
    print(alpha_1, alpha_2, sigma_xy, sigma_z, const)
    
    # Set up the sampler.
    ndim, nwalkers = 5, 50
    # sometimes sigma_xy or sigma_z obtained from optimize routine is negative, we should use abs() passing to the emcee routine.
    pos = [[np.random.uniform(0.9,1.1), np.random.uniform(0.9,1.1), np.random.uniform(0.1, 15.0), np.random.uniform(0.1, 15.0), np.random.uniform(-1.0, 1.0)] for i in xrange(nwalkers)]
    print(np.shape(pos))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=3.0, args=(k_p, mu_p, Pk_wnow_obs, tck_Pk_wovers, tck_Pk_sm, ivar_Pk_wnow))
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
    print("Scalereduction: ", scalereduction)
    
    burnin = 2000
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    # Compute the quantiles.
    alpha_1, alpha_2, sigma_xy, sigma_z, const = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                              zip(*np.percentile(samples, [15.865, 50, 84.135], axis=0)))
    
#    print("""MCMC result:
#          alpha_1={0[0]}+{0[1]}-{0[2]}
#          alpha_2={1[0]}+{1[1]}-{1[2]}
#          sigma_xy={2[0]}+{2[1]}-{2[2]}
#          sigma_z={3[0]}+{3[1]}-{3[2]}
#          const={6[0]}+{6[1]}-{6[2]}""".format(alpha_1, alpha_2, sigma_xy, sigma_z, const))

    del sampler
    
    return alpha_1, alpha_2, sigma_xy, sigma_z, const
    

#################################################################################################
######################################--------main code---------#################################
#################################################################################################
# simulation run name
sim_run = 'run2_3'
N_dataset = 20
N_mu_bin = 100
#N_skip_header = 11
#N_skip_footer = 31977

sim_z=['0', '0.6', '1.0']
sim_a = ['1.0000', '0.6250', '0.5000']
sim_seed = [0, 9]
sim_wig = ['NW', 'WG']
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
tck_Pk_wovers = interpolate.splrep(k_smooth, Pk_wovers)
tck_Pk_sm = interpolate.splrep(k_smooth, Pk_smooth)

# firstly, read one file and get k bins we want for the fitting range
##dir0='./run2_3_Pk_obs_2d_wnw_mean_ALL_ksorted_mu_masscut/'
##dir0 = './run2_3_sub_Pk_obs_true_2d_wnw_mean_ALL_ksorted_mu/'

inputf = './run2_3_fof_Pk_obs_true_2d_wnw_mean_ALL_ksorted_mu_masscut/Rkaver.wnw_mean_fof_a_0.5000_mcut34.dat'
k_p, mu_p = np.loadtxt(inputf, dtype='f8', comments='#', delimiter=' ', usecols=(0,1), unpack=True)
print(k_p, mu_p)

Volume = 1380.0**3.0   # the volume of simulation box

#np.random.seed(123)
odir = './params_obs_mc_fitted_mean_ksorted_mu_masscut/'
##odir = './params_true_mc_fitted_mean_ksorted_mu_masscut/'
if not os.path.exists(odir):
    os.makedirs(odir)

#---------------------------------------------------- Part 1, fit fof P(k) with mass cut ---------------------------------------------
def fit_fof():
    rec_id = 1
    dir0 = './run2_3_fof_Pk_obs_true_2d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
    for mcut_id in xrange(4,5):
        np.random.seed()
        for z_id in xrange(3):
            for space_id in xrange(1): # fit it in real space
                ifile_Pk = './{}_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                ##ifile_Pk = dir0+ '{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                print(ifile_Pk)
                Pk_wnow_obs = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns, also P_obs(c2) or P_true(c3)
                Pk_wnow_obs = Pk_wnow_obs - 1.0
                
                ifile_Cov_Pk = './{}_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                ##ifile_Cov_Pk = './{}_Cov_Pk_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wnw_mean_fof_a_{}_mcut{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                Cov_Pk_wnow = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
                ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnow)                              # the mean sigma error

                alpha_1, alpha_2, sigma_xy, sigma_z, const = mcmc_routine(k_p, mu_p, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_wovers, tck_Pk_sm, z_id, space_id)
                #theta = [alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], const[0]]
                print(alpha_1, alpha_2, sigma_xy, sigma_z, const)
                
                #chi2_value = chi2(theta, k_p, mu_p, Pk_wnow_obs, tck_Pk_wovers, ivar_Pk_wnow)
                #print chi2_value
                
                # output parameters into a file
                ofile_params= odir +'{}kave{}.wnw_mean_fof_a_{}_mcut{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                write_params(ofile_params, alpha_1, alpha_2, sigma_xy, sigma_z, const)

#---------------------------------------------------------- Part 2, fit subsample P(k)----------------------------------------------------
def fit_sub():
    rec_id = 1
    np.random.seed()
    for z_id in xrange(3):
        for space_id in xrange(2): # fit it in real space
            ifile_Pk = dir0+ '{}kave{}.wnw_mean_fof_a_{}ga.dat'.format( rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            print(ifile_Pk)
            Pk_wnow_obs = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns, also P_obs(c2) or P_true(c3)
            Pk_wnow_obs = Pk_wnow_obs - 1.0
            
            ifile_Cov_Pk = './{}_sub_Cov_Pk_obs_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wnw_mean_sub_a_{}.dat'.format(sim_run, rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            Cov_Pk_wnow = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
            ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnow)                              # the mean sigma error
            
            alpha_1, alpha_2, sigma_xy, sigma_z, const = mcmc_routine(k_p, mu_p, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_wovers, tck_Pk_sm, z_id, space_id)
            #theta = [alpha_1[0], alpha_2[0], sigma_xy[0], sigma_z[0], const[0]]
            print(alpha_1, alpha_2, sigma_xy, sigma_z, const)
            
            #chi2_value = chi2(theta, k_p, mu_p, Pk_wnow_obs, tck_Pk_wovers, ivar_Pk_wnow)
            #print chi2_value
            
            # output parameters into a file
            ofile_params= odir +'{}kave{}.wnw_mean_sub_a_{}_params.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            write_params(ofile_params, alpha_1, alpha_2, sigma_xy, sigma_z, const)

def main():
    t0 = time.clock()
    fit_fof()
    #fit_sub()
    t1 = time.clock()
    print(t1-t0)

if __name__ == '__main__':
    main()


