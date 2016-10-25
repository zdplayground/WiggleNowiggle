#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# !--- Don't copy this code to post-reconstruction case, or copy the code from post-reconstruction's after modification----!
# Copy from the code pre_mcmc_fit_wig-now_selfset_ZV.py, -- 10/14/2016.
# Modify it correspondingly to fit params for post-reconstruction with mass cuts in redshift space using Zvonimir's model.
# We can set parameters alpha_1(alpha_perp), alpha_2(alpha_para), Sigma_0, Sigma_2, Sigma_4, b_0, f, b_scale0, b_scale2, b_scale4 as free parameters and Sigma_sm as
# fixed from Hee-Jong's reconstruction process.
# The fitting model for post-reconstruction is
#
# 1. The fitting k range is from the k_min of the data file to about 0.3 h/Mpc.
# (2. Made the prior range broader and tested the fitting result is consistent with that from narrow prior range, 08/16/2016.)
# 3. When we change params_indices, we need to check whether function lnlike needs to be modified or not.
# (4. Use some subroutines wittern in Fortran. Try the case fixing \Sigma_zy=\Sigma_z=7.8364 (in real space), obtained from the equation given by Zvonimir
#    \Sigam^2/2 = 1/3 \int d^3 q P_lin/q^2 (1 - j_0[ R_bao * q] ) ~ 30 (Mpc/h)^2.)
# 5. Note that b_0 is obtained either from large scale range or fitted as a free parameter.
# 6. Fix the prior of \Sigma_xy and \Sigma_z as positive. This may solve the puzzle why the error bar of \Sigma is huge and asymmetric, 09/19/2016.
# 7. Run the code by chmod +x filename.py; ./filename.py -rec_id 0/1 -set_Sigma_sm_theory True
#

#from __future__ import print_function
import emcee
from emcee.utils import MPIPool
#import corner
import time
import numpy as np
import scipy.optimize as op
from scipy import interpolate
import os, sys
#sys.path.append('../subsample_FoF_data/')
from growth_fun import growth_factor
from post_lnprob_module_rsd_ZV import match_params, cal_pk_model, lnprior
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import argparse

# !-----------------------------------------------------------------------------------------!
# !----- Check the setting alpha_1 == alpha_2 or not, depending on the params_indices set---!
# !-----------------------------------------------------------------------------------------!
# input Pk_obs = \hat{P}_wig - \hat{P}_now; sigma_fog represents \Sigma_s in the damping term of finger-of-god
def lnlike(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf):
    alpha_1, alpha_2, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4 = match_params(theta, params_indices, fix_params)
    # Set alpha_1=alpha_2 if we want the isotropic case.
    ##alpha_1 = alpha_2  # be careful to comment it if both alpha_1 and alpha_2 are free parameters.
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    Pk_linw = interpolate.splev(k_t, tck_Pk_linw, der=0)
    Pk_sm = interpolate.splev(k_t, tck_Pk_sm, der=0)
    
    Pk_model = cal_pk_model(Pk_linw, Pk_sm, k_t, mu_t, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4, norm_gf)
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
    header_line = '# The fitted parameters {} (by row) with their upward and downward one sigma error, and the reduced \chi^2.\n'.format(params_name[:])
    with open(filename, 'w') as fwriter:
        fwriter.write(header_line)
        for i in xrange(len(params_name)):
            fwriter.write("#{0}: {1:.7f} {2:.7f} {3:.7f}\n".format(params_name[i], params_mcmc[i][0], params_mcmc[i][1], params_mcmc[i][2]))
        fwriter.write("#reduced_chi2: {0:.7f}".format(reduced_chi2))


# MCMC routine
def mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name, pool):
    ti = time.clock()
    
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    #result = op.fmin_powell(chi2, theta, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf))
    # Here we assume op.minimize works well to estimate parameters.
    result = op.minimize(chi2, theta, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf), method='Powell')
    theta_optimize = result["x"]
    print("Optimized parameters from Powell: ", theta_optimize) # only output parameters which are free to change

    theta_optimize = theta
    num_alpha = params_indices[0] + params_indices[1]
    num_Sigma = sum(params_indices[2: 2+4])   # we have 4 parameters for \Sigma
    print("# of Sigma params: ", num_Sigma)
    if params_indices[2] == 1 or params_indices[3] == 1:
        for i in xrange(num_alpha, num_alpha+num_Sigma):
            theta_optimize[i] = abs(theta_optimize[i])
    print("Initial parameters for MCMC: ", theta_optimize)
    
    # Set up the sampler.
    # sometimes sigma_xy or sigma_z obtained from optimize routine is negative, we should use abs() passing to the emcee routine.
    p0 = [theta_optimize + params_T*np.random.uniform(-1.0,1.0,N_params) for i in xrange(N_walkers)]
    #print(p0, np.shape(p0))
    # use MPI in emcee
    sampler = emcee.EnsembleSampler(N_walkers, N_params, lnprob, a=2.0, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf), pool=pool)
    print(type(sampler))
    
    # Clear and run the production chain.
    print("Running MCMC...")
    pos, prob, state = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    
    sampler.run_mcmc(pos, N_walkersteps, rstate0=state)  # the chain member has the shape: (N_walkers, nlinks, dim)
    print("Done.")
    print("Autocorrelation time: ", sampler.acor) # show the autocorrelation time
    #print(sampler.get_autocorr_time(window=50, fast=False))  # not sure about the function of window
    print ("Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))
    
    burnin = N_walkersteps/5
    sample_chains = sampler.chain[:, burnin:, :]
    scalereduction = gelman_rubin_convergence(sample_chains, N_walkers, np.size(sample_chains, axis=1), N_params)
    print "Scalereduction: ", scalereduction
    
    samples = sampler.chain[:, burnin:, :].reshape((-1, N_params))
    ##samples = sampler.chain[:, 1000:, :].reshape((-1, N_params))
    # Compute the quantiles.
    theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.865, 50, 84.135], axis=0)))
    print(theta_mcmc)
    print("MCMC result: ")
    for i in xrange(len(theta)):
        print("{0}={1[0]}+{1[1]}-{1[2]}".format(params_name[i], theta_mcmc[i]))

    del sampler
#    pool.close()
    tf = time.clock()
    print("One mcmc running set time: ", tf-ti)
    return np.array(theta_mcmc)



# Define a function to set parameters which are free and which are fixed.
def set_params(all_params, params_indices, all_names, all_temperature):
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
    return N_params, theta, fix_params, params_T, params_name

#################################################################################################
######################################--------main code---------#################################
#################################################################################################
#------- fit the FoF and DM Subsample mean P(k, mu)_wig - P_now by G^2 b^2(P_linw - P_sm)*C_G^2 with isotropic constraints -------#
def fit_subsamplefof_mean():
    parser = argparse.ArgumentParser(description='This is the MCMC code to get the fitting parameters using Zvonimir model, made by Zhejie Ding.')
    parser.add_argument('-rec_id', "--rec_id", help='The id of reconstruction, either 0 or 1.', required=True)   #0: pre-reconstruct; 1: post-reconstruct
    args = parser.parse_args()
    rec_id = int(args.rec_id)
    print("rec_id: ", rec_id)
    
    N_walkers = 200
    N_walkersteps = 20000
    space_id = 1                       # in redshift space
    print("N_walkers: ", N_walkers, "N_walkersteps: ", N_walkersteps, "\n")
    
    # 0: parameter fixed, 1: parameter free.
    params_indices = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]    # b0 needs to be fitted. For this case, make sure \Sigma_0,2,4 positive, which should be set in mcmc_routine.
    ##params_indices = [0, 1, 0, 0, 1, 1]    # For this case, make sure \alpha_1 = \alpha_2 in the function lnlike(..) and set sigma_xy, sigma_z equal to Sigma_z.
    print("params_indices: ", params_indices)
    
    # simulation run name
    N_dataset = 20
    N_mu_bin = 100
    #N_skip_header = 11
    #N_skip_footer = 31977
    Omega_m = 0.3075
    G_0 = growth_factor(0.0, Omega_m) # G_0 at z=0, normalization factor
    Volume = 1380.0**3.0   # the volume of simulation box


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

    # Sigma_sm = sqrt(2.* Sig_RR) in post-reconstruction case, for pre-reconstruction, we don't use sub_Sigma_RR.
    Sigma_RR_list = [[37, 48.5, 65.5, 84.2, 110],
                     [33, 38, 48.5, 63.5, 91.5],
                     [31, 38, 49, 65, 86]]
    sub_Sigma_RR = 50.0       # note from Hee-Jong's recording

    inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
    k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    tck_Pk_sm = interpolate.splrep(k_smooth, Pk_smooth)

    inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
    k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    tck_Pk_linw = interpolate.splrep(k_wiggle, Pk_wiggle)


    # firstly, read one file and get k bins we want for the fitting range
    dir0='/Users/ding/Documents/playground/WiggleNowiggle/subsample_FoF_data_HS/Pk_obs_2d_wnw_mean_DD_ksorted_mu_masscut/'
    inputf = dir0 +'fof_kaver.wnw_diff_a_0.6250_mcut35_fraction0.126.dat'
    k_p, mu_p = np.loadtxt(inputf, dtype='f8', comments='#', delimiter=' ', usecols=(0,1), unpack=True)
    #print(k_p, mu_p)
    N_fitbin = len(k_p)
    #print('# of (k, mu) bins: ', N_fitbin)

    # for output parameters fitted
    odir = './params_{}_wig-now_b_bscale_fitted_mean_dset/'.format(rec_dirs[rec_id])
    if not os.path.exists(odir):
        os.makedirs(odir)
    

    alpha_1, alpha_2, sigma_0, sigma_2, sigma_4, f, b_0, b_scale0, b_scale2, b_scale4  = 1.0, 1.0, 8.0, 8.0, 8.0, 0.2, 1.0, 0.0, 0.0, 0.0
    all_names = "alpha_1", "alpha_2", "sigma_0", "sigma_2", "sigma_4", "sigma_sm", "f", "b_0", "b_scale0", "b_scale2", "b_scale4"    # the same order for params_indices
    all_temperature = 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0
    pool = MPIPool(loadbalance=True)
    for z_id in xrange(1):
        norm_gf = growth_factor(float(sim_z[z_id]), Omega_m)/G_0
#        for mcut_id in xrange(N_masscut):
#            np.random.seed()
#            sigma_sm = (float(Sigma_RR_list[z_id][mcut_id])*2.0)**0.5
#            all_params = alpha_1, alpha_2, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4
#            N_params, theta, fix_params, params_T, params_name = set_params(all_params, params_indices, all_names, all_temperature)
#
#            ifile_Pk = './run2_3_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
#            print(ifile_Pk)
#            Pk_wnw_diff_obs = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.
#
#            ifile_Cov_Pk = './run2_3_Cov_Pk_obs_2d_wnw_{}_ksorted_mu_masscut/{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
#            Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
#            ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                                   # the mean sigma error
#
#            params_mcmc = mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name, pool)
#
#            chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf)
#            reduced_chi2 = chi_square/(N_fitbin-N_params)
#            print("Reduced chi2: {}\n".format(reduced_chi2))
#            # output parameters into a file
#            ofile_params = odir + 'ZV_fof_{}kave{}.wnw_diff_a_{}_mcut{}_params{}_Sigma_sm{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], ''.join(map(str, params_indices)), round(sigma_sm, 3))
#            write_params(ofile_params, params_mcmc, params_name, reduced_chi2)

        np.random.seed()
        sigma_sm = 10.0  # for all redshifts, it's the same value.
        all_params = alpha_1, alpha_2, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4
        N_params, theta, fix_params, params_T, params_name = set_params(all_params, params_indices, all_names, all_temperature)
        
        # Fit for DM power spectrum
        ifile_Pk = './run2_3_sub_Pk_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wig_minus_now_mean_sub_a_{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
        Pk_wnw_diff_true = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.
        print(ifile_Pk)
        ifile_Cov_Pk = './run2_3_sub_Cov_Pk_2d_wnw_{}_ksorted_mu/{}kave{}.wig_minus_now_mean_sub_a_{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
        Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
        ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                              # the mean sigma error

        params_mcmc = mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name, pool)
        chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf)
        reduced_chi2 = chi_square/(N_fitbin-N_params)
        print('Reduced chi2: {}\n'.format(reduced_chi2))
        ofile_params = odir + 'ZV_sub_{}kave{}.wnw_diff_a_{}_params{}_Sigma_sm{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)), round(sigma_sm, 3))
        write_params(ofile_params, params_mcmc, params_name, reduced_chi2)
    pool.close()

def main():
    t0 = time.clock()
    fit_subsamplefof_mean()
    t1 = time.clock()
    print("Running time: ", t1-t0)

if __name__ == '__main__':
    main()



