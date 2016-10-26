#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Copy the code from post_mcmc_fit_wig-now_selfset_HS.py, add multiple Monte carlo chains. -- 10/25/2016.
# Modified it to fit params for FoF galaxy and subsampled dark matter power spectrum. It works for both real and redshift sapce as long as parameter indices are set correspodingly.
# We can set parameters alpha_1(alpha_perp), alpha_2(alpha_para), Sigma_xy, Sigma_z, b and b_scale as free or fixed by setting params_indices.
# The fitting model is
# (P_wig - P_now)_obs = (1+b_scale *k^2)G^2 *(b_0+ f* mu^2*(1-S(k)))^2 /(1+k^2 *mu^2*sigma_fog^2)^2*(Plin - Psm) C_G^2, where both b_0, b_scale, f, and sigma_fog and Smoothing factor are the fitting parameters.
# Compared with the real space, there are three more parameters f, Sigma_sm counting Kaiser effect and sigma_fog counting FoG effect, respectively.
#
# 1. The fitting k range is from the k_min of the data file to about 0.3 h/Mpc.
# (2. Made the prior range broader and tested the fitting result is consistent with that from narrow prior range, 08/16/2016.)
# 3. When we change params_indices, we need to check whether function lnlike needs to be modified or not.
# (4. Use some subroutines wittern in Fortran. Try the case fixing \Sigma_zy=\Sigma_z=7.8364 (in real space), obtained from the equation given by Zvonimir
#    \Sigam^2/2 = 1/3 \int d^3 q P_lin/q^2 (1 - j_0[ R_bao * q] ) ~ 30 (Mpc/h)^2.)
# 5. Note that b_0 is obtained either from large scale range or fitted as a free parameter.
# 6. Fix the prior of \Sigma_xy and \Sigma_z as positive. This may solve the puzzle why the error bar of \Sigma is huge and asymmetric, 09/19/2016.
# 7. Run the code by chmod +x filename.py;
#    For example, ./filename.py -rec_id 0/1 -set_Sigma_xyz_theory 'False' -set_Sigma_sm_theory 'True'
#  ! for pre-reconstruction, set Sigma_sm=0, which gives 1-S(k)=0.
#

from __future__ import print_function
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
from post_lnprob_module_rsd_HS import match_params, cal_pk_model, lnprior
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import argparse



# !-----------------------------------------------------------------------------------------!
# !----- Check the setting alpha_1 == alpha_2 or not, depending on the params_indices set---!
# !-----------------------------------------------------------------------------------------!
# input Pk_obs = \hat{P}_wig - \hat{P}_now; sigma_fog represents \Sigma_s in the damping term of finger-of-god
def lnlike(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf):
    alpha_1, alpha_2, sigma_xy, sigma_z, sigma_sm, sigma_fog, f, b_0, b_scale = match_params(theta, params_indices, fix_params)
    # set alpha_1=alpha_2 and sigma_xy = sigma_z
    #alpha_1 = alpha_2  # be careful to comment it if both alpha_1 and alpha_2 are free parameters.
    ##sigma_xy = sigma_z
    coeff = 1.0/alpha_1*(1.0+mu_p**2.0*(pow(alpha_1/alpha_2, 2.0)-1.0))**0.5
    k_t = k_p*coeff
    mu_t = mu_p/(alpha_2*coeff)
    Pk_linw = interpolate.splev(k_t, tck_Pk_linw, der=0)
    Pk_sm = interpolate.splev(k_t, tck_Pk_sm, der=0)

    Pk_model = cal_pk_model(Pk_linw, Pk_sm, k_t, mu_t, sigma_xy, sigma_z, sigma_sm, sigma_fog, f, b_0, b_scale, norm_gf)
    diff = Pk_model - Pk_obs
    return -0.5* np.sum(diff**2.0 *ivar)

def lnprob(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf):
    lp = lnprior(theta, params_indices, fix_params)
    if (lp < -1.e20):
        return -np.inf
    return lp + lnlike(theta, params_indices, fix_params, k_p, mu_p, Pk_obs, ivar, tck_Pk_linw, tck_Pk_sm, norm_gf)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)


# Gelman&Rubin convergence criterion, copied from Florian's code RSDfit_challenge_hex_steps_fc_hoppper.py
# It's a little bit different from the function that I used before.
def gelman_rubin_convergence(withinchainvar, meanchain, n, Nchains, ndim):

    # Calculate Gelman & Rubin diagnostic
    # 1. Remove the first half of the current chains
    # 2. Calculate the within chain and between chain variances
    # 3. estimate your variance from the within chain and between chain variance
    # 4. Calculate the potential scale reduction parameter

    meanall = np.mean(meanchain, axis=0)
    W = np.mean(withinchainvar, axis=0)
    B = np.arange(ndim,dtype=np.float)
    for jj in range(0, ndim):
        B[jj] = 0.
    for jj in range(0, Nchains):
        B = B + n*(meanall - meanchain[jj])**2/(Nchains-1.)
    estvar = (1. - 1./n)*W + B/n
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
def mcmc_routine(ndim, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name, pool):
    ti = time.clock()

    Nchains = 4
    minlength = 1000
    epsilon = 0.01
    ichaincheck = 50
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    #result = op.fmin_powell(chi2, theta, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf))
    # result = op.minimize(chi2, theta, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf), method='Powell')
    # theta_optimize = result["x"]
    # print("Optimized parameters from Powell: ", theta_optimize) # only output parameters which are free to change

    theta_optimize = theta
    num_alpha = params_indices[0] + params_indices[1]
    num_Sigma = sum(params_indices[2: 2+3])     # we have 3 parameters for \Sigma, if \Sigma_sm is fixed.
    print("# of Sigma params: ", num_Sigma)
    # sometimes sigma_xy or sigma_z obtained from optimize routine is negative, we should use abs() passing to the emcee routine.
    if params_indices[2] == 1 or params_indices[3] == 1: #
        for i in xrange(num_alpha, num_alpha+num_Sigma):
            theta_optimize[i] = abs(theta_optimize[i])

    print("Initial parameters for MCMC: ", theta_optimize)

    pos = []
    sampler = []
    rstate = np.random.get_state()
    # Set up the sampler.
    for jj in xrange(Nchains):
        pos.append([theta_optimize + params_T*np.random.uniform(-1.0,1.0, ndim) for i in xrange(N_walkers)])

        sampler.append(emcee.EnsembleSampler(N_walkers, ndim, lnprob, a=2.0, args=(params_indices, fix_params, k_range, mu_range, Pk_wnow_obs, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf), pool=pool))
    print(type(sampler))

    # Clear and run the production chain.
    print("Running MCMC...")

    withinchainvar = np.zeros((Nchains,ndim))
    meanchain = np.zeros((Nchains,ndim))
    scalereduction = np.arange(ndim,dtype=np.float)
    for jj in range(0, ndim):
        scalereduction[jj] = 2.

    itercounter = 0
    chainstep = minlength
    loopcriteria = 1
    while loopcriteria:
        itercounter = itercounter + chainstep
        print("chain length =",itercounter," minlength =",minlength)

        for jj in xrange(Nchains):
            # Since we write the chain to a file we could put storechain=False, but in that case
            # the function sampler.get_autocorr_time() below will give an error
            for result in sampler[jj].sample(pos[jj], iterations=chainstep, rstate0=np.random.get_state(), storechain=True, thin=1):
                pos[jj] = result[0]
                #print(pos)
                chainchi2 = -2.*result[1]
                rstate = result[2]

            # we do the convergence test on the second half of the current chain (itercounter/2)
            chainsamples = sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim))
            #print("len chain = ", chainsamples.shape)
            withinchainvar[jj] = np.var(chainsamples, axis=0)
            meanchain[jj] = np.mean(chainsamples, axis=0)

        scalereduction = gelman_rubin_convergence(withinchainvar, meanchain, itercounter/2, Nchains, ndim)
        print("scalereduction = ", scalereduction)

        loopcriteria = 0
        for jj in range(0, ndim):
            if np.absolute(1.0-scalereduction[jj]) > epsilon:
                loopcriteria = 1

        chainstep = ichaincheck

    print("Done.")

    # Close the processes.
    #pool.close()

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional vector.
    for jj in range(0, Nchains):
        print("Mean acceptance fraction for chain ", jj,": ", np.mean(sampler[jj].acceptance_fraction))
    # Estimate the integrated autocorrelation time for the time series in each parameter.
    #for jj in range(0, Nchains):
    #    print("Autocorrelation time for chain ", jj,": ", sampler[jj].get_autocorr_time())


    ###################################
    ## Compute the quantiles ##########
    ###################################

    #samples=[]
    mergedsamples=[]

    for jj in range(0, Nchains):
        #samples.append(sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim)))
        mergedsamples.extend(sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim)))
    print("length of merged chain = ", sum(map(len,mergedsamples))/ndim)

    theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(mergedsamples, [15.86555, 50, 84.13445], axis=0)))

    print("MCMC result: ")
    for i in xrange(len(theta)):
        print("{0}={1[0]}+{1[1]}-{1[2]}".format(params_name[i], theta_mcmc[i]))

    del sampler
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
    parser = argparse.ArgumentParser(description='This is the MCMC code to get the fitting parameters, made by Zhejie Ding.')
    parser.add_argument('-rec_id', "--rec_id", help='The id of reconstruction, either 0 or 1.', required=True)   #0: pre-reconstruct; 1: post-reconstruct
    parser.add_argument('-space_id', "--space_id", help='0 for real space, 1 for redshift space.', required=True)
    parser.add_argument('-set_Sigma_xyz_theory', "--set_Sigma_xyz_theory", help='Determine whether the parameters \Sigma_xy and \Sigma_z are fixed or not, either True or False', required=True)
    parser.add_argument('-set_Sigma_sm_theory', "--set_Sigma_sm_theory", help='Determine whether we use sigma_sm from theory in the fitting model. \
                         If False, sigma_sm=0 (be careful that sigma_sm=\inf in real space case)', required=True)
    args = parser.parse_args()
    print("args: ", args)

    rec_id = int(args.rec_id)
    space_id = int(args.space_id)
    set_Sigma_xyz_theory = args.set_Sigma_xyz_theory
    set_Sigma_sm_theory = args.set_Sigma_sm_theory
    print("rec_id: ", rec_id, "space_id: ", space_id)
    print("set_Sigma_xyz_theory: ", set_Sigma_xyz_theory, "set_Sigma_sm_theory: ", set_Sigma_sm_theory)

    N_walkers = 40   # increase N_walkers would decrease the minimum number of walk steps which make fitting parameters convergent, but running time increases.
    N_walkersteps = 5000
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


    print("N_walkers: ", N_walkers, "N_walkersteps: ", N_walkersteps, "\n")
    if rec_id == 0:
        ##Sigma_0 = 8.3364           # the approximated value of \Sigma_xy and \Sigma_z, unit Mpc/h, at z=0.
        Sigma_0 = 7.8364               # suggested by Zvonimir, at z=0
    elif rec_id == 1:
        Sigma_0 = 2.84

    ##space_id = 1                       # in redshift space
    # 0: parameter fixed, 1: parameter free.
    #params_indices = [1, 1, 1, 1, 1, 1]  # It doesn't fit \Sigma and bscale well. Yes, it dosen't work well (it's kind of overfitting).
    ##params_indices = [1, 1, 1, 1, 0, 1, 1, 0, 0]    # b0 needs to be fitted. For this case, make sure \Sigma_xy and \Sigma_z positive, which should be set in mcmc_routine.
    ##params_indices = [0, 1, 0, 1, 1, 0]      # make sure \alpha_xy = \alpha_z and \Sigma_xy = \Sigma_z

    params_indices = [1, 1, 1, 1, 0, 0, 0, 0, 0]      # Set sigma_fog=0, f=0, b_scale=0, b_0=1.0 for subsamled DM case in real space.
    ##params_indices = [1, 1, 0, 0, 1, 1]    # with fixed Sigma from theoretical value, then need to set sigma_xy, sigma_z equal to Sigma_z
    ##params_indices = [0, 1, 0, 0, 1, 1]    # For this case, make sure \alpha_1 = \alpha_2 in the function lnlike(..) and set sigma_xy, sigma_z equal to Sigma_z.
    print("params_indices: ", params_indices)

    ##alpha_1, alpha_2, sigma_fog, f, b_0, b_scale  = 1.0, 1.0, 2.0, 0.2, 1.0, 0.0
    alpha_1, alpha_2, sigma_fog, f, b_0, b_scale  = 1.0, 1.0, 0.0, 0.0, 1.0, 0.0  # ! only for real space, i.e., set sigma_fog, f equal to 0.
    all_names = "alpha_1", "alpha_2", "sigma_xy", "sigma_z", "sigma_sm", "sigma_fog", "f", "b_0", "b_scale"    # the same order for params_indices
    all_temperature = 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.1

    pool = MPIPool(loadbalance=True)
    for z_id in xrange(3):
        norm_gf = growth_factor(float(sim_z[z_id]), Omega_m)/G_0
        Sigma_z = Sigma_0 * norm_gf/2.0      # divided by 2.0 for estimated \Sigma of post-reconstruction
        ##Sigma_z = Sigma_0* norm_gf

        if set_Sigma_xyz_theory == "True":
            print("Sigma_z: ", Sigma_z)
            sigma_xy, sigma_z = Sigma_z, Sigma_z
        else:
            if params_indices[2] == 0:
                sigma_xy = 0.0
            else:
                sigma_xy = 10.0
            if params_indices[3] == 0:
                sigma_z = 0.0
            else:
                sigma_z = 10.0

        np.random.seed()
#        Set it for FoF fitting
#        for mcut_id in xrange(N_masscut):
#            if set_Sigma_sm_theory == "True":
#                sigma_sm = (float(Sigma_RR_list[z_id][mcut_id])*2.0)**0.5
#            else:
#                sigma_sm = 0.0
#
#            all_params = alpha_1, alpha_2, sigma_xy, sigma_z, sigma_sm, sigma_fog, f, b_0, b_scale
#            N_params, theta, fix_params, params_T, params_name = set_params(all_params, params_indices, all_names, all_temperature)
#
#            ifile_Pk = './run2_3_Pk_obs_2d_wnw_mean_{}_ksorted_mu_masscut/{}kave{}.wig_minus_now_mean_fof_a_{}_mcut{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
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
#            if set_Sigma_xyz_theory == "False":
#                ofile_params = odir + 'fof_{}kave{}.wnw_diff_a_{}_mcut{}_params{}_Sigma_sm{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], ''.join(map(str, params_indices)), round(sigma_sm,3))
#            else:
#                ofile_params = odir + 'fof_{}kave{}.wnw_diff_a_{}_mcut{}_params{}_isotropic_Sigmaz_{}_Sigma_sm{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id], ''.join(map(str, params_indices)), round(Sigma_z, 3), round(sigma_sm,3))
#            print(ofile_params)
#            write_params(ofile_params, params_mcmc, params_name, reduced_chi2)
        # set it for DM subsample fitting
        sub_sigma_sm = (sub_Sigma_RR*2.0)**0.5
        print("sub_sigma_sm: ", sub_sigma_sm)
        all_params = alpha_1, alpha_2, sigma_xy, sigma_z, sub_sigma_sm, sigma_fog, f, b_0, b_scale  # set \Sigma_sm = sqrt(50*2)=10, for post-rec
        N_params, theta, fix_params, params_T, params_name = set_params(all_params, params_indices, all_names, all_temperature)

        # Fit for DM power spectrum
        ifile_Pk = './run2_3_sub_Pk_2d_wnw_mean_{}_ksorted_mu/{}kave{}.wig_minus_now_mean_sub_a_{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
        Pk_wnw_diff_true = np.loadtxt(ifile_Pk, dtype='f4', comments='#', usecols=(2,)) # be careful that there are k, \mu, P(k, \mu) columns.

        ifile_Cov_Pk = './run2_3_sub_Cov_Pk_2d_wnw_{}_ksorted_mu/{}kave{}.wig_minus_now_mean_sub_a_{}.dat'.format(rec_dirs[rec_id], rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
        Cov_Pk_wnw = np.loadtxt(ifile_Cov_Pk, dtype='f4', comments='#')
        ivar_Pk_wnow = N_dataset/np.diag(Cov_Pk_wnw)                              # the mean sigma error

        params_mcmc = mcmc_routine(N_params, N_walkers, N_walkersteps, theta, params_T, params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf, params_name, pool)
        chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_p, mu_p, Pk_wnw_diff_true, ivar_Pk_wnow, tck_Pk_linw, tck_Pk_sm, norm_gf)
        reduced_chi2 = chi_square/(N_fitbin-N_params)
        print('Reduced chi2: {}\n'.format(reduced_chi2))
        if rec_id == 1:
            if set_Sigma_xyz_theory == "False":
                ofile_params = odir + 'sub_{}kave{}.wnw_diff_a_{}_params{}_Sigma_sm{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)), round(sub_sigma_sm,3))
            else:
                ofile_params = odir + 'sub_{}kave{}.wnw_diff_a_{}_params{}_isotropic_Sigmaz_{}_Sigma_sm{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)), round(Sigma_z, 3), round(sub_sigma_sm,3))
        elif rec_id == 0:
            if set_Sigma_xyz_theory == "False":
                ofile_params = odir + 'sub_{}kave{}.wnw_diff_a_{}_params{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)))
            else:
                ofile_params = odir + 'sub_{}kave{}.wnw_diff_a_{}_params{}_isotropic_Sigmaz_{}.dat'.format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], ''.join(map(str, params_indices)), round(Sigma_z, 3))

#        write_params(ofile_params, params_mcmc, params_name, reduced_chi2)
    pool.close()


def main():
    t0 = time.clock()
    fit_subsamplefof_mean()
    t1 = time.clock()
    print(t1-t0)

if __name__ == '__main__':
    main()
