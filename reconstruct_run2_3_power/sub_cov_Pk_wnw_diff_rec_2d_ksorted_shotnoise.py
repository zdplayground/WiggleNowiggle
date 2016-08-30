# Copy the code sub_cov_Pk_rec_2d_ksorted_shotnoise.py on 07/14/2016. This code is corresponding to cov_Pk_wnw_diff_rec_2d_ksorted_shotnoise_masscut.py.
# 1. Modify it to get the mean subsample (P_wig-P_now)/G^2b^2 of run2 and run3 together.

#!/usr/bin/env python
import os
import time
import numpy as np
from scipy import interpolate
import growth_fun
from growth_fun import growth_factor

Omega_m = 0.3075
G_0 = growth_factor(0.0, Omega_m) # G_0 at z=0, normalization factor

N_skip_header = 11         # for reading in P(k, \mu)
N_skip_footer = 31977      # for reading in number of halos selected.
N_dataset = 20

sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = [['N1', 'W1'], ['N3', 'W3']] # NW, WG for run2; N3, W3 for run3.
sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']


inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_smooth_z0.dat'
k_smooth, Pk_smooth = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

tck_Pk_smooth = interpolate.splrep(k_smooth, Pk_smooth, k=3)   # k=3, use cubic spline to fit
print(interpolate.splev(k_smooth, tck_Pk_smooth, der=0))
inputf = '/Users/ding/Documents/playground/WiggleNowiggle/Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)


dir0='/Users/ding/Documents/playground/WiggleNowiggle/DESI_Yu/'

inputf = dir0 +'Analysisz0/sub/ALL/Rkaver.{}0_sub_1.0000ga.dat'.format(sim_wig[0][0])
data_m = np.genfromtxt(inputf, dtype='f8', comments='#', delimiter='', skip_header=N_skip_header) # skip the first data row, the first 10 rows are comments.
#print(data_m)
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
k_p = kk[indices_p]
mu_p = data_m[indices_p, 1]
Pk_sm_obsk = interpolate.splev(k_p, tck_Pk_smooth, der=0)    # interpolate theoretical linear P(k) at points k_p
Volume = 1380.0**3.0   # the volume of simulation box

# define a function reading data files, rec_id=0, reading Pk without reconstruction; rec=1, with reconstruction.
def read_fftPk_file(main_dir, z_id, rec_id, space_id, wig_type, sim_seed_id):
    dir1 = 'Analysisz{}/sub/'.format(sim_z[z_id])
    filename = '{}kave{}.{}{}_sub_{}ga.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_type, sim_seed_id, sim_a[z_id])
    inputf = main_dir + dir1 + rec_dirs[rec_id]+'/' + filename
    print(inputf)
    
    data_m= np.genfromtxt(inputf, dtype='f8', comments='#', delimiter='', skip_header=N_skip_header)
    #data_m = data_m[~np.isnan(data_m).any(axis=1)]
    k, Pk_obs = data_m[indices_p, 0], data_m[indices_p, 2]
    N_halos = np.genfromtxt(inputf, dtype='f8', comments=None, skip_header=2, skip_footer=N_skip_footer, usecols=(3,))
    print(N_halos)
    num_den = N_halos/Volume
    # substract shot noise which is 1/n, n=N/V
    Pk_true = Pk_obs - 1/num_den
    
    return k, Pk_obs, Pk_true

# define a function writing output results
def write_output(odir_prefix, var_name, header_line, variable, filename):
    outputf_path = odir_prefix + var_name
    if not os.path.exists(outputf_path):
        os.makedirs(outputf_path)

    outputf = outputf_path + filename
    np.savetxt(outputf, variable, fmt='%.7e', delimiter = ' ', header=header_line, newline = '\n', comments='#')

####--------------------------------------------------------------------------------------------------------------####
####----------------------------- calculate the mean P_wig/P_now and its covariance matrix -----------------------####
def Pk_wnw_diff_divideby_Psm():

    rec_id = 1          # rec_id=0: before reconstruction; rec_id=1: after reconstruction
    odir_prefix = './run2_3_'
    for z_id in xrange(3):
        for space_id in xrange(1):
#            Pk_mnow_obs = np.array([], dtype=np.float64).reshape(0, n_fitbin)        # m means matrix
#            Pk_mnow_true = np.array([], dtype=np.float64).reshape(0, n_fitbin)
#            Pk_mwig_obs = np.array([], dtype=np.float64).reshape(0, n_fitbin)
#            Pk_mwig_true = np.array([], dtype=np.float64).reshape(0, n_fitbin)
            Pk_mwnw_diff_obs = np.array([], dtype=np.float64).reshape(0, n_fitbin)
            Pk_mwnw_diff_true = np.array([], dtype=np.float64).reshape(0, n_fitbin)
            
            for run_id in xrange(2):
                for sim_seed_id in xrange(10):
                    k_i, Pk_now_obs, Pk_now_true = read_fftPk_file(dir0, z_id, rec_id, space_id, sim_wig[run_id][0], sim_seed_id)
#                    Pk_mnow_obs = np.vstack([Pk_mnow_obs, Pk_now_obs])
#                    Pk_mnow_true = np.vstack([Pk_mnow_true, Pk_now_true])
                    k_i, Pk_wig_obs, Pk_wig_true = read_fftPk_file(dir0, z_id, rec_id, space_id, sim_wig[run_id][1], sim_seed_id)
#                    Pk_mwig_obs = np.vstack([Pk_mwig_obs, Pk_wig_obs])
#                    Pk_mwig_true = np.vstack([Pk_mwig_true, Pk_wig_true])

                    Pk_wnw_diff_true = (Pk_wig_true-Pk_now_true)/Pk_sm_obsk
                    Pk_mwnw_diff_true = np.vstack([Pk_mwnw_diff_true, Pk_wnw_diff_true])
        
            Pk_wnw_diff_true_mean = np.mean(Pk_mwnw_diff_true, axis=0)
            
            for i in xrange(n_fitbin):
                Pk_mwnw_diff_true[:, i] = Pk_mwnw_diff_true[:, i] - Pk_wnw_diff_true_mean[i]
            Cov_Pk = np.dot(Pk_mwnw_diff_true.T, Pk_mwnw_diff_true)/(N_dataset -1.0)
            #print(Cov_Pk, Cov_Pk.shape)
            # add the const factor with growth function and bias (b=1.0 for dark matter P(k))
            bias = 1.0
            z = float(sim_z[z_id])
            print("z=", z, "G: ", growth_factor(z, Omega_m)/G_0)
            G2b2 = (growth_factor(z, Omega_m)/G_0*bias)**2.0
            Pk_wnw_diff_true_mean = Pk_wnw_diff_true_mean/G2b2
            Cov_Pk = Cov_Pk/G2b2**2.0
            
            var_name = 'sub_Pk_2d_wnw_mean_{}_ksorted_mu/'.format(rec_dirs[rec_id])
            filename = "{}kave{}.wnw_diff_mean_a_{}ga.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            header_line = ' After sorting k, the mean (P(k)_wig-P(k)_now)/(G^2b^2) in 2d (k, mu) case\n     k       mu    Pk_wnw_diff_true_mean'
            write_output(odir_prefix, var_name, header_line, rec_id, space_id, z_id, np.array([k_p, mu_p, Pk_wnw_diff_true_mean]).T, filename)

            var_name = 'sub_Cov_Pk_2d_wnw_mean_{}_ksorted_mu/'.format(rec_dirs[rec_id])
            header_line = ' Cov(P_2d_wnw_diff(k1), P_2d_wnw_diff(k2)), (wnw_diff means wig-now, 2d: k, mu; and k1, k2 are the k bin indices).'
            write_output(odir_prefix, var_name, header_line, rec_id, space_id, z_id, Cov_Pk, filename)

####--------------------------------------------------------------------------------------------------------------####
####--------------------- calculate the mean P_wig-P_now and its covariance matrix -----------------------####
#-- This is form MCMC fitting with both the bias parameter b and b_scale, as Zvonimir suggested.
def Pk_wig_minus_now():
    rec_id = 1          # rec_id=0: before reconstruction; rec_id=1: after reconstruction
    odir_prefix = './run2_3_'
    for z_id in xrange(3):
        for space_id in xrange(1):
            Pk_mwnw_diff_obs = np.array([], dtype=np.float64).reshape(0, n_fitbin)
            for run_id in xrange(2):
                for sim_seed_id in xrange(10):
                    k_obs, Pk_now_obs, Pk_now_true = read_fftPk_file(dir0, z_id, rec_id, space_id, sim_wig[run_id][0], sim_seed_id)
                    
                    k_obs, Pk_wig_obs, Pk_wig_true = read_fftPk_file(dir0, z_id, rec_id, space_id, sim_wig[run_id][1], sim_seed_id)
                    
                    Pk_wnw_diff_obs = Pk_wig_obs-Pk_now_obs       # I think it's fine either using obs or true because of subtraction.
                    Pk_mwnw_diff_obs = np.vstack([Pk_mwnw_diff_obs, Pk_wnw_diff_obs])
            
            Pk_wnw_diff_obs_mean = np.mean(Pk_mwnw_diff_obs, axis=0)
            
            for i in xrange(n_fitbin):
                Pk_mwnw_diff_obs[:, i] = Pk_mwnw_diff_obs[:, i] - Pk_wnw_diff_obs_mean[i]
            Cov_Pk = np.dot(Pk_mwnw_diff_obs.T, Pk_mwnw_diff_obs)/(N_dataset -1.0)   # it's correct
            print(Cov_Pk, Cov_Pk.shape)
            
            var_name = 'sub_Pk_2d_wnw_mean_{}_ksorted_mu/'.format(rec_dirs[rec_id])
            filename = "{}kave{}.wig_minus_now_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            header_line = ' After sorting k, the mean (P(k)_wig-P(k)_now) in 2d (k, mu) case\n     k       mu       Pk_wnw_diff_obs_mean'
            write_output(odir_prefix, var_name, header_line, np.array([k_p, mu_p, Pk_wnw_diff_obs_mean]).T, filename)
            
            var_name = 'sub_Cov_Pk_2d_wnw_mean_{}_ksorted_mu/'.format(rec_dirs[rec_id])
            filename = "{}kave{}.wig_minus_now_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            header_line = ' Cov(P_obs_2d_wnw_diff(k1), P_obs_2d_wnw_diff(k2)), (wnw_diff means Pwig-Pnow, 2d: k, mu; and k1, k2 are the k bin indices).'
            write_output(odir_prefix, var_name, header_line, Cov_Pk, filename)


def main():
    t0 = time.clock()
    #Pk_wnw_diff_divideby_Psm()
    Pk_wig_minus_now()
    t1 = time.clock()
    print("Running time: ", t1-t0)

if __name__ == '__main__':
    main()

