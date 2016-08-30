# Copy the same code from reconstruct_run2_power on 06/30/2016, modify it to get mean P(k) and its covariance matrix of run2 and run3 data.
# 1. modify it to generate the mean observed and true P_now, P_wig and P_wig/P_now of run2 and run3 simulations, 07/14/2016. 
#
#!/usr/bin/env python
import os
import time
import numpy as np

N_skip_header = 10         # for reading in P(k, \mu)
N_skip_footer = 411        # for reading in number of halos selected.
N_dataset = 20

sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = [['N1', 'W1'], ['N3', 'W3']] # NW, WG for run2; N3, W3 for run3.

sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']

mcut_Npar_list = [[37, 149, 516, 1524, 3830],
                  [35, 123, 374, 962, 2105],
                  [34, 103, 290, 681, 1390]]
N_masscut = np.size(mcut_Npar_list, axis=1)

dir0='/Users/ding/Documents/playground/WiggleNowiggle/DESI_Yu/'

inputf = dir0 +'/Analysisz0/z0mcut37/ALL/Rpkr.N10_fof_1.0000ga.dat'
data_m = np.genfromtxt(inputf, dtype='f8', comments='#', delimiter='', skip_header=N_skip_header) # skip the first data row, the first 10 rows are comments.
#print(data_m)
num_kbin = np.size(data_m, axis=0)

Volume = 1380.0**3.0   # the volume of simulation box

# define a function reading data files, rec_id=0, reading Pk without reconstruction; rec=1, with reconstruction.
def read_fofPk_file(main_dir, z_id, mcut_id, rec_id, space_id, wig_type, sim_seed_id):
    dir1 = 'Analysisz{}/'.format(sim_z[z_id])
    if z_id == 2:
        dir2 = 'z1mcut{}/'.format(mcut_Npar_list[z_id][mcut_id])
    else:
        dir2 = 'z{}mcut{}/'.format(sim_z[z_id], mcut_Npar_list[z_id][mcut_id])
    filename = '{}pk{}.{}{}_fof_{}ga.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_type, sim_seed_id, sim_a[z_id])
    inputf = main_dir + dir1 + dir2 + rec_dirs[rec_id]+'/' + filename
    print(inputf)
    
    k, Pk_obs = np.genfromtxt(inputf, dtype='f8', comments='#', delimiter='', skip_header=N_skip_header, usecols=(0, 1), unpack=True)
    N_halos = np.genfromtxt(inputf, dtype='f8', comments=None, skip_header=2, skip_footer=N_skip_footer, usecols=(3,))
    print(N_halos)
    num_den = N_halos/Volume
    # substract shot noise which is 1/n, n=N/V
    Pk_true = Pk_obs - 1/num_den
    
    return k, Pk_obs, Pk_true

def read_subPk_file(main_dir, z_id, rec_id, space_id, wig_type, sim_seed_id):
    dir1 = 'Analysisz{}/sub/'.format(sim_z[z_id])
    filename = '{}pk{}.{}{}_sub_{}ga.dat'.format(rec_fprefix[rec_id], sim_space[space_id], wig_type, sim_seed_id, sim_a[z_id])
    inputf = main_dir + dir1 + rec_dirs[rec_id]+'/' + filename
    print(inputf)
    
    k, Pk_obs = np.genfromtxt(inputf, dtype='f8', comments='#', delimiter='', skip_header=N_skip_header, usecols=(0, 1), unpack=True)
    N_halos = np.genfromtxt(inputf, dtype='f8', comments=None, skip_header=2, skip_footer=N_skip_footer, usecols=(3,))
    print(N_halos)
    num_den = N_halos/Volume
    # substract shot noise which is 1/n, n=N/V
    Pk_true = Pk_obs - 1/num_den
    
    return k, Pk_obs, Pk_true

# define a function writing output results
def write_output(odir_prefix, var_name, filename, header_line, variable):
    outputf_path = odir_prefix + var_name
    if not os.path.exists(outputf_path):
        os.makedirs(outputf_path)
    outputf = outputf_path + filename
    np.savetxt(outputf, variable, fmt='%.7e', delimiter = ' ', header=header_line, newline = '\n', comments='#')

####--------------------------------------------------------------------------------------------------------------####
####----------------------------- calculate the mean P_wig/P_now and its covariance matrix -----------------------####
def get_fof_Pk():
    t0 = time.clock()
    rec_id = 1          # rec_id=0: before reconstruction; rec_id=1: after reconstruction
    odir_prefix = './run2_3_'
    for z_id in xrange(3):
        for mcut_id in xrange(N_masscut):
            for space_id in xrange(2):
                Pk_mnow_obs = np.array([], dtype=np.float64).reshape(0, num_kbin)
                Pk_mwig_obs = np.array([], dtype=np.float64).reshape(0, num_kbin)
                Pk_mwnw_obs = np.array([], dtype=np.float64).reshape(0, num_kbin)
                Pk_mnow_true = np.array([], dtype=np.float64).reshape(0, num_kbin)
                Pk_mwig_true = np.array([], dtype=np.float64).reshape(0, num_kbin)
                Pk_mwnw_true = np.array([], dtype=np.float64).reshape(0, num_kbin)
                for run_id in xrange(2):
                    for sim_seed_id in xrange(10):
                        k_i, Pk_now_obs, Pk_now_true = read_fofPk_file(dir0, z_id, mcut_id, rec_id, space_id, sim_wig[run_id][0], sim_seed_id)
                        k_i, Pk_wig_obs, Pk_wig_true = read_fofPk_file(dir0, z_id, mcut_id, rec_id, space_id, sim_wig[run_id][1], sim_seed_id)
                        Pk_wnw_obs = Pk_wig_obs/Pk_now_obs
                        Pk_wnw_true = Pk_wig_true/Pk_now_true
                        
                        Pk_mnow_obs = np.vstack([Pk_mnow_obs, Pk_now_obs])
                        Pk_mwig_obs = np.vstack([Pk_mwig_obs, Pk_wig_obs])
                        Pk_mwnw_obs = np.vstack([Pk_mwnw_obs, Pk_wnw_obs])
                        Pk_mnow_true = np.vstack([Pk_mnow_true, Pk_now_true])
                        Pk_mwig_true = np.vstack([Pk_mwig_true, Pk_wig_true])
                        Pk_mwnw_true = np.vstack([Pk_mwnw_true, Pk_wnw_true])
            
                #print(Pk_matrix, Pk_matrix.shape)
                Pk_now_obs_mean = np.mean(Pk_mnow_obs, axis=0)
                Pk_wig_obs_mean = np.mean(Pk_mwig_obs, axis=0)
                Pk_wnw_obs_mean = np.mean(Pk_mwnw_obs, axis=0)
                Pk_now_true_mean = np.mean(Pk_mnow_true, axis=0)
                Pk_wig_true_mean = np.mean(Pk_mwig_true, axis=0)
                Pk_wnw_true_mean = np.mean(Pk_mwnw_true, axis=0)
#                for i in xrange(num_kbin):
#                    Pk_matrix[:, i] = Pk_matrix[:, i] - Pk_mean[i]
#                Cov_Pk = np.dot(Pk_matrix.T, Pk_matrix)/(N_dataset -1.0)
                #print(Cov_Pk, Cov_Pk.shape)
                
                var_name = 'Pk_1d_obs_true_mean_{}_ksorted_masscut/'.format(rec_dirs[rec_id])
                filename = "{}pk{}.now_mean_fof_a_{}_mcut{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                header_line = ' After sorting k, the mean P(k)_now in 1d\n   k       Pk_now_obs_mean     Pk_now_true_mean'
                write_output(odir_prefix, var_name, filename, header_line, np.array([k_i, Pk_now_obs_mean, Pk_now_true_mean]).T)

                filename = "{}pk{}.wig_mean_fof_a_{}_mcut{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                header_line = ' After sorting k, the mean P(k)_wig in 1d\n   k       Pk_wig_obs_mean     Pk_wig_true_mean'
                write_output(odir_prefix, var_name, filename, header_line, np.array([k_i, Pk_wig_obs_mean, Pk_wig_true_mean]).T)

                filename = "{}pk{}.wnw_obs_mean_fof_a_{}_mcut{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id], mcut_Npar_list[z_id][mcut_id])
                header_line = ' After sorting k, the mean P(k)_wnw in 1d\n   k       Pk_wnw_obs_mean     Pk_wnw_true_mean'
                write_output(odir_prefix, var_name, filename, header_line, np.array([k_i, Pk_wnw_obs_mean, Pk_wnw_true_mean]).T)
    
#                var_name = 'Cov_Pk_1d_wnw_mean_{}_ksorted_mu_masscut/'.format(rec_dirs[rec_id])
#                header_line = ' Cov(P_1d_wnw(k1), P_1d_wnw(k2)), (wnw means wig/now, 1d: k; and k1, k2 are the k bin indices).'
#                write_output(odir_prefix, var_name, header_line, rec_id, space_id, z_id, mcut_id, Cov_Pk)
    t1 = time.clock()
    print("Running time: ", t1-t0)

def get_sub_Pk():
    t0 = time.clock()
    rec_id = 0          # rec_id=0: before reconstruction; rec_id=1: after reconstruction
    odir_prefix = './run2_3_'
    for z_id in xrange(3):
        for space_id in xrange(2):
            Pk_mnow_obs = np.array([], dtype=np.float64).reshape(0, num_kbin)
            Pk_mwig_obs = np.array([], dtype=np.float64).reshape(0, num_kbin)
            Pk_mwnw_obs = np.array([], dtype=np.float64).reshape(0, num_kbin)
            Pk_mnow_true = np.array([], dtype=np.float64).reshape(0, num_kbin)
            Pk_mwig_true = np.array([], dtype=np.float64).reshape(0, num_kbin)
            Pk_mwnw_true = np.array([], dtype=np.float64).reshape(0, num_kbin)
            for run_id in xrange(2):
                for sim_seed_id in xrange(10):
                    k_i, Pk_now_obs, Pk_now_true = read_subPk_file(dir0, z_id, rec_id, space_id, sim_wig[run_id][0], sim_seed_id)
                    k_i, Pk_wig_obs, Pk_wig_true = read_subPk_file(dir0, z_id, rec_id, space_id, sim_wig[run_id][1], sim_seed_id)
                    Pk_wnw_obs = Pk_wig_obs/Pk_now_obs
                    Pk_wnw_true = Pk_wig_true/Pk_now_true
                    
                    Pk_mnow_obs = np.vstack([Pk_mnow_obs, Pk_now_obs])
                    Pk_mwig_obs = np.vstack([Pk_mwig_obs, Pk_wig_obs])
                    Pk_mwnw_obs = np.vstack([Pk_mwnw_obs, Pk_wnw_obs])
                    Pk_mnow_true = np.vstack([Pk_mnow_true, Pk_now_true])
                    Pk_mwig_true = np.vstack([Pk_mwig_true, Pk_wig_true])
                    Pk_mwnw_true = np.vstack([Pk_mwnw_true, Pk_wnw_true])
            
            #print(Pk_matrix, Pk_matrix.shape)
            Pk_now_obs_mean = np.mean(Pk_mnow_obs, axis=0)
            Pk_wig_obs_mean = np.mean(Pk_mwig_obs, axis=0)
            Pk_wnw_obs_mean = np.mean(Pk_mwnw_obs, axis=0)
            Pk_now_true_mean = np.mean(Pk_mnow_true, axis=0)
            Pk_wig_true_mean = np.mean(Pk_mwig_true, axis=0)
            Pk_wnw_true_mean = np.mean(Pk_mwnw_true, axis=0)
            
            var_name = 'Pk_1d_obs_true_mean_{}_ksorted_masscut/'.format(rec_dirs[rec_id])
            filename = "{}pk{}.now_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            header_line = ' After sorting k, the mean P(k)_now in 1d\n   k       Pk_now_obs_mean     Pk_now_true_mean'
            write_output(odir_prefix, var_name, filename, header_line, np.array([k_i, Pk_now_obs_mean, Pk_now_true_mean]).T)
            
            filename = "{}pk{}.wig_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            header_line = ' After sorting k, the mean P(k)_wig in 1d\n   k       Pk_wig_obs_mean     Pk_wig_true_mean'
            write_output(odir_prefix, var_name, filename, header_line, np.array([k_i, Pk_wig_obs_mean, Pk_wig_true_mean]).T)
            
            filename = "{}pk{}.wnw_obs_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            header_line = ' After sorting k, the mean P(k)_wnw in 1d\n   k       Pk_wnw_obs_mean     Pk_wnw_true_mean'
            write_output(odir_prefix, var_name, filename, header_line, np.array([k_i, Pk_wnw_obs_mean, Pk_wnw_true_mean]).T)



def main():
#    get_fof_Pk()
    get_sub_Pk()

if __name__ == '__main__':
    main()

