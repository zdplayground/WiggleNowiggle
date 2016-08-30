# Calculate \sigma_v^2 = \Sigma/2 for the theoretical wiggled power spectrum and DM wiggled power spectrum, 08/21/2016
import numpy as np
import os
import math
import scipy
from scipy import interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
from growth_fun import growth_factor
import matplotlib.pyplot as plt

sim_z=['0', '0.6', '1.0']
sim_seed = [0, 9]
sim_wig = ['NW', 'WG']
sim_a = ['1.0000', '0.6250', '0.5000']
sim_space = ['r', 's']     # r for real space; s for redshift space
title_space = ['in real space', 'in redshift space']
rec_dirs = ['DD', 'ALL']   # "ALL" folder stores P(k, \mu) after reconstruction process, while DD is before reconstruction.
rec_fprefix = ['', 'R']

R_bao = 110.                # unit: Mpc/h
space_id = 0
k_cut = 0.3
ns = 0.965
Omega_m = 0.3075
G_0 = growth_factor(0.0, Omega_m) # G_0 at z=0, normalization factor
G_z = np.array([growth_factor(float(z), Omega_m)/G_0 for z in sim_z])
print("Growth factor: ", G_z)

inputf = '../Zvonimir_data/planck_camb_56106182_matterpower_z0.dat'
k_wiggle, Pk_wiggle = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
tck_Pk_linw = interpolate.splrep(k_wiggle, Pk_wiggle, k=3)
print(k_wiggle[0], k_wiggle[-1], len(k_wiggle))

const = 1./(6.0*math.pi**2.0)
##sigma_v2_err = quad(lambda x: interpolate.splev(x, tck_Pk_linw, der=0), k_wiggle[0], k_wiggle[-1])
sigma_v2_err = quad(lambda x: interpolate.splev(x, tck_Pk_linw, der=0)*(1.0-math.sin(x*R_bao)/(x*R_bao)), k_wiggle[0], 100.0) # the expression is from Zvonimir's suggestion.
print(sigma_v2_err)
sigma_v2 = sigma_v2_err[0]*const
Sigma_Plinw = (2*sigma_v2)**0.5
print("Sigma_Plinw: ", Sigma_Plinw)
print("Sigma(z): ", Sigma_Plinw*G_z)

dir0 = "/Users/ding/Documents/playground/WiggleNowiggle/reconstruct_run2_3_power/"

tranfer_fun_file = dir0+"transfer_fun_Planck2015_TT_lowP.dat"
k_trans, Trans_fun = np.loadtxt(tranfer_fun_file, dtype='f8', comments='#', unpack=True)
tck_Tf_k = interpolate.splrep(k_trans, Trans_fun)

# This extrapolate method P(k)=a*k^b is not accurate. Use trans_fun_Pk instead.
def extrapolate_Pk(k_cut, sub_kwig, sub_Pk_wig, tck_sub_Pk_wig):
    k_m = k_cut
    Pk_m = interpolate.splev(k_m, tck_sub_Pk_wig, der=0)
    Pk_m_slop = interpolate.splev(k_m, tck_sub_Pk_wig, der=1)
    print("Pk slop at k_m: ", Pk_m_slop)
    n_s = (k_m * Pk_m_slop)/Pk_m
    const_a = Pk_m/(k_m ** n_s)
    kwig_ext = np.linspace(k_m+1.e-3, 1.0, 500)
    Pk_wig_ext = const_a * kwig_ext ** n_s
    sub_kwig = np.append(sub_kwig, kwig_ext)
    sub_Pk_wig = np.append(sub_Pk_wig, Pk_wig_ext)
    return sub_kwig, sub_Pk_wig

# Apply transfer function to estimate P(k) for k>0.3 h/Mpc in order to realize the extrapolation of simulated P_wig.
def trans_fun_Pk(k_cut, sub_kwig, sub_Pk_wig, tck_sub_Pk_wig, tck_Tf_k, ns):
    k_m = k_cut
    Pk_m = interpolate.splev(k_m, tck_sub_Pk_wig, der=0)
    Tf_0 = interpolate.splev(k_m, tck_Tf_k, der=0)
    Pk_0 = Pk_m/(pow(k_m, ns) * Tf_0**2.0)
    print(Pk_0)
    kwig_ext = np.linspace(k_m+1.e-3, 100.0, 200)
    Pk_wig_ext = Pk_0 * kwig_ext**ns * interpolate.splev(kwig_ext, tck_Tf_k, der=0)**2.0
    #print(Pk_wig_ext)
    sub_kwig = np.append(sub_kwig, kwig_ext)
    sub_Pk_wig = np.append(sub_Pk_wig, Pk_wig_ext)
    return sub_kwig, sub_Pk_wig

# Define a function to plot Sigma (theoretical and simulated) in terms of z.
def plot_Sigma_z(sim_z, sub_Sigma, Sigma_Plinw, odir, figname):
    G_index = 1.
    z_axis = np.linspace(0.0, 1.2, 200)
    fit_Sigma = np.array([(growth_factor(z, Omega_m)/G_0)**G_index for z in z_axis])*sub_Sigma[0]/G_z[0]**G_index
    # We don't show error from the integration, because it's not one sigma error.
    plt.plot(0.0, Sigma_Plinw, 'rs', label=r'$\Sigma(0)$ from $P_{\mathtt{lin,w}}$')
    plt.plot(map(float, sim_z[:]), sub_Sigma, 'ks', label=r'$\Sigma(z)$ from $\hat{P}_{\mathtt{sim,w}}$')
    plt.plot(z_axis, fit_Sigma, 'g--', label=r'$\Sigma_0 G(z)$')
    plt.legend(frameon=False, loc="upper right", fontsize=20)
    plt.title(r"$\Sigma$ calculated {}".format(title_space[0]), fontsize=20)
    plt.xlim([-0.05, 1.2])
    plt.xlabel(r"$z$", fontsize=24)
    plt.ylim([4.0, 9.0])
    plt.ylabel(r"$\Sigma$ $[Mpc/h]$", fontsize=24)
    plt.savefig(odir+figname)
    plt.show()
    plt.close()



def cal_sub_Sigma():
    for rec_id in xrange(2):
        idir = dir0+"run2_3_Pk_1d_obs_true_mean_{}_ksorted_masscut/".format(rec_dirs[rec_id])
        inputf = idir+"{}pk{}.wig_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[0])
        sub_kwig, sub_Pk_wig = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0,2), unpack=True)
        for i in xrange(len(sub_kwig)):
            if sub_kwig[i] > k_cut:
                break
        index_boundary = i
        print(index_boundary)

        sub_sigma_v2 = np.array([], dtype=float).reshape(0,2)
        for z_id in xrange(3):
            inputf = idir+"{}pk{}.wig_mean_sub_a_{}.dat".format(rec_fprefix[rec_id], sim_space[space_id], sim_a[z_id])
            sub_kwig_all, sub_Pk_wig_all = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0,2), unpack=True)
            sub_kwig = np.array(sub_kwig_all[0: index_boundary])
            sub_Pk_wig = np.array(sub_Pk_wig_all[0: index_boundary])
            
            print("k_max: ", sub_kwig[-1], "Pk_min: ", sub_Pk_wig[-1])
            tck_sub_Pk_wig = interpolate.splrep(sub_kwig, sub_Pk_wig, k=3)
            #sub_kwig_ext, sub_Pk_wig_ext = extrapolate_Pk(k_cut, sub_kwig, sub_Pk_wig, tck_sub_Pk_wig)
            sub_kwig_ext, sub_Pk_wig_ext = trans_fun_Pk(k_cut, sub_kwig, sub_Pk_wig, tck_sub_Pk_wig, tck_Tf_k, ns)
            #plt.loglog(sub_kwig_all, sub_Pk_wig_all, sub_kwig_ext, sub_Pk_wig_ext)
            #plt.show()

            tck_sub_Pk_wig_ext = interpolate.splrep(sub_kwig_ext, sub_Pk_wig_ext)
            sigma_v2_err = quad(lambda x: interpolate.splev(x, tck_sub_Pk_wig_ext, der=0), sub_kwig[0], 100.0)
            sigma_v2_err = np.array(sigma_v2_err)*const
            print(sigma_v2_err)
            sub_sigma_v2 = np.vstack([sub_sigma_v2, sigma_v2_err])

        print("sub_sigma_v2: ", sub_sigma_v2)
        sub_Sigma = (2.0* sub_sigma_v2[:,0])**0.5
        #sub_Sigma_err = sub_sigma_v2[:, 1]/(2.*Sigma)
        odir = "param_Sigma_integrate_Pkwig_ALL_DD_dset/"
        if not os.path.exists(odir):
            os.makedirs(odir)
        ofile = dir0+odir+ "{}kave{}.wig_Sigma_mean_sub.dat".format(rec_fprefix[rec_id], sim_space[space_id])
        header_line = "# \Sigma calculated from \Sigma^2/2=\sigma_v^2=(1/3)*1/(2\pi)^3 \int d^3 q Pw/q^2, the first row is the result from Plin,w (z=0); the other three rows are from simulated P_wig.\n"
        header_line += "# z(redshift)    \Sigma \n"
        with open(ofile, 'w') as fwriter:
            fwriter.write(header_line)
            fwriter.write('{} {}\n'.format(0.0, Sigma_Plinw))
            for i in xrange(3):
                fwriter.write('{} {}\n'.format(sim_z[i], sub_Sigma[i]))
        
        odir = dir0+"ALL_DD_figs_Sigma_integrate_Pkwig/"
        if not os.path.exists(odir):
            os.makedirs(odir)
        figname = "{}kave{}.wig_Sigma_mean_sub.pdf".format(rec_fprefix[rec_id], sim_space[space_id])
        plot_Sigma_z(sim_z, sub_Sigma, Sigma_Plinw, odir, figname)
    

def main():
    #cal_sub_Sigma()
    print("Done.")
                          
if __name__ == '__main__':
    main()

