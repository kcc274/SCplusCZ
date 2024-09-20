#!/usr/bin/env python
import run_sc_cz as rsc
import numpy as np

### This is the accompanying code of the paper http://arxiv.org/abs/2409.12009
### We refer to the equations in this paper below


# theta range (N_first_theta,N_last_theta)
# the index of the theta array 
N_first_theta=0    
N_last_theta=24

# the corresponding angle of the theta array, linear binning is assumed
start_theta = 0.3
end_theta = 5.0
step_theta = 0.2

# the number of times that the iterative multiplicative update rules will be applied 
N_max_interation=100  

N_MC=100          # number of random runs to get the best fit and error bar


#spec z missing bin.  e.g. if we have 10 bins, then for full bin, it should be 10; missing bin 10, this should be 9; missing bin 6-10,  N_loss_first=6; only SC mean N_loss_first=0)
N_loss_first=10   

# does this assume that the number of bins are 10 ???


alpha=0# weight eq.21; alpha=0, CZ only

alpha1=1 # weight eq.23
alpha2=1 # weight eq.23

# the same for both J1 and J2 
n=-1    #eq.16 error bar index
m=1     #eq.16 weight function index

P_R_flag=2                    # 1 for P method, 2 for R method

# this is the mock index of the 
Mock0 = 153
Mock1 = 153
Bad_mocks = [230, 238, 386]


def get_theta_range(start=start_theta, end=end_theta, step=step_theta):
    theta_range = np.arange(start, end, step)
    return theta_range


def read_Cpp_input(nmock):
    fdir = "/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/input_data/bin_10/cp"
    fname = "mock_%d.npz" % (nmock)
    full_fname = "%s/%s" % (fdir, fname)
    cp_load = np.load(full_fname)
    cp = cp_load['array']
    return cp

def read_Csp_input(nmock):
    fdir = "/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/input_data/bin_10/cij_/logM_1p"
    fname = "mock_%d.npz" % (nmock)
    full_fname = "%s/%s" % (fdir, fname)
    cij_load = np.load(full_fname)
    cij_ = cij_load['array']
    return cij_

def read_Cii_logM(nmock):
    fdir = "/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/input_data/bin_10/cii_logM/logM_1p"
    fname = "mock_%d.npz" % (nmock)
    full_fname = "%s/%s" % (fdir, fname)
    data = np.load(full_fname)
    cii_logM = data['array']
    return cii_logM


def read_Cdata_cov():
    cp_cov_input_load = np.load('/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/input_data/bin_10/cov/cp_cov/cov_mock100.npz')
    cp_cov_input = cp_cov_input_load['cov_mock']

    cij_cov_input_load = np.load('/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/input_data/bin_10/cov/cij_cov/cov_mock100_1p_logM_cij_.npz')
    cij_cov_input = cij_cov_input_load['cov_mock']
    
    return cp_cov_input, cij_cov_input


def read_Cii_matter():
    cii_m_load = np.load('/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/input_data/bin_10/cii_m/Cii_m_bin10.npz')
    cii_m = cii_m_load['cii_m']
    
    return cii_m  


if __name__=="__main__":
    mock_list = [i for i in range(Mock0, Mock1 + 1) if i not in Bad_mocks]
    Ntotal = len(mock_list)
    print(Ntotal)
    cp_cov_input, cij_cov_input = read_Cdata_cov()
    for Nmock in range(Mock0, Mock1+1):
        if Nmock in Bad_mocks:
            continue
        output_filename = '/home/zwl/PycharmProjects/pythonProject/code/self_calibration/sc_cz/result/bin_10_cz_R/logM_1p/Mock_new_{}'.format(Nmock)

        Cpp_input = read_Cpp_input(Nmock)        # the name cp_input and input_Cij_ ???
        Csp_input = read_Csp_input(Nmock)

        if P_R_flag==2:
            cii_logm = read_Cii_logM(Nmock)
        
        N_sample, N_sample, N_all_theta = Cpp_input.shape
        print(N_sample)

        if P_R_flag==2:
            cii_m = read_Cii_matter()

        if P_R_flag==1:
            rsc.run_self_cali_P( Cpp_input, cp_cov_input, Csp_input, cij_cov_input, N_first_theta, N_last_theta, N_sample,
                         N_max_interation, output_filename,N_MC, N_loss_first,alpha,alpha1,alpha2,n,m )
        else:
            rsc.run_self_cali_R( Cpp_input, cp_cov_input, Csp_input, cij_cov_input, cii_m, cii_logm, N_first_theta, N_last_theta,
                          N_sample, N_max_interation,output_filename, N_MC, N_loss_first, alpha, alpha1, alpha2, n,m)
        
