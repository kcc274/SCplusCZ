import numpy as np
import update_rule as ud


#main P method
def run_self_cali_P(cp, cov, cij_, p_covij_,N_first_theta, N_last_theta, N_sample, N_max_interation, filename,N_MC=1, N_loss_first=10
                  ,alpha=1, alpha1=1,alpha2=1,n=-1,m=1):
    cp = cp[:, :, N_first_theta:N_last_theta]
    covij = cov[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]
    cij_ = cij_[:, :, N_first_theta:N_last_theta]
    covij_ = p_covij_[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]

    _, _, N_theta = cp.shape
    if N_MC > 1:
        # 产生100组data realization
        cp_array = ud.get_cp_array(cp, covij, N_sample, N_theta, N_MC)
        cij_array = ud.get_cij_array(cij_, covij_, N_sample, N_theta, N_MC)

    else:
        cp_array = cp[:, :, :, np.newaxis]
        cij_array = cij_[:, :, :, np.newaxis]
    cr1 = np.zeros_like(cp)
    cr2 = np.zeros_like(cp)
    # MC
    P_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    cp_model_array_minJ = np.zeros_like(cp_array)
    cij_model_array_minJ = np.zeros_like(cp_array)
    J_array = np.zeros(N_MC)
    J_allMC_a2_1 = np.zeros((N_MC, N_max_interation))
    J_allMC_a2_2 = np.zeros((N_MC, N_max_interation))
    cr1_array_MC = np.zeros_like(cp_array)
    cr2_array_MC = np.zeros_like(cp_array)
    for MC in range(N_MC):
        print('===> MC #', MC)
        cp_MC = cp_array[:,:,:,MC]
        cij_MC = cij_array[:,:,:,MC]

        # 完全随机产生一些P作为初始条件
        pij = ud.diag_domin_pij_init(N_sample) # 随机赋值,对角占优矩阵
        pij = np.asarray(pij, order='C')
        for i in range(N_theta):
            cr1[:, :, i] = np.abs(
                np.matmul(np.linalg.solve(pij.transpose(), cp_MC[:, :, i]), np.linalg.pinv(pij)))
            cr2[:, :, i] = np.abs(np.matmul(cij_MC[:, :, i], np.linalg.pinv(pij)))
        Pij_output,  J, cp_model, cij_model, cr1_array,cr2_array, J1, J2 = ud.MUR_P_J(pij, cp_MC, cij_MC, cr1,cr2,
                     covij, covij_, N_max_interation, N_theta, N_sample, alpha, alpha1,alpha2,N_loss_first,n,m)
        
        J_row21 = len(J1)
        J_allMC_a2_1[MC, :J_row21] = J1
        J_row22 = len(J2)
        J_allMC_a2_2[MC, :J_row22] = J2

        P_array_minJ[:, :, MC] = Pij_output[np.argmin(J)]
        J_array[MC] = J[np.argmin(J)]
        cp_model_array_minJ[:, :, :, MC] = cp_model[np.argmin(J)]
        cij_model_array_minJ[:, :, :, MC] = cij_model[np.argmin(J)]
        cr1_array_MC[:,:,:,MC] = cr1_array[np.argmin(J)]
        cr2_array_MC[:, :, :, MC] = cr2_array[np.argmin(J)]

    np.savez(filename, cp=cp, covij=covij, cp_array=cp_array, P_array_minJ=P_array_minJ, J_array=J_array,
                      cp_model_array_minJ=cp_model_array_minJ, cij_model_array_minJ=cij_model_array_minJ,
                J1=J_allMC_a2_1, J2=J_allMC_a2_2,cp_MC=cp_MC,cij_array=cij_array,cr1_array_MC=cr1_array_MC,
                       cr2_array_MC=cr2_array_MC)


#main R method
def run_self_cali_R(cp, cov, cij_, p_covij_, cii_m,cii_logm, N_first_theta, N_last_theta, N_sample, N_max_interation, filename,N_MC=1, N_loss_first=10
                  ,alpha=1, alpha1=1,alpha2=1,n=-1,m=1):
    cp = cp[:, :, N_first_theta:N_last_theta]
    covij = cov[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]
    cij_ = cij_[:, :, N_first_theta:N_last_theta]
    covij_ = p_covij_[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]

    _, _, N_theta = cp.shape
    if N_MC > 1:
        # 产生100组data realization
        cp_array = ud.get_cp_array(cp, covij, N_sample, N_theta, N_MC)
        cij_array = ud.get_cij_array(cij_, covij_, N_sample, N_theta, N_MC)

    else:
        cp_array = cp[:, :, :, np.newaxis]
        cij_array = cij_[:, :, :, np.newaxis]

    #cp_array = np.zeros((N_sample, N_sample, N_last_theta, N_MC))
    # MC
    R_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    P_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    cp_model_array_minJ = np.zeros_like(cp_array)
    cij_model_array_minJ = np.zeros_like(cp_array)
    J_array = np.zeros(N_MC)
    J_allMC_a2_1 = np.zeros((N_MC, N_max_interation))
    J_allMC_a2_2 = np.zeros((N_MC, N_max_interation))
    for MC in range(N_MC):
        print('===> MC #', MC)
        cp_MC = cp_array[:, :, :, MC]
        cij_MC = cij_array[:, :, :, MC]

        # 完全随机产生一些P作为初始条件，找bi'，直到J增加为止, 选个最小J对应的R作为输入算法2
        pij = ud.diag_domin_pij_init(N_sample) # 随机赋值,对角占优矩阵
        N_max_b = 2000
        b_low = 0.1
        b_step = 0.001
        b_array, J_b = ud.get_b_(pij, cp_MC, cij_MC, cii_m,cii_logm,N_max_b, N_theta,N_sample,covij, covij_,b_low,b_step,alpha, alpha1,alpha2,N_loss_first,n,m)
        b_ = b_array[np.argmin(J_b)]
        Rij = pij*b_

        Rij = np.asarray(Rij, order='C')

        Rij_output, Pij_output,  J, cp_model, cij_model, J1, J2 = ud.MUR_R_J(Rij, cp_MC, cij_MC, cii_m,cii_logm,
                   covij, covij_, N_max_interation, N_theta, N_sample, alpha, alpha1,alpha2,N_loss_first,n,m)
        J_row21 = len(J1)
        J_allMC_a2_1[MC, :J_row21] = J1
        J_row22 = len(J2)
        J_allMC_a2_2[MC, :J_row22] = J2


        # 存入大矩阵
        R_array_minJ[:, :, MC] = Rij_output[np.argmin(J)]
        P_array_minJ[:, :, MC] = Pij_output[np.argmin(J)]
        J_array[MC] = J[np.argmin(J)]
        cp_model_array_minJ[:, :, :, MC] = cp_model[np.argmin(J)]
        cij_model_array_minJ[:, :, :, MC] = cij_model[np.argmin(J)]

    np.savez(filename, cp=cp, covij=covij, cp_array=cp_array,
             R_array_minJ=R_array_minJ,P_array_minJ=P_array_minJ,
             J_array=J_array, cp_model_array_minJ=cp_model_array_minJ, 
             cij_model_array_minJ=cij_model_array_minJ,
             J1=J_allMC_a2_1, J2=J_allMC_a2_2,cp_MC=cp_MC,cij_array=cij_array)


#################    #################    #################    #################    #################    ################
# the following part is for testing purposes...

#error test
def run_get_error_bar(cp, cov, cij_, p_covij_, cii_m,cii_logm,R_initial,N_first_theta, N_last_theta, N_sample, N_max_interation, filename,N_MC=1, N_loss_first=5,N_loss_last=5
                  ,alpha=1, alpha1=1,alpha2=1,n=-1,m=1):
    cp = cp[:, :, N_first_theta:N_last_theta]
    covij = cov[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]
    p_covij_ = p_covij_[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]
    cij_ = cij_[:, :, N_first_theta:N_last_theta]
    if N_loss_last-N_loss_first != 0:
        cij_[N_loss_first:N_loss_last,:,:] = 0
    _, _, N_theta = cp.shape

    # MC
    R_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    P_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    J_array = np.zeros((N_sample, N_sample, N_MC))
    CHI2_array = np.zeros((N_sample, N_sample, N_MC))
    J_allMC_a2_1 = np.zeros((N_sample, N_sample, N_MC, N_max_interation))
    J_allMC_a2_2 = np.zeros((N_sample, N_sample, N_MC, N_max_interation))

    for MC in range(N_MC):
        print('===> MC #', MC)
        # algorithm1, fixed-point-based

        Rij = R_initial - 0.001*N_MC/2 + 0.001 * MC
        Rij = np.asarray(Rij, order='C')
        for aa in range(N_sample):
            for bb in range(N_sample):
                Rij_output, Pij_output, J, CHI2, cp_model, cij_model, J1, J2 = ud.MUR_R_J_get_error_bar(Rij, cp, cij_, cii_m,
                                                                                                  cii_logm,
                                                                                                  covij, p_covij_,
                                                                                                  N_max_interation,
                                                                                                  N_theta,
                                                                                                  N_sample, aa, bb, alpha,
                                                                                                  alpha1, alpha2,
                                                                                                  N_loss_first, n, m)
                J_array[aa,bb,MC] = J[np.argmin(J)]
                CHI2_array[aa,bb,MC] = CHI2[np.argmin(CHI2)]
                R_array_minJ[aa, bb, MC] = Rij[aa, bb]
                P_array_minJ[aa, bb, MC] = Pij_output[np.argmin(J)][aa, bb]
                J_row21 = len(J1)
                J_allMC_a2_1[aa,bb,MC, :J_row21] = J1
                J_row22 = len(J2)
                J_allMC_a2_2[aa,bb,MC, :J_row22] = J2

    np.savez(filename, cp=cp, covij=covij,
             R_array_minJ=R_array_minJ,P_array_minJ=P_array_minJ,
             J_array=J_array,CHI2_array=CHI2_array, J1=J_allMC_a2_1, J2=J_allMC_a2_2)


def run_self_cali_p2(cp, cov, cij_, p_covij_,cii_m,cii_logm,N_first_theta, N_last_theta, N_sample, N_max_interation,
                     filename, N_MC=1, N_loss_first=5,N_loss_last=5, alpha=1, alpha1=1,alpha2=1,n=-1,m=1):
    cp = cp[:, :, N_first_theta:N_last_theta]
    covij = cov[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]
    cij_ = cij_[:, :, N_first_theta:N_last_theta]
    covij_ = p_covij_[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]

    if N_loss_last-N_loss_first != 0:
        cij_[N_loss_first:N_loss_last,:,:] = 0

    _, _, N_theta = cp.shape  #??? wow, interesting
    if N_MC > 1:
        # 产生100组data realization
        cp_array = ud.get_cp_array(cp, covij, N_sample, N_theta, N_MC)
        cij_array = ud.get_cij_array(cij_, covij_, N_sample, N_theta, N_MC)

    else:
        cp_array = cp[:, :, :, np.newaxis]
        cij_array = cij_[:, :, :, np.newaxis]

    # MC
    P_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    cp_model_array_minJ = np.zeros_like(cp_array)
    cij_model_array_minJ = np.zeros_like(cp_array)
    J_array = np.zeros(N_MC)
    J_allMC_a2_1 = np.zeros((N_MC, N_max_interation))
    J_allMC_a2_2 = np.zeros((N_MC, N_max_interation))
    for MC in range(N_MC):
        print('===> MC #', MC)
        cp_MC = cp_array[:, :, :, MC]
        cij_MC = cij_array[:, :, :, MC]

        # 完全随机产生一些P作为初始条件，跑算法1，直到J增加为止, 选个最小J对应的P作为输入算法2
        pij = ud.diag_domin_pij_init(N_sample) # 随机赋值,对角占优矩阵
        pij = np.asarray(pij, order='C')
        cr = cii_m*1.5

        Pij_output, J, cp_model, cij_model, cr_array, J1, J2 = ud.MUR_P_J2(pij, cp_MC, cij_MC, cr, cii_m, cii_logm,
                                                                               covij, covij_,
                                                                               N_max_interation, N_theta,
                                                                               N_sample, alpha, alpha1, alpha2,
                                                                               N_loss_first, n, m)

        J_row21 = len(J1)
        J_allMC_a2_1[MC, :J_row21] = J1
        J_row22 = len(J2)
        J_allMC_a2_2[MC, :J_row22] = J2


        # 存入大矩阵

        P_array_minJ[:, :, MC] = Pij_output[np.argmin(J)]
        J_array[MC] = J[np.argmin(J)]
        cp_model_array_minJ[:, :, :, MC] = cp_model[np.argmin(J)]
        cij_model_array_minJ[:, :, :, MC] = cij_model[np.argmin(J)]

    np.savez(filename, cp=cp, covij=covij, cp_array=cp_array, P_array_minJ=P_array_minJ,
             J_array=J_array, cp_model_array_minJ=cp_model_array_minJ,
             cij_model_array_minJ=cij_model_array_minJ,
             J1=J_allMC_a2_1, J2=J_allMC_a2_2,cp_MC=cp_MC,cij_array=cij_array)


def run_self_cali_p_for_test_a(cp, cov, cij_, p_covij_,N_first_theta, N_last_theta, N_sample, N_max_interation, N_MC=1, N_loss_first=5,N_loss_last=5
                  ,alpha=1, alpha1=1,alpha2=1,n=-1,m=1):
    cp = cp[:, :, N_first_theta:N_last_theta]
    covij = cov[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]
    cij_ = cij_[:, :, N_first_theta:N_last_theta]
    covij_ = p_covij_[:, :, N_first_theta:N_last_theta, N_first_theta:N_last_theta]

    _, _, N_theta = cp.shape
    if N_MC > 1:
        # 产生100组data realization
        cp_array = ud.get_cp_array(cp, covij, N_sample, N_theta, N_MC)
        cij_array = ud.get_cij_array(cij_, covij_, N_sample, N_theta, N_MC)

    else:
        cp_array = cp[:, :, :, np.newaxis]
        cij_array = cij_[:, :, :, np.newaxis]
    cr1 = np.zeros_like(cp)
    cr2 = np.zeros_like(cp)
    # MC
    P_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    cp_model_array_minJ = np.zeros_like(cp_array)
    cij_model_array_minJ = np.zeros_like(cp_array)
    J_array = np.zeros(N_MC)
    J_allMC_a2_1 = np.zeros((N_MC, N_max_interation))
    J_allMC_a2_2 = np.zeros((N_MC, N_max_interation))
    cr1_array_MC = np.zeros_like(cp_array)
    cr2_array_MC = np.zeros_like(cp_array)
    for MC in range(N_MC):
        cp_MC = cp_array[:,:,:,MC]
        cij_MC = cij_array[:,:,:,MC]

        # 完全随机产生一些P作为初始条件，跑算法1，直到J增加为止, 选个最小J对应的P作为输入算法2
        pij = ud.diag_domin_pij_init(N_sample) # 随机赋值,对角占优矩阵
        pij = np.asarray(pij, order='C')
        for i in range(N_theta):
            cr1[:, :, i] = np.abs(
                np.matmul(np.linalg.solve(pij.transpose(), cp_MC[:, :, i]), np.linalg.pinv(pij)))
            cr2[:, :, i] = np.abs(np.matmul(cij_MC[:, :, i], np.linalg.pinv(pij)))
        Pij_output,  J, cp_model, cij_model, cr1_array,cr2_array, J1, J2 = ud.MUR_P_J(pij, cp_MC, cij_MC, cr1,cr2,
                                                                                          covij, covij_,
                                                                                          N_max_interation, N_theta,
                                                                                          N_sample, alpha, alpha1,alpha2,N_loss_first,n,m)
        J_row21 = len(J1)
        J_allMC_a2_1[MC, :J_row21] = J1
        J_row22 = len(J2)
        J_allMC_a2_2[MC, :J_row22] = J2

        # 存入大矩阵
        P_array_minJ[:, :, MC] = Pij_output[np.argmin(J)]
        J_array[MC] = J[np.argmin(J)]
        cp_model_array_minJ[:, :, :, MC] = cp_model[np.argmin(J)]
        cij_model_array_minJ[:, :, :, MC] = cij_model[np.argmin(J)]
        cr1_array_MC[:,:,:,MC] = cr1_array[np.argmin(J)]
        cr2_array_MC[:, :, :, MC] = cr2_array[np.argmin(J)]
    return P_array_minJ

