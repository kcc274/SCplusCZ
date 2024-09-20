from numba import jit
import numpy as np
import copy
import main as mn


theta_range = mn.get_theta_range()


#main P method
@jit(nopython=True)
def MUR_P_J(fij, cp, cij_, cr1,cr2, covij, p_covij_, N_max_interation, N_theta, N_sample, alpha=1,alpha1=1,alpha2=1,
            N_loss_first=10,n=-1,m=1):
    J_dummy = []
    fij_dummy = []
    cp_model_dummy = []
    cr1_dummy = []
    cr2_dummy = []
    Cij_dummy = []
    J_1_dummy = []
    J_2_dummy = []
    for i in range(N_max_interation):
        if i == 0:
            fij_dummy.append(fij)
        fij_input = fij.copy()
        w = updatew(cp, cr1, cr2, fij_input, cij_, covij, p_covij_, N_theta, N_sample, alpha, alpha1, alpha2, N_loss_first,n,m)
        cii_input1 = cr1.copy()
        cii_input2 = cr2.copy()
        cr1,cr2 = updatecr(cp, cii_input1,cii_input2, cij_, w, covij, p_covij_, N_theta, N_sample, alpha1, alpha2, N_loss_first,n,m)
        fij = w
        fij_dummy.append(fij)

        # 计算 J
        J = 0.
        Jt = 0.
        cp_model = np.zeros_like(cp)
        Cij = np.zeros_like(cij_)
        Cij_ = cij_
        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    for k in range(N_sample):
                        cp_model[a,b,theta] += fij[k,a]*fij[k,b]*cr1[k,k,theta]
                    if a >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    elif b >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    else:
                        J += (theta_range[theta]**n)*alpha1/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2

        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    Cij[a, b, theta] = fij[a, b] * cr2[a, a, theta]
                    if a < N_loss_first:
                        Jt += (theta_range[theta]**n)*0.5  * (Cij_[a, b, theta] - Cij[a, b, theta]) ** 2 / (p_covij_[a, b, theta, theta])**m
        J_1_dummy.append(J)
        J_2_dummy.append(Jt)
        J_total = alpha*J+Jt
        J_dummy.append(J_total)

        cp_model_dummy.append(cp_model)
        cr1_dummy.append(cr1)
        cr2_dummy.append(cr2)
        Cij_dummy.append(Cij)

    return fij_dummy, J_dummy, cp_model_dummy, Cij_dummy, cr1_dummy, cr2_dummy,J_1_dummy, J_2_dummy


#main R method
@jit(nopython=True)
def MUR_R_J(fij, cp, cij_,cii_m,cii_logm, covij, p_covij_, N_max_interation, N_theta, N_sample, alpha=1,alpha1=1,alpha2=1,N_loss_first=10,n=-1,m=1):
    J_dummy = []
    fij_dummy = []
    Pij_dummy = []
    cp_model_dummy = []
    Cij_dummy = []
    J_1_dummy = []
    J_2_dummy = []
    for i in range(N_max_interation):
        fij_input = fij.copy()
        R = updateR(cp, cii_m,cii_logm, fij_input, cij_, covij, p_covij_, N_theta, N_sample,alpha, alpha1, alpha2, N_loss_first,n,m)
        b_ = np.sum(R,axis=0)
        pij = np.zeros_like(R)
        for aa in range(N_sample):
            for bb in range(N_sample):
                pij[aa,bb] = R[aa,bb]/b_[bb]

        Pij_dummy.append(pij)
        fij = R
        fij_dummy.append(fij)

        # 计算 J
        J = 0.
        Jt = 0.
        cp_model = np.zeros_like(cp)
        Cij = np.zeros_like(cij_)
        Cij_ = cij_
        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    for k in range(N_sample):
                        cp_model[a,b,theta] += fij[k,a]*fij[k,b]*cii_m[k,k,theta]
                    if a >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    elif b >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    else:
                        J += (theta_range[theta]**n)*alpha1/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
        b2 = np.zeros_like(fij)
        for ii in range(N_sample):
            count = 0
            for theta in theta_range:
                if theta <= 1.5:
                    b2[ii, ii] += cii_logm[ii, ii, count] / cii_m[ii, ii, count]
                    count += 1
        bb = np.sqrt(b2/count)

        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    Cij[a, b, theta] = fij[a, b] * bb[a,a] * cii_m[a, a, theta]
                    if a < N_loss_first:
                        Jt += (theta_range[theta]**n)*0.5 / (p_covij_[a, b, theta, theta])**m * (Cij_[a, b, theta] - Cij[a, b, theta]) ** 2
        J_1_dummy.append(J)
        J_2_dummy.append(Jt)
        J_total = alpha*J+Jt
        J_dummy.append(J_total)
        cp_model_dummy.append(cp_model)
        Cij_dummy.append(Cij)

    return fij_dummy,Pij_dummy,  J_dummy, cp_model_dummy, Cij_dummy, J_1_dummy, J_2_dummy


# draw random realization
def get_cp_array(cp_, covij_, N_sample_, N_theta_, N_MC_):
    cp_array = np.zeros((N_sample_, N_sample_, N_theta_, N_MC_))
    for MC in range(N_MC_):
        cp_MC = np.zeros_like(cp_)
        # 生成一组观测
        for i in range(N_sample_):
            # 随机抽样
            cp_MC[i, i, :] = np.random.multivariate_normal(cp_[i, i, :], covij_[i, i, :, :])
            for j in range(i + 1, N_sample_):
                cp_MC[i, j, :] = np.random.multivariate_normal(cp_[i, j, :], covij_[i, j, :, :])
                cp_MC[j, i, :] = copy.deepcopy(cp_MC[i, j, :])
        #cp_MC[cp_MC < 0] = 1e-5  # 非负
        # 存入大矩阵
        cp_array[:, :, :, MC] = cp_MC
    return cp_array


def get_cij_array(cij_, covij_, N_sample_, N_theta_, N_MC_):
    cij_array = np.zeros((N_sample_, N_sample_, N_theta_, N_MC_))
    for MC in range(N_MC_):
        cij_MC = np.zeros_like(cij_)
        # 生成一组观测
        for i in range(N_sample_):
            # 随机抽样
            cij_MC[i, i, :] = np.random.multivariate_normal(cij_[i, i, :], covij_[i, i, :, :])
            for j in range(i + 1, N_sample_):
                cij_MC[i, j, :] = np.random.multivariate_normal(cij_[i, j, :], covij_[i, j, :, :])
                cij_MC[j, i, :] = np.random.multivariate_normal(cij_[j, i, :], covij_[j, i, :, :])
        #cij_MC[cij_MC < 0] = 1e-5  # 非负
        # 存入大矩阵
        cij_array[:, :, :, MC] = cij_MC
    return cij_array


def get_cii_array(cii_, covii_, N_sample_, N_theta_, N_MC_):
    cii_array = np.zeros((N_sample_, N_sample_, N_theta_, N_MC_))
    for MC in range(N_MC_):
        cii_MC = np.zeros_like(cii_)
        # 生成一组观测
        for i in range(N_sample_):
            # 随机抽样
            cii_MC[i, i, :] = np.random.multivariate_normal(cii_[i, i, :], covii_[i, i, :, :])
        #cp_MC[cp_MC < 0] = 1e-5  # 非负
        # 存入大矩阵
        cii_array[:, :, :, MC] = cii_MC
    return cii_array


# generate a random diagnoal dominate matrix
def diag_domin_pij_init(N_sample):
    fij = np.random.rand(N_sample, N_sample)
    for i in range(N_sample):
        fij[i, i] = 0.6 + 0.3 * np.random.rand()
        a = np.random.rand(N_sample - 1)
        a = a / np.sum(a) * (1 - fij[i, i])
        a_sort = np.sort(a)[::-1]  # big to small
        if i == 0:
            fij[i + 1:, i] = a_sort
        elif i == N_sample - 1:
            fij[:i, i] = a_sort[::-1]
        else:
            a_sort_ = np.array([a_sort[0], a_sort[1]])
            np.random.shuffle(a_sort_)
            fij[i - 1, i] = a_sort_[0]
            fij[i + 1, i] = a_sort_[1]
            indices = list(range(i - 1)) + list(range(i + 2, N_sample))
            a_sort_ = a_sort[2:]
            np.random.shuffle(a_sort_)
            fij[indices, i] = a_sort_
    return fij


@jit(nopython=True)
def update_EM(Yk, W, Ck, nbin):
    W1 = np.copy(W)
    Rys = np.zeros_like(Yk)
    for iq in range(0, nbin):
        Ainv = np.linalg.pinv(W1)
        Rys[:, :, iq] = np.dot(Yk[:, :, iq], Ainv.T)  #
        Rsst = np.sum(Ck, axis=2)  #
        Ryst = np.sum(Rys, axis=2)
        W1 = np.copy(np.abs(np.dot(Ryst, np.linalg.pinv(Rsst))))  #

    return W1


#R method get initial bi'
@jit(nopython=True)
def get_b_(fij, cp, cij_, cii_m,cii_logm, N_max_interation, N_theta, N_sample, covij, p_covij_, b_low,b_step, alpha=1, alpha1=1,alpha2=1,N_loss_first=5,n=-1,m=1):
    J_dummy = []
    b_dummy = []

    for i in range(N_max_interation):
        b_ = b_low+b_step*i
        R = b_*fij
        b_dummy.append(b_)

        # 计算 J
        J = 0.
        Jt = 0.
        cp_model = np.zeros_like(cp)
        Cij = np.zeros_like(cij_)
        Cij_ = cij_
        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    for k in range(N_sample):
                        cp_model[a,b,theta] += R[k,a]*R[k,b]*cii_m[k,k,theta]
                    if a >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    elif b >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    else:
                        J += (theta_range[theta]**n)*alpha1/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
        b2 = np.zeros_like(fij)
        for ii in range(N_sample):
            count = 0
            for theta in theta_range:
                if theta <= 1.5:
                    b2[ii, ii] += cii_logm[ii, ii, count] / cii_m[ii, ii, count]
                    count += 1
        bb = np.sqrt(b2/count)

        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    Cij[a, b, theta] = R[a, b] * bb[a,a] * cii_m[a, a, theta]
                    if a < N_loss_first:
                        Jt += (theta_range[theta]**n)*0.5 / (p_covij_[a, b, theta, theta])**m * (Cij_[a, b, theta] - Cij[a, b, theta]) ** 2

        J_total = alpha*J+Jt
        J_dummy.append(J_total)
    return b_dummy, J_dummy



@jit(nopython=True)
def fixed_point_based_J(fij, cp, cij_, cii_m,cii_logm, N_max_interation, N_theta, N_sample, covij, p_covij_, alpha=1, alpha1=1,alpha2=1,N_loss_first=10,n=-1,m=1):
    J_dummy = []
    chi2_dummy = []
    fij_dummy = []
    for i in range(N_max_interation):
        b2 = np.zeros_like(cii_m)
        for N in range(N_sample):
            count = 0
            for theta in theta_range:
                if theta <= 1.5:
                    b2[N, N] += cii_logm[N, N, count] / cii_m[N, N, count]
                    count += 1
        bb = np.sqrt(b2 / count)
        cr = bb*cii_m
        fij_input = fij.copy()
        w = update_EM(cp, fij_input.transpose(), cr, N_theta)
        fij = w.transpose()
        fij_dummy.append(fij)

        # 计算 J

        J = 0.
        Jt = 0
        cp_model = np.zeros_like(cp)
        Cij = np.zeros_like(cij_)
        Cij_ = cij_

        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    for k in range(N_sample):
                        cp_model[a,b,theta] += fij[k,a]*fij[k,b]*cr[k,k,theta]
                    if a >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    elif b >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    else:
                        J += (theta_range[theta]**n)*alpha1/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2

        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    Cij[a, b, theta] = fij[a, b] * cr[a, a, theta]
                    if a < N_loss_first:
                        Jt += (theta_range[theta]**n)*0.5 / (p_covij_[a, b, theta, theta])**m * (Cij_[a, b, theta] - Cij[a, b, theta]) ** 2
        J_total = alpha*J + Jt
        J_dummy.append(J_total)

        # 计算\chi^2
        chi2 = 0.
        for ii in range(N_sample):
            for j in range(ii,N_sample):
                cp_diff = cp_model[ii,j,:]-cp[ii,j,:]
                chi2 = chi2 + np.dot(cp_diff, np.linalg.solve(covij[ii,j,:,:],cp_diff))

        chi2_dummy.append(chi2)

        if i > 3:
            # stop criterion
            if J_total > 1.5 * (J_dummy[i - 3] + J_dummy[i - 2] + J_dummy[i - 1]) / 3:  # 1.5 times larger than last three J
                print('==> Algorithm 1 breaks at iteration', i)
                break

    return fij_dummy, chi2_dummy, J_dummy



# update P method C'_{ii} and C^{X}_{ii}, Eq.A.10
@jit(nopython=True)
def updatecr(cp, cr1,cr2, cij_, w, covij, p_covij_, N_theta, N_sample,alpha1=1,alpha2=1,N_loss_first=10,n=-1,m=1):
    new_cr1 = np.zeros_like(cr1)
    new_cr2 = np.zeros_like(cr2)
    grad_J1_N = np.zeros_like(cr1)
    grad_J2_N = np.zeros_like(cr1)
    grad_J1_P = np.zeros_like(cr1)
    grad_J2_P = np.zeros_like(cr1)
    for a in range(N_sample):
        for theta in range(N_theta):
            for i in range(N_sample):
                for j in range(N_sample):
                    if i >= N_loss_first:
                        grad_J1_N[a, a, theta] += (theta_range[theta]**n)*cp[i, j, theta] * w[a, i] * w[a, j] * alpha2/(covij[i, j, theta, theta])**m
                    elif j >= N_loss_first:
                        grad_J1_N[a, a, theta] += (theta_range[theta]**n)*cp[i, j, theta] * w[a, i] * w[a, j] * alpha2/(covij[i, j, theta, theta])**m
                    else:
                        grad_J1_N[a, a, theta] += (theta_range[theta]**n)*cp[i,j,theta]*w[a,i]*w[a,j] * alpha1/(covij[i, j, theta, theta])**m
                    for k in range(N_sample):
                        if i >= N_loss_first:
                            grad_J1_P[a,a,theta] += (theta_range[theta]**n)*w[k,i]*w[k,j]*cr1[k,k,theta]*w[a,i]*w[a,j] * alpha2/(covij[i, j, theta, theta])**m
                        elif j >= N_loss_first:
                            grad_J1_P[a,a,theta] += (theta_range[theta]**n)*w[k,i]*w[k,j]*cr1[k,k,theta]*w[a,i]*w[a,j] * alpha2/(covij[i, j, theta, theta])**m
                        else:
                            grad_J1_P[a,a,theta] += (theta_range[theta]**n)*w[k,i]*w[k,j]*cr1[k,k,theta]*w[a,i]*w[a,j] * alpha1/(covij[i, j, theta, theta])**m
            for j in range(N_sample):
                grad_J2_N[a, a, theta] += (theta_range[theta] ** n) * cij_[a, j, theta] * w[a, j] / (p_covij_[a, j, theta, theta]) ** m
                grad_J2_P[a, a, theta] += (theta_range[theta] ** n) * w[a, j] * cr2[a, a, theta] * w[a, j] / (p_covij_[a, j, theta, theta]) ** m

            new_cr1[a, a, theta] = cr1[a, a, theta] * (grad_J1_N[a, a, theta]) / (grad_J1_P[a, a, theta])
            new_cr2[a, a, theta] = cr2[a, a, theta] * (grad_J2_N[a, a, theta]) / (grad_J2_P[a, a, theta])

    return new_cr1,new_cr2


# update P method P_{ab'}(here P is w), Eq.A.9
@jit(nopython=True)
def updatew(cp, cr1,cr2, w, cij_, covij, p_covij_, N_theta, N_sample, alpha=1, alpha1=1, alpha2=1, N_loss_first=10,n=-1,m=1):
    new_w = np.zeros_like(w)
    grad_J1_N = np.zeros_like(w)
    grad_J2_N = np.zeros_like(w)
    grad_J1_P = np.zeros_like(w)
    grad_J2_P = np.zeros_like(w)
    grad_J_N = np.zeros_like(w)
    grad_J_P = np.zeros_like(w)
    numerator_denominator = np.zeros_like(w)
    for a in range(N_sample):
        for b in range(N_sample):
            for theta in range(N_theta):
                for j in range(N_sample):
                    if j >= N_loss_first:
                        grad_J1_N[a,b] += (theta_range[theta]**n)*2*cp[b, j, theta]*w[a, j]*cr1[a, a, theta]*alpha2/(covij[b, j, theta,theta])**m
                    else:
                        grad_J1_N[a,b] += (theta_range[theta]**n)*2*cp[b, j, theta]*w[a, j]*cr1[a, a, theta]*alpha1/(covij[b, j, theta,theta])**m
            for theta in range(N_theta):
                grad_J2_N[a, b] += (theta_range[theta]**n)*cij_[a, b, theta] * cr2[a, a, theta]/(p_covij_[a, b, theta, theta])**m
            if a < N_loss_first:
                grad_J_N[a, b] = alpha*grad_J1_N[a, b] + grad_J2_N[a, b]
            else:
                grad_J_N[a, b] = grad_J1_N[a, b]
            for theta in range(N_theta):
                for j in range(N_sample):
                    for k in range(N_sample):
                        if j >= N_loss_first:
                            grad_J1_P[a,b] += (theta_range[theta]**n)*2*w[k,b]*w[k,j]*cr1[k,k,theta]*w[a,j]*cr1[a,a,theta]*alpha2/(covij[b, j, theta, theta])**m
                        else:
                            grad_J1_P[a, b] += (theta_range[theta]**n)*2 * w[k, b] * w[k, j] * cr1[k, k, theta] * w[a, j] * cr1[a, a, theta]*alpha1/(covij[b, j, theta, theta])**m
            for theta in range(N_theta):
                grad_J2_P[a, b] += (theta_range[theta]**n)*w[a, b] * cr2[a, a, theta] * cr2[a, a, theta]/(p_covij_[a, b, theta, theta])**m
            if a < N_loss_first:
                grad_J_P[a, b] = alpha*grad_J1_P[a, b] + grad_J2_P[a, b]
            else:
                grad_J_P[a,b] = grad_J1_P[a, b]
    for a in range(N_sample):
        for b in range(N_sample):
            sum_aa1 = 0
            sum_aa2 = 0
            for aa in range(N_sample):
                sum_aa1 += w[aa,b]/grad_J_P[aa,b]
            for aa in range(N_sample):
                sum_aa2 += w[aa,b]*grad_J_N[aa,b]/grad_J_P[aa,b]
            numerator = grad_J_N[a,b]*sum_aa1+1-sum_aa2
            denominator = grad_J_P[a,b]*sum_aa1
            numerator_denominator[a,b] = numerator/denominator
            new_w[a,b]=w[a,b]*numerator_denominator[a,b]

    if np.any(new_w < 0):
        for a in range(N_sample):
            for b in range(N_sample):
                sum_aa1 = 0
                sum_aa2 = 0
                for aa in range(N_sample):
                    sum_aa1 += w[aa, b] / grad_J_P[aa, b]
                for aa in range(N_sample):
                    sum_aa2 += w[aa, b] * grad_J_N[aa, b] / grad_J_P[aa, b]
                numerator = grad_J_N[a, b] * sum_aa1 + 1
                denominator = grad_J_P[a, b] * sum_aa1 + sum_aa2
                numerator_denominator[a, b] = numerator / denominator
                new_w[a, b] = w[a, b] * numerator_denominator[a, b]
    return new_w


# update R method R_{ab'}(here R is w), Eq.A.15
@jit(nopython=True)
def updateR(cp, cii_m, cii_logm, w, cij_, covij, p_covij_, N_theta, N_sample, alpha=1, alpha1=1, alpha2=1, N_loss_first=10,n=-1,m=1):
    new_w = np.zeros_like(w)
    grad_J1_N = np.zeros_like(w)
    grad_J2_N = np.zeros_like(w)
    grad_J1_P = np.zeros_like(w)
    grad_J2_P = np.zeros_like(w)
    grad_J_N = np.zeros_like(w)
    grad_J_P = np.zeros_like(w)
    b2 = np.zeros_like(w)
    for i in range(N_sample):
        count = 0
        for theta in theta_range:
            if theta <= 1.5:
                b2[i,i] += cii_logm[i, i, count]/cii_m[i, i, count]
                count += 1
    bb = np.sqrt(b2/count)
    for a in range(N_sample):
        for b in range(N_sample):
            for theta in range(N_theta):
                for j in range(N_sample):
                    if j >= N_loss_first:
                        grad_J1_N[a,b] += (theta_range[theta]**n)*2*cp[b, j, theta]*w[a, j]*cii_m[a, a, theta]*alpha2/(covij[b, j, theta,theta])**m
                    else:
                        grad_J1_N[a,b] += (theta_range[theta]**n)*2*cp[b, j, theta]*w[a, j]*cii_m[a, a, theta]*alpha1/(covij[b, j, theta,theta])**m

            for theta in range(N_theta):
                grad_J2_N[a, b] += (theta_range[theta]**n)*cij_[a, b, theta] * bb[a,a] * cii_m[a, a, theta]/(p_covij_[a, b, theta, theta])**m

            if a < N_loss_first:
                grad_J_N[a, b] = alpha*grad_J1_N[a, b] + grad_J2_N[a, b]
            else:
                grad_J_N[a, b] = grad_J1_N[a, b]
            for theta in range(N_theta):
                for j in range(N_sample):
                    for k in range(N_sample):
                        if j >= N_loss_first:
                            grad_J1_P[a,b] += (theta_range[theta]**n)*2*w[k,b]*w[k,j]*cii_m[k,k,theta]*w[a,j]*cii_m[a,a,theta]*alpha2/(covij[b, j, theta, theta])**m
                        else:
                            grad_J1_P[a, b] += (theta_range[theta]**n)*2 * w[k, b] * w[k, j] * cii_m[k, k, theta] * w[a, j] * cii_m[a, a, theta]*alpha1/(covij[b, j, theta, theta])**m

            for theta in range(N_theta):
                grad_J2_P[a, b] += (theta_range[theta]**n)*w[a, b] * bb[a,a] * cii_m[a, a, theta] * bb[a,a] * cii_m[a, a, theta]/(p_covij_[a, b, theta, theta])**m

            if a < N_loss_first:
                grad_J_P[a, b] = alpha*grad_J1_P[a, b] + grad_J2_P[a, b]
            else:
                grad_J_P[a,b] = grad_J1_P[a, b]

            new_w[a, b] = w[a, b] * grad_J_N[a,b]/grad_J_P[a,b]

    return new_w


#################    #################    #################    #################    #################    ################
# the following part is for testing purposes...
@jit(nopython=True)
def updatecr2(cp, cr,cii_logm,cii_m, cij_, w, covij, p_covij_, N_theta, N_sample,alpha=1,alpha1=1,alpha2=1,N_loss_first=5,n=-1,m=1):
    new_cr = np.zeros_like(cr)
    grad_J1_N = np.zeros_like(cr)
    grad_J2_N = np.zeros_like(cr)
    grad_J1_P = np.zeros_like(cr)
    grad_J2_P = np.zeros_like(cr)
    b2 = np.zeros_like(w)
    for i in range(N_sample):
        count = 0
        for theta in theta_range:
            if theta <= 1.5:
                b2[i,i] += cii_logm[i, i, count]/cii_m[i, i, count]
                count += 1
    bb = np.sqrt(b2/count)
    for a in range(N_sample):
        for theta in range(N_theta):
            if theta <= 1.5:
                for i in range(N_sample):
                    for j in range(N_sample):
                        if i >= N_loss_first:
                            grad_J1_N[a, a, theta] += (theta_range[theta] ** n) * cp[i, j, theta] * w[a, i] * w[
                                a, j] * alpha2 / (covij[i, j, theta, theta]) ** m
                        elif j >= N_loss_first:
                            grad_J1_N[a, a, theta] += (theta_range[theta] ** n) * cp[i, j, theta] * w[a, i] * w[
                                a, j] * alpha2 / (covij[i, j, theta, theta]) ** m
                        else:
                            grad_J1_N[a, a, theta] += (theta_range[theta] ** n) * cp[i, j, theta] * w[a, i] * w[
                                a, j] * alpha1 / (covij[i, j, theta, theta]) ** m
                        for k in range(N_sample):
                            if i >= N_loss_first:
                                grad_J1_P[a, a, theta] += (theta_range[theta] ** n) * w[k, i] * w[k, j] * cr[
                                    k, k, theta] * w[a, i] * w[a, j] * alpha2 / (covij[i, j, theta, theta]) ** m
                            elif j >= N_loss_first:
                                grad_J1_P[a, a, theta] += (theta_range[theta] ** n) * w[k, i] * w[k, j] * cr[
                                    k, k, theta] * w[a, i] * w[a, j] * alpha2 / (covij[i, j, theta, theta]) ** m
                            else:
                                grad_J1_P[a, a, theta] += (theta_range[theta] ** n) * w[k, i] * w[k, j] * cr[
                                    k, k, theta] * w[a, i] * w[a, j] * alpha1 / (covij[i, j, theta, theta]) ** m
                for j in range(N_sample):
                    grad_J2_N[a, a, theta] += 0.5 * (theta_range[theta] ** n) * cij_[a, j, theta] * bb[a, a] * w[
                        a, j] * (cii_m[a, a, theta] / cr[a, a, theta]) ** 0.5 / (p_covij_[a, j, theta, theta]) ** m
                    grad_J2_P[a, a, theta] += 0.5 * (theta_range[theta] ** n) * w[a, j] * (w[a, j] * bb[a, a]) ** 2 * \
                                              cii_m[a, a, theta] / (p_covij_[a, j, theta, theta]) ** m
                if a < N_loss_first:
                    new_cr[a, a, theta] = cr[a, a, theta] * (
                                alpha * grad_J1_N[a, a, theta] + grad_J2_N[a, a, theta]) / (
                                                      alpha * grad_J1_P[a, a, theta] + grad_J2_P[a, a, theta])
                else:
                    new_cr[a, a, theta] = cr[a, a, theta] * (grad_J1_N[a, a, theta]) / (grad_J1_P[a, a, theta])

    return new_cr



@jit(nopython=True)
def updatew2(cp, cr,cii_logm,cii_m, w, cij_, covij, p_covij_, N_theta, N_sample, alpha=1, alpha1=1, alpha2=1, N_loss_first=5,n=-1,m=1):
    new_w = np.zeros_like(w)
    grad_J1_N = np.zeros_like(w)
    grad_J2_N = np.zeros_like(w)
    grad_J1_P = np.zeros_like(w)
    grad_J2_P = np.zeros_like(w)
    grad_J_N = np.zeros_like(w)
    grad_J_P = np.zeros_like(w)
    numerator_denominator = np.zeros_like(w)
    b2 = np.zeros_like(w)
    for i in range(N_sample):
        count = 0
        for theta in theta_range:
            if theta <= 1.5:
                b2[i,i] += cii_logm[i, i, count]/cii_m[i, i, count]
                count += 1
    bb = np.sqrt(b2/count)
    for a in range(N_sample):
        for b in range(N_sample):
            for theta in range(N_theta):
                if theta <= 1.5:
                    for j in range(N_sample):
                        if j >= N_loss_first:
                            grad_J1_N[a, b] += (theta_range[theta] ** n) * 2 * cp[b, j, theta] * w[a, j] * cr[
                                a, a, theta] * alpha2 / (covij[b, j, theta, theta]) ** m
                        else:
                            grad_J1_N[a, b] += (theta_range[theta] ** n) * 2 * cp[b, j, theta] * w[a, j] * cr[
                                a, a, theta] * alpha1 / (covij[b, j, theta, theta]) ** m
            for theta in range(N_theta):
                if theta <= 1.5:
                    grad_J2_N[a, b] += (theta_range[theta] ** n) * cij_[a, b, theta] * bb[a, a] * (
                        cr[a, a, theta] * cii_m[a, a, theta]) ** 0.5 / (p_covij_[a, b, theta, theta]) ** m
            if a < N_loss_first:
                grad_J_N[a, b] = alpha*grad_J1_N[a, b] + grad_J2_N[a, b]
            else:
                grad_J_N[a, b] = grad_J1_N[a, b]

            for theta in range(N_theta):
                if theta <= 1.5:
                    for j in range(N_sample):
                        for k in range(N_sample):
                            if j >= N_loss_first:
                                grad_J1_P[a, b] += (theta_range[theta] ** n) * 2 * w[k, b] * w[k, j] * cr[k, k, theta] * \
                                                   w[a, j] * cr[a, a, theta] * alpha2 / (covij[b, j, theta, theta]) ** m
                            else:
                                grad_J1_P[a, b] += (theta_range[theta] ** n) * 2 * w[k, b] * w[k, j] * cr[k, k, theta] * \
                                                   w[a, j] * cr[a, a, theta] * alpha1 / (covij[b, j, theta, theta]) ** m

            for theta in range(N_theta):
                if theta <= 1.5:
                    grad_J2_P[a, b] += (theta_range[theta] ** n) * w[a, b] * bb[a, a] ** 2 * cr[a, a, theta] * cii_m[
                        a, a, theta] / (p_covij_[a, b, theta, theta]) ** m

            if a < N_loss_first:
                grad_J_P[a, b] = alpha*grad_J1_P[a, b] + grad_J2_P[a, b]
            else:
                grad_J_P[a,b] = grad_J1_P[a, b]
    for a in range(N_sample):
        for b in range(N_sample):
            sum_aa1 = 0
            sum_aa2 = 0
            for aa in range(N_sample):
                sum_aa1 += w[aa,b]/grad_J_P[aa,b]
            for aa in range(N_sample):
                sum_aa2 += w[aa,b]*grad_J_N[aa,b]/grad_J_P[aa,b]
            numerator = grad_J_N[a,b]*sum_aa1+1-sum_aa2
            denominator = grad_J_P[a,b]*sum_aa1
            numerator_denominator[a,b] = numerator/denominator
            new_w[a,b]=w[a,b]*numerator_denominator[a,b]

    if np.any(new_w < 0):
        for a in range(N_sample):
            for b in range(N_sample):
                sum_aa1 = 0
                sum_aa2 = 0
                for aa in range(N_sample):
                    sum_aa1 += w[aa, b] / grad_J_P[aa, b]
                for aa in range(N_sample):
                    sum_aa2 += w[aa, b] * grad_J_N[aa, b] / grad_J_P[aa, b]
                numerator = grad_J_N[a, b] * sum_aa1 + 1
                denominator = grad_J_P[a, b] * sum_aa1 + sum_aa2
                numerator_denominator[a, b] = numerator / denominator
                new_w[a, b] = w[a, b] * numerator_denominator[a, b]

    return new_w,bb

#another P method test
@jit(nopython=True)
def MUR_P_J2(fij, cp, cij_, cr, cii_logm, cii_m, covij, p_covij_, N_max_interation, N_theta, N_sample, alpha=1,alpha1=1,alpha2=1,N_loss_first=5,n=-1,m=1):
    J_dummy = []
    fij_dummy = []
    cp_model_dummy = []
    cr_dummy = []
    Cij_dummy = []
    J_1_dummy = []
    J_2_dummy = []
    for i in range(N_max_interation):
        if i == 0:
            fij_dummy.append(fij)
        fij_input = fij.copy()
        w,bb = updatew2(cp, cr,cii_logm,cii_m,fij_input, cij_, covij, p_covij_, N_theta, N_sample, alpha, alpha1, alpha2, N_loss_first,n,m)
        cii_input = cr.copy()
        cr = updatecr2(cp, cii_input,cii_logm,cii_m, cij_, w, covij, p_covij_, N_theta, N_sample, alpha, alpha1, alpha2, N_loss_first,n,m)
        fij = w
        fij_dummy.append(fij)

        # 计算 J
        J = 0.
        Jt = 0.
        cp_model = np.zeros_like(cp)
        Cij = np.zeros_like(cij_)
        Cij_ = cij_
        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    for k in range(N_sample):
                        cp_model[a,b,theta] += fij[k,a]*fij[k,b]*cr[k,k,theta]
                    if a >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    elif b >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    else:
                        J += (theta_range[theta]**n)*alpha1/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2

        for theta in range(N_theta):
            if theta <= 1.5:
                for a in range(N_sample):
                    for b in range(N_sample):
                        Cij[a, b, theta] = fij[a, b] * bb[a, a] * (cr[a, a, theta] * cii_m[a, a, theta]) ** 0.5
                        if a < N_loss_first:
                            Jt += ((theta_range[theta] ** n) * 0.5  * (Cij_[a, b, theta] - Cij[a, b, theta]) ** 2 )/ (p_covij_[a, b, theta, theta]) ** m

        J_1_dummy.append(J)
        J_2_dummy.append(Jt)
        J_total = alpha*J+Jt
        J_dummy.append(J_total)

        cp_model_dummy.append(cp_model)
        cr_dummy.append(cr)
        Cij_dummy.append(Cij)
        if i > 3:
            # stop criterion
            if J_total > 1.5*(J_dummy[i-3]+J_dummy[i-2]+J_dummy[i-1])/3: # 1.5 times larger than last three J
                print('==> Algorithm 2 breaks at iteration', i)
                break
    return fij_dummy, J_dummy, cp_model_dummy, Cij_dummy, cr_dummy, J_1_dummy, J_2_dummy





#R method error test(unimportance)
@jit(nopython=True)
def updateR_get_error_bar(cp, cii_m, cii_logm, w, cij_, covij, p_covij_, N_theta, N_sample,element1,element2, alpha=1, alpha1=1, alpha2=1, N_loss_first=5,n=-1,m=1):
    new_w = np.zeros_like(w)
    grad_J1_N = np.zeros_like(w)
    grad_J2_N = np.zeros_like(w)
    grad_J1_P = np.zeros_like(w)
    grad_J2_P = np.zeros_like(w)
    grad_J_N = np.zeros_like(w)
    grad_J_P = np.zeros_like(w)
    b2 = np.zeros_like(w)
    for i in range(N_sample):
        count = 0
        for theta in theta_range:
            if theta <= 1.5:
                b2[i,i] += cii_logm[i, i, count]/cii_m[i, i, count]
                count += 1
    bb = np.sqrt(b2/count)
    for a in range(N_sample):
        for b in range(N_sample):
            for theta in range(N_theta):
                for j in range(N_sample):
                    if j >= N_loss_first:
                        grad_J1_N[a,b] += (theta_range[theta]**n)*2*cp[b, j, theta]*w[a, j]*cii_m[a, a, theta]*alpha2/(covij[b, j, theta,theta])**m
                    else:
                        grad_J1_N[a,b] += (theta_range[theta]**n)*2*cp[b, j, theta]*w[a, j]*cii_m[a, a, theta]*alpha1/(covij[b, j, theta,theta])**m

            for theta in range(N_theta):
                grad_J2_N[a, b] += (theta_range[theta]**n)*cij_[a, b, theta] * bb[a,a] * cii_m[a, a, theta]/(p_covij_[a, b, theta, theta])**m

            if a < N_loss_first:
                grad_J_N[a, b] = alpha*grad_J1_N[a, b] + grad_J2_N[a, b]
            else:
                grad_J_N[a, b] = grad_J1_N[a, b]
            for theta in range(N_theta):
                for j in range(N_sample):
                    for k in range(N_sample):
                        if j >= N_loss_first:
                            grad_J1_P[a,b] += (theta_range[theta]**n)*2*w[k,b]*w[k,j]*cii_m[k,k,theta]*w[a,j]*cii_m[a,a,theta]*alpha2/(covij[b, j, theta, theta])**m
                        else:
                            grad_J1_P[a, b] += (theta_range[theta]**n)*2 * w[k, b] * w[k, j] * cii_m[k, k, theta] * w[a, j] * cii_m[a, a, theta]*alpha1/(covij[b, j, theta, theta])**m

            for theta in range(N_theta):
                grad_J2_P[a, b] += (theta_range[theta]**n)*w[a, b] * bb[a,a] * cii_m[a, a, theta] * bb[a,a] * cii_m[a, a, theta]/(p_covij_[a, b, theta, theta])**m

            if a < N_loss_first:
                grad_J_P[a, b] = alpha*grad_J1_P[a, b] + grad_J2_P[a, b]
            else:
                grad_J_P[a,b] = grad_J1_P[a, b]
            new_w[a, b] = w[a, b] * grad_J_N[a, b] / grad_J_P[a, b]
    new_w[element1,element2] = w[element1,element2]

    return new_w


#R method error test(unimportance)
@jit(nopython=True)
def MUR_R_J_get_error_bar(fij, cp, cij_,cii_m,cii_logm, covij, p_covij_, N_max_interation, N_theta, N_sample,element1,
                          element2, alpha=1,alpha1=1,alpha2=1,N_loss_first=5,n=-1,m=1):
    J_dummy = []
    chi2_dummy = []
    fij_dummy = []
    Pij_dummy = []
    cp_model_dummy = []
    Cij_dummy = []
    J_1_dummy = []
    J_2_dummy = []
    for i in range(N_max_interation):
        fij_input = fij.copy()
        R = updateR_get_error_bar(cp, cii_m,cii_logm, fij_input, cij_, covij, p_covij_, N_theta, N_sample,element1,element2,alpha, alpha1, alpha2, N_loss_first,n,m)
        b_ = np.sum(R,axis=0)
        pij = np.zeros_like(R)
        for aa in range(N_sample):
            for bb in range(N_sample):
                pij[aa,bb] = R[aa,bb]/b_[bb]
        Pij_dummy.append(pij)
        fij = R
        fij_dummy.append(fij)

        # 计算 J
        J = 0.
        Jt = 0.
        cp_model = np.zeros_like(cp)

        Cij = np.zeros_like(cij_)
        Cij_ = cij_
        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    for k in range(N_sample):
                        cp_model[a,b,theta] += fij[k,a]*fij[k,b]*cii_m[k,k,theta]
                    if a >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    elif b >= N_loss_first:
                        J += (theta_range[theta]**n)*alpha2/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
                    else:
                        J += (theta_range[theta]**n)*alpha1/(covij[a, b, theta, theta])**m * 0.5 * (cp[a, b, theta] - cp_model[a, b, theta]) ** 2
        b2 = np.zeros_like(fij)
        for ii in range(N_sample):
            count = 0
            for theta in theta_range:
                if theta <= 1.5:
                    b2[ii, ii] += cii_logm[ii, ii, count] / cii_m[ii, ii, count]
                    count += 1
        bb = np.sqrt(b2/count)

        for theta in range(N_theta):
            for a in range(N_sample):
                for b in range(N_sample):
                    Cij[a, b, theta] = fij[a, b] * bb[a,a] * cii_m[a, a, theta]
                    if a < N_loss_first:
                        Jt += (theta_range[theta]**n)*0.5 / (p_covij_[a, b, theta, theta])**m * (Cij_[a, b, theta] - Cij[a, b, theta]) ** 2
        J_1_dummy.append(J)
        J_2_dummy.append(Jt)
        J_total = alpha*J+Jt
        J_dummy.append(J_total)
        chi2_1 = 0
        chi2_2 = 0
        for ii in range(N_sample):
            for jj in range(N_sample):
                chi2_1 += np.dot(np.dot((cp[ii, jj, :] - cp_model[ii, jj, :]).T, np.linalg.inv(covij[ii, jj, :, :])),
                                 (cp[ii, jj, :] - cp_model[ii, jj, :]))
                if ii < N_loss_first:
                    chi2_2 += np.dot(
                        np.dot((Cij_[ii, jj, :] - Cij[ii, jj, :]).T, np.linalg.inv(p_covij_[ii, jj, :, :])),
                        (Cij_[ii, jj, :] - Cij[ii, jj, :]))

        chi2 = alpha*chi2_1+chi2_2
        chi2_dummy.append(chi2)
        cp_model_dummy.append(cp_model)
        Cij_dummy.append(Cij)

    return fij_dummy,Pij_dummy, J_dummy,chi2_dummy, cp_model_dummy, Cij_dummy, J_1_dummy, J_2_dummy
