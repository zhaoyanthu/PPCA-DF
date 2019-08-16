import pickle
import numpy as np
import pandas as pd
import copy
import random
import statistics
import math
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


def get_corrupt_data(original_matrix, cp, missing_pattern):
    """
    Return corrupted data matrix and real missing ratio if it is entire row missing
    :param original_matrix: matrix for desired formulation
    :param cp: for random missing is missing ratio test2
    :param missing_pattern: Missing at random and entire row missing
    :return:
    """
    CT_construct_flag = False
    num_row = original_matrix.shape[0]
    all_zero_sum_num = num_row * -1
    num_roll = 0

    CT = copy.deepcopy(original_matrix)
    col_idx_list_to_corrupt = range(original_matrix.shape[1])
    while (CT_construct_flag == False):
        num_roll += 1
        CT = missing_pattern(original_matrix, CT, cp, col_idx_list_to_corrupt)
        sum_column = np.sum(CT, axis=0)
        if all_zero_sum_num in sum_column:  # At least one observation is needed
            CT_construct_flag = False
            col_idx_list_to_corrupt = [col_idx for col_idx in range(original_matrix.shape[1]) if
                                       sum_column[col_idx] == all_zero_sum_num]
        else:
            CT_construct_flag = True

    # calculate real missing ratio
    if missing_pattern == entire_row_missing:
        cp = round((np.count_nonzero(CT == -1) / original_matrix.size) * 100, 2)

    return CT, cp


# separate TOD
def get_TOD_data(TT, tod_idx, num_TOD):
    if tod_idx >= num_TOD:
        assert False, 'tod_idx > num_TOD'
    original_matrix = TT[:, tod_idx::num_TOD]
    return original_matrix


def random_missing(orignal_matrix, CT, cp, col_idx_list):
    # CT = copy.deepcopy(original_matrix)
    for i in range(orignal_matrix.shape[0]):
        for j in col_idx_list:
            CT[i, j] = orignal_matrix[i, j] if random.random() > (cp / 100) else -1
    return CT


def entire_row_missing(original_matrix, CT, cp, col_idx_list):
    """
    for entire row missing don't need CT and col_idx_list, this is just for xingshitongyi
    :param original_matrix:
    :param CT:
    :param cp: num_missing_rows
    :param col_idx_list:
    :return:
    """
    CT = copy.deepcopy(original_matrix)
    num_missing_rows = cp
    row_idx_list = range(original_matrix.shape[0])
    missing_rows_idx = random.sample(row_idx_list, num_missing_rows)
    for row_idx in missing_rows_idx:
        CT[row_idx, :] = -1
    return CT


def get_probe_vehicle_data(original_matrix, prr, prr_pattern='same_prr'):
    """
    Get PVT for 3 different types of methods
    1. prr is the same for all columns
    2. prr is different for each column and generate randomly in a given range
    3. prr is different for each column and given in advance

    :param original_matrix: matrix sample from. Cannot whole row be zero
    :param prr: int for 1, [prr lb, prr ub] for 2, [prr col1, prr col2, ......, prr coln] for 3
    :param prr_pattern: same_prr, range_prr, given_prr
    :return: PVT
    """
    if (type(prr), str(prr_pattern)) not in [(int, 'same_prr'), (list, 'range_prr'), (list, 'given_prr')]:
        assert False, 'prr not consistent with prr pattern!'

    num_row, num_col = original_matrix.shape[0], original_matrix.shape[1]

    penetration_rate_list = []
    if prr_pattern == 'same_prr':
        penetration_rate_list = [(prr / 100) for i in range(num_col)]
    elif prr_pattern == 'range_prr':
        prr_lb, prr_ub = prr[0], prr[1]
        penetration_rate_list = [(random.randrange(prr_lb, prr_ub, 1) / 100) for i in range(num_col)]
    elif prr_pattern == 'given_prr':
        penetration_rate_list = [(num / 100) for num in prr]

    PVT = copy.deepcopy(original_matrix)
    for row_idx in range(num_row):
        PVT_row_construct_OK_flag = False  # for each row at least 1 observation is needed
        while PVT_row_construct_OK_flag is False:
            for col_idx in range(num_col):
                penetration_rate = penetration_rate_list[col_idx]
                PVT[row_idx, col_idx] = np.random.binomial(original_matrix[row_idx, col_idx], penetration_rate)
            row_sum = np.sum(PVT[row_idx, :])
            if row_sum != 0:
                PVT_row_construct_OK_flag = True
    return PVT


def performance_measure(TT, TT_til, CT):
    '''
    Calculate RMSE and MAPE of estimation
    :param TT: Ground truth
    :param TT_til: Estimation
    :param CT: Corrupted data
    :return: RMSE and MAPE of estimation
    '''
    num_missing_entries = np.count_nonzero(CT == -1)

    estimate_list_all = list(np.ravel(TT_til))
    ground_truth_list_all = list(np.ravel(TT))
    CT_list = list(np.ravel(CT))

    estimate_list = [estimate_list_all[i] for i in range(len(ground_truth_list_all)) if ground_truth_list_all[i] != 0 \
                     and CT_list[i] == -1]
    ground_truth_list = [ground_truth_list_all[i] for i in range(len(ground_truth_list_all)) if
                         ground_truth_list_all[i] != 0 and CT_list[i] == -1]

    mape_list = [abs(x - y) / y for x, y in zip(estimate_list, ground_truth_list)]
    MAPE = np.mean(mape_list) if mape_list else 0
    rmse_list = [(x - y) ** 2 for x, y in zip(estimate_list, ground_truth_list)]
    RMSE = np.sqrt(np.mean(rmse_list) if rmse_list else 0)

    return num_missing_entries, MAPE, RMSE


def data_fusion_PPCA_update_prr_eta2(CT, PVT, rank, tol_mu=1e-6, tol_W=1e-6, tol_sigma2=1e-6, tol_prr=1e-6,
                                     tol_eta2=1e-6, max_iter_num=1000):
    """
    PPCA-DF method
    :param CT: corrupted data matrix
    :param rank: the dimension for latent coordinates
    :param tol_mu: threshold for mu
    :param tol_W: threshold for W
    :param tol_sigma2: threshold for sigma2
    :param tol_prr: threshold for penetration rate pr
    :param tol_eta2: threshold for eta2
    :param max_iter_num: maximum iteration number
    :return: imputed matrix and converage flag
    """
    # Initialize W/mu/sigma_2/prr/eta2
    W = W_initialize(CT, rank, initial_method='random')
    sigma_2 = sigma_2_initialize()
    prr = prr_initialize(CT, PVT)
    eta2 = prr * (1 - prr)
    # calculate mu_y, xn_bar list
    mu_y = np.reshape(np.mean(PVT, axis=1), (-1, 1))
    xn_bar_list = get_xn_bar(CT, PVT)
    mu = np.reshape(np.array(copy.deepcopy(xn_bar_list)), (-1, 1))

    # dimension
    d = CT.shape[0]
    N = CT.shape[1]

    # iteration
    iter_num = 0
    converge_flag = False
    error_mu_list, error_W_list, error_sigma2_list, error_prr_list, error_eta2_list = [], [], [], [], []
    while (converge_flag == False):
        # list to store 5 expectations for all samples
        E_x_list, E_t_list, E_t_t_list, E_x_x_list, E_x_t_list = [], [], [], [], []

        # calculate joint distribution for mu and C for [x_n y_n t_n]
        C_i_joint = get_joint_distribution_data_fusion_PPCA_update_prr_eta2(CT, PVT, mu, W, sigma_2, prr, eta2)

        for i in range(N):
            # calculate 5 expectations
            E_x, E_t, E_t_t, E_x_x, E_x_t = get_5_expectation_data_fusion_PPCA(np.reshape(CT[:, i], (-1, 1)),
                                                                               np.reshape(PVT[:, i], (-1, 1)), rank, W,
                                                                               mu, mu_y, C_i_joint)
            E_x_list.append(E_x), E_t_list.append(E_t), E_t_t_list.append(E_t_t), E_x_x_list.append(
                E_x_x), E_x_t_list.append(E_x_t)

        # update parameters
        # print(iter_num)
        mu_update = 1 / N * np.sum(E_x_list, axis=0) - 1 / N * W @ (np.sum(E_t_list, axis=0))
        # print(np.linalg.inv(np.sum(E_t_t_list, axis=0)))
        W_update = (np.sum(E_x_t_list, axis=0) - mu @ (np.sum(E_t_list, axis=0)).T) @ np.linalg.inv(
            np.sum(E_t_t_list, axis=0))

        WTWE_t_t_list = [W.T @ W @ matrix for matrix in E_t_t_list]
        WE_t_x_list = [W @ matrix.T for matrix in E_x_t_list]
        sigma_2_update = 1 / N * (np.trace(np.sum(E_x_x_list, axis=0)) + np.trace(np.sum(WTWE_t_t_list, axis=0)) -
                                  2 * mu.T @ np.sum(E_x_list, axis=0) - 2 * np.trace(np.sum(WE_t_x_list, axis=0)) +
                                  2 * mu.T @ W @ np.sum(E_t_list, axis=0)) + mu.T @ mu
        sigma_2_update = sigma_2_update / d

        diag_xbar_eta_inv = np.linalg.inv(np.diag(np.array(xn_bar_list) * eta2))
        term1 = np.sum([PVT[:, n:n+1].T @ diag_xbar_eta_inv @ np.reshape(E_x_list[n], (-1, 1)) for n in range(N)])
        term2 = np.sum([np.trace(E_x_x_list[n] @ diag_xbar_eta_inv) for n in range(N)])
        prr_update = term1 / term2

        eta2_update = 0
        for n_idx in range(N):
            for i_idx in range(d):
                yni = PVT[i_idx, n_idx]
                eta2_update += (yni ** 2 - 2 * yni * prr * E_x_list[n_idx][i_idx, 0] + (prr ** 2) * E_x_x_list[n_idx][
                    i_idx, i_idx]) / xn_bar_list[i_idx]
        eta2_update = eta2_update / (N * d)

        # calculate error
        error_mu = abs(np.linalg.norm(mu - mu_update))
        error_W = abs(np.linalg.norm(W - W_update))
        error_sigma_2 = abs(np.linalg.norm(sigma_2 - sigma_2_update))
        error_prr = abs(np.linalg.norm(prr - prr_update))
        error_eta2 = abs(np.linalg.norm(eta2 - eta2_update))

        error_mu_list.append(error_mu), error_W_list.append(error_W), error_sigma2_list.append(error_sigma_2)
        error_prr_list.append(error_prr), error_eta2_list.append(error_eta2)
        if ((error_mu < tol_mu and error_W < tol_W and error_sigma_2 < tol_sigma2 and error_prr < tol_prr
             and error_eta2 < tol_eta2) or iter_num > max_iter_num):
            mu = copy.deepcopy(mu_update)
            W = copy.deepcopy(W_update)
            sigma_2 = copy.deepcopy(sigma_2_update)
            prr = copy.deepcopy(prr_update)
            eta2 = copy.deepcopy(eta2_update)
            # print(iter_num)
            # print(mu, W, sigma_2)
            if iter_num > max_iter_num:
                converge_flag = False
            else:
                converge_flag = True
                # print("Converge: ", iter_num, mu, W, sigma_2)
            break

        else:
            iter_num += 1
            mu = copy.deepcopy(mu_update)
            W = copy.deepcopy(W_update)
            sigma_2 = copy.deepcopy(sigma_2_update)
            prr = copy.deepcopy(prr_update)
            eta2 = copy.deepcopy(eta2_update)

    # do imputation
    est_matrix = copy.deepcopy(CT * 1.0)
    C_i_joint = get_joint_distribution_data_fusion_PPCA_update_prr_eta2(CT, PVT, mu, W, sigma_2, prr, eta2)
    for col_idx in range(N):
        xi = np.reshape(CT[:, col_idx], (-1, 1))
        yi = np.reshape(PVT[:, col_idx], (-1, 1))

        obs_idx_list = [num for num in range(len(xi)) if xi[num] != -1]
        missing_idx_list = [num for num in range(len(xi)) if xi[num] == -1]
        y_idx_list = range(len(xi), 2 * len(xi))
        t_idx_list = range(2 * len(xi), 2 * len(xi) + rank)

        # Calculate CXY, CY-1 matrix first
        CY_minus1_row1 = np.hstack(
            (C_i_joint[obs_idx_list, :][:, obs_idx_list], C_i_joint[obs_idx_list, :][:, y_idx_list]))
        CY_minus1_row2 = np.hstack((C_i_joint[y_idx_list, :][:, obs_idx_list], C_i_joint[y_idx_list, :][:, y_idx_list]))
        CY_minus1 = np.linalg.inv(np.vstack((CY_minus1_row1, CY_minus1_row2)))

        CXY_row1 = np.hstack((C_i_joint[t_idx_list, :][:, obs_idx_list], C_i_joint[t_idx_list, :][:, y_idx_list]))
        CXY_row2 = np.hstack(
            (C_i_joint[missing_idx_list, :][:, obs_idx_list], C_i_joint[missing_idx_list, :][:, y_idx_list]))
        CXY = np.vstack((CXY_row1, CXY_row2))

        # Calculate conditional mu
        xoy = np.vstack((np.reshape(xi[obs_idx_list, :], (-1, 1)), yi))
        muxoy = np.vstack((np.reshape(mu[obs_idx_list, :], (-1, 1)), mu_y))
        mu_xm_xoy = np.reshape(mu[missing_idx_list, :], (-1, 1)) + ((CXY @ CY_minus1) @ (xoy - muxoy))[rank:, :]
        est = mu_xm_xoy

        if est.shape[0] > 0 and est.shape[1] > 0:
            est_matrix[missing_idx_list, col_idx] = est.ravel()

    return est_matrix, converge_flag


def mu_initialize(CT):
    mu_list = []
    for row_idx in range(CT.shape[0]):
        row = CT[row_idx, :]
        tmp = [num for num in row if num != -1]
        mu_list.append(np.mean(tmp) if tmp else 0)
    mu_array = []
    for col_num in range(CT.shape[1]):
        mu_array.append(mu_list)
    mu = (np.array(mu_array)).T
    return mu


def sigma_2_initialize():
    sigma_2 = np.random.uniform()
    return sigma_2


def prr_initialize(CT, PVT):
    non_missing_idx = abs(CT + 1) > 0.999
    prr_initial = np.sum(PVT[non_missing_idx]) / np.sum(CT[non_missing_idx])
    return prr_initial


def W_initialize(CT, rank, initial_method='random'):
    if initial_method is 'random':
        W = np.random.rand(CT.shape[0], rank)
    return W


def get_xn_bar(CT, PVT):
    """
    Calculate xn bar from historical average. If entire row missing then use PVT to calculate prr first and then
    do scale up and average.

    return: xn_bar_list
    """
    # calculate prr first
    non_missing_matrix_CT = (CT != -1)
    prr = np.sum(PVT[non_missing_matrix_CT]) / np.sum(CT[non_missing_matrix_CT])
    xn_bar_list = []
    for row_idx in range(CT.shape[0]):
        row = CT[row_idx, :]
        non_missing_entries = (row != -1)
        if True in non_missing_entries:
            xni_bar = np.mean(row[non_missing_entries])
        else:
            PVT_list = PVT[row_idx, :]
            xni_bar = np.mean(PVT_list) / prr
        xn_bar_list.append(xni_bar)

    return xn_bar_list


def get_joint_distribution_data_fusion_PPCA_update_prr_eta2(CT, PVT, mu, W, sigma_2, prr, eta2):
    """
    Use to get the joint distribution of [xi, yi, ti] under a specific set of parameters (mu, W, sigma2, prr, eta2)
    :param mu: the mean vector for xi
    :param W:
    :param sigma_2:
    :param prr: penetration rate
    :param eta2: the variance coefficient
    :param rank: the number of dimension for latent variables
    :return: covariance matrix
    """
    # get xn bar first
    xn_bar_list = get_xn_bar(CT, PVT)

    tmp1 = sigma_2 * np.identity((W @ W.T).shape[0]) + W @ W.T
    tmp2 = prr * (sigma_2 * np.identity((W @ W.T).shape[0]) + W @ W.T)
    tmp3 = W
    tmp4 = prr * (sigma_2 * np.identity((W @ W.T).shape[0]) + W @ W.T).T
    tmp5 = eta2 * np.diag(xn_bar_list) + prr ** 2 * (sigma_2 * np.identity((W @ W.T).shape[0]) + W @ W.T)
    tmp6 = prr * W
    tmp7 = W.T
    tmp8 = prr * W.T
    tmp9 = np.identity(W.shape[1])

    row1 = np.hstack((tmp1, tmp2, tmp3))
    row2 = np.hstack((tmp4, tmp5, tmp6))
    row3 = np.hstack((tmp7, tmp8, tmp9))

    C_i_joint = np.vstack((row1, row2, row3))

    return C_i_joint


def get_5_expectation_data_fusion_PPCA(xi, yi, rank, W, mu, mu_y, C_i_joint):
    """
    Calculate the five expectation used for updating parameters
    """
    obs_idx_list = [num for num in range(len(xi)) if xi[num] != -1]
    missing_idx_list = [num for num in range(len(xi)) if xi[num] == -1]
    y_idx_list = range(len(xi), 2 * len(xi))
    t_idx_list = range(2 * len(xi), 2 * len(xi) + rank)

    # Calculate CXY, CY-1 matrix first
    CY_minus1_row1 = np.hstack((C_i_joint[obs_idx_list, :][:, obs_idx_list], C_i_joint[obs_idx_list, :][:, y_idx_list]))
    CY_minus1_row2 = np.hstack((C_i_joint[y_idx_list, :][:, obs_idx_list], C_i_joint[y_idx_list, :][:, y_idx_list]))
    CY_minus1 = np.linalg.inv(np.vstack((CY_minus1_row1, CY_minus1_row2)))

    CXY_row1 = np.hstack((C_i_joint[t_idx_list, :][:, obs_idx_list], C_i_joint[t_idx_list, :][:, y_idx_list]))
    CXY_row2 = np.hstack(
        (C_i_joint[missing_idx_list, :][:, obs_idx_list], C_i_joint[missing_idx_list, :][:, y_idx_list]))
    CXY = np.vstack((CXY_row1, CXY_row2))

    # Calculate conditional mu
    xoy = np.vstack((np.reshape(xi[obs_idx_list, :], (-1, 1)), yi))
    muxoy = np.vstack((np.reshape(mu[obs_idx_list, :], (-1, 1)), mu_y))
    mu_t_xoy = ((CXY @ CY_minus1)[:rank, :]) @ (xoy - muxoy)
    mu_xm_xoy = np.reshape(mu[missing_idx_list, :], (-1, 1)) + ((CXY @ CY_minus1)[rank:, :]) @ (xoy - muxoy)

    # Calculate conditional covariance
    Sigma_t_xoy = C_i_joint[t_idx_list, :][:, t_idx_list] - CXY_row1 @ CY_minus1 @ CXY_row1.T
    Sigma_xm_xoy = C_i_joint[missing_idx_list, :][:, missing_idx_list] - CXY_row2 @ CY_minus1 @ CXY_row2.T
    Sigma_txm_xoy = C_i_joint[t_idx_list, :][:, missing_idx_list] - CXY_row1 @ CY_minus1 @ CXY_row2.T

    # Calculate expectation
    E_x = np.vstack((mu_xm_xoy, np.reshape(xi[obs_idx_list, :], (-1, 1))))
    E_t = mu_t_xoy
    E_t_t = Sigma_t_xoy + mu_t_xoy @ mu_t_xoy.T

    tmp_row1 = np.hstack(
        (Sigma_xm_xoy + mu_xm_xoy @ mu_xm_xoy.T, mu_xm_xoy @ (np.reshape(xi[obs_idx_list], (-1, 1))).T))
    tmp_row2 = np.hstack((np.reshape(xi[obs_idx_list], (-1, 1)) @ mu_xm_xoy.T,
                          np.reshape(xi[obs_idx_list], (-1, 1)) @ (np.reshape(xi[obs_idx_list], (-1, 1))).T))
    E_x_x = np.vstack((tmp_row1, tmp_row2))
    E_x_t = np.vstack((Sigma_txm_xoy.T + mu_xm_xoy @ mu_t_xoy.T, np.reshape(xi[obs_idx_list], (-1, 1)) @ mu_t_xoy.T))

    # Change order for observed/missing data
    idx_order_list = missing_idx_list + obs_idx_list
    idx_reorder_list = [idx_order_list.index(i) for i in range(len(xi))]

    E_x = E_x[idx_reorder_list, :]
    E_x_x = E_x_x[idx_reorder_list, :][:, idx_reorder_list]
    E_x_t = E_x_t[idx_reorder_list, :]
    return E_x, E_t, E_t_t, E_x_x, E_x_t


def impute(original_matrix, CT, PVT, rank, ub_list, lmd=1e-5, tol_mu=1e-6, 
    tol_sigma2=1e-6, tol_prr = 1e-6, tol_W=1e-6, tol_eta2 = 1e-6, max_iter_num=100, tol_SVD_iterative=1e-6,
    consecutive_iter_error=1e-8):

    TT_til, converge_flag = data_fusion_PPCA_update_prr_eta2(CT, PVT, rank, tol_mu, tol_W, tol_sigma2, tol_prr, \
                                                                tol_eta2, max_iter_num)
    return TT_til, converge_flag


if __name__ == "__main__":
    # Step 0: Hyperparameter
    rank = 2

    # Step 1: ground-truth data
    num_TOD = 24
    num_day = 20
    tod = 0
    with open('TT.pickle', 'rb') as file_in:
        TT, weekday, dates = pickle.load(file_in)
    TT = TT[:, -num_day * num_TOD:]
    original_matrix = get_TOD_data(TT, tod, num_TOD)
    ub_list = [1.5 * np.max(original_matrix[:, i]) for i in range(original_matrix.shape[1])]

    # Step 2: generate input
    missing_pattern = random_missing  # entire_row_missing / random_missing
    prr = 10
    cp = 10
    CT, cp = get_corrupt_data(original_matrix, cp, missing_pattern=missing_pattern)
    PVT = get_probe_vehicle_data(original_matrix, prr)

    # Step 3: reconstruction
    TT_til, converge_flag = impute(original_matrix, CT, PVT, rank, ub_list)
    num_missing_entries, MAPE, RMSE = performance_measure(original_matrix, TT_til, CT)
    
    # print('Ground truth: \n', TT)
    # print('Input 1, loop detector data: \n', CT)
    # print('Input 2, probe vehicle data: \n', PVT)
    # print('Reconstructed results: \n', TT_til)
    print('MAPE: ', MAPE, ' RMSE: ', RMSE)