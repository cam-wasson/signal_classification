from __future__ import annotations

import datetime
# from dataclasses import dataclass
from matplotlib import use as muse
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time

import optimization_util as opt_util

import sys
sys.path.append('../kalman_filter_bank')
sys.path.append('..')

import util
from filter_bank import SinusoidalFilterBank, run_filter_bank
from Labelling.ExtremaCluster import compute_cluster_dict

muse('Qt5Agg')

OBJECTIVES = {
    'pos_mse': {
        'fn': opt_util.position_error,
        'required_keys': ['context'],
    },
    'vel_mse': {
        'fn': opt_util.velocity_error,
        'required_keys': ['context'],
    },
    'spread_max': {
        'fn': opt_util.spread_max,
        'required_keys': ['context'],
    },
    'anova_loss': {
        'fn': opt_util.anova_loss,
        'required_keys': ['context'],
    }
    # add more objectives here
}

GRADIENTS = {
    'pos_mse': {
        'fn': opt_util.pos_mse_grad,
        'required_keys': ['context']
    },
    'vel_mse': {
        'fn': opt_util.vel_mse_grad,
        'required_keys': ['context']
    }
}


def pos_mse_precontext(selections, max_freq_fft=2.0, spectrum_thresh=None) -> dict:

    conn = util.connect('btc', db_root='../data')
    # get training data
    train_price_df = util.fetch_price_space(conn=conn, selection=selections['train'])

    # get test data
    test_price_df = util.fetch_price_space(conn=conn, selection=selections['test'])

    # read in measurement data
    z_train, z_test = ((train_price_df.Open.values - train_price_df.Open.values[0]) / train_price_df.Open.values[0],
                       (test_price_df.Open.values - test_price_df.Open.values[0]) / test_price_df.Open.values[0])
    dt = 1/(24*60)  # len(train_price_df.Date.unique()) / len(train_price_df)

    # pad front/back of signal for stronger FFT estimation of signal edges
    pad_len = util.compute_pad_length(z_train)
    z_pad, pad_bounds = util.pad_signal(z_train, L=pad_len)

    # produce truth training data
    if spectrum_thresh is None:
        fft_dict_train = util.extract_low_pass_components(z_pad, dt, max_freq=max_freq_fft)
    else:
        fft_dict_train = util.extract_low_pass_components_cdf_thresh(z_pad, dt, cdf_thresh=spectrum_thresh)
    fft_dict_train['truth'] = fft_dict_train['truth'][pad_bounds[0]:pad_bounds[1]]  # fft reconstruction w/o pad

    pad_len = util.compute_pad_length(z_test)
    z_pad, pad_bounds = util.pad_signal(z_test, L=pad_len)
    if spectrum_thresh is None:
        fft_dict_test = util.extract_low_pass_components(z_pad, dt, max_freq=max_freq_fft)
    else:
        fft_dict_test = util.extract_low_pass_components_cdf_thresh(z_pad, dt, cdf_thresh=spectrum_thresh)
    fft_dict_test['truth'] = fft_dict_test['truth'][pad_bounds[0]:pad_bounds[1]]  # fft reconstruction w/o pad

    # store everything important
    fft_dict_train['raw'] = z_train
    fft_dict_test['raw'] = z_test
    fft_dict_train['dt'] = dt
    fft_dict_test['dt'] = dt

    return {'train': fft_dict_train, 'test': fft_dict_test}


def pos_mse_context(pre_ctx, filter_values):
    # instantiate the context data class
    pos_mse_ctx = opt_util.PositionErrorContext

    # populate with values
    pos_mse_ctx.filter_state = filter_values
    pos_mse_ctx.truth_position = pre_ctx['truth']

    return pos_mse_ctx


def vel_mse_precontext(selections, max_freq_fft=2.0, spectrum_thresh=None):
    # re-use the same pre-context generation as the position
    return pos_mse_precontext(selections, max_freq_fft, spectrum_thresh=spectrum_thresh)


def vel_mse_context(pre_ctx, filter_values):
    # instantiate the context data class
    vel_mse_ctx = opt_util.VelocityErrorContext()

    # populate with values
    vel_mse_ctx.truth_position = pre_ctx['truth']
    vel_mse_ctx.filter_state = filter_values
    vel_mse_ctx.dt = pre_ctx['dt']

    return vel_mse_ctx


def spread_max_precontext(selections, max_freq_fft=2.0, cluster_cdf_threshold=.9, dt=1/(24*60)):
    conn = util.connect('btc', db_root='../data')

    # get training data
    train_price_df = util.fetch_price_space(conn=conn, selection=selections['train'])

    # get test data
    test_price_df = util.fetch_price_space(conn=conn, selection=selections['test'])

    # read in measurement data
    z_train, z_test = ((train_price_df.Open.values - train_price_df.Open.values[0]) / train_price_df.Open.values[0],
                       (test_price_df.Open.values - test_price_df.Open.values[0]) / test_price_df.Open.values[0])

    # compute clusters on the training/test sets
    cluster_train = compute_cluster_dict(z_train, max_freq_fft, cluster_cdf_threshold, dt)
    cluster_test = compute_cluster_dict(z_test, max_freq_fft, cluster_cdf_threshold, dt)

    # assemble the precontext
    spread_pre_context = dict({'train': {'raw': z_train,
                                         'truth': cluster_train['fft_dict']['truth'],
                                         'cluster_dict': cluster_train,
                                         'dt': dt},
                               'test': {'raw': z_test,
                                        'truth': cluster_test['fft_dict']['truth'],
                                        'cluster_dict': cluster_test,
                                        'dt': dt}})
    return spread_pre_context


def spread_max_context(pre_ctx, filter_values, dt=None):
    # assign dt value
    # if 'dt' in pre_context.keys() and dt is None:
    #     dt = pre_context['dt']

    # instantiate the context data class
    spread_max_ctx = opt_util.SpreadMaxContext(measurement=pre_ctx['raw'],
                                               filter_state=filter_values,
                                               cluster_dictionary=pre_ctx['cluster_dict'])
    return spread_max_ctx


def anova_precontext(selections, omegas, max_freq_fft=2.0, cluster_cdf_threshold=.9, dt=1/(24*60)):
    pre_ctx = spread_max_precontext(selections, max_freq_fft, cluster_cdf_threshold, dt)
    for k in pre_ctx.keys():
        pre_ctx[k]['omegas'] = omegas
    return pre_ctx


def anova_context(pre_ctx, filter_dict):
    # instantiate the context data class
    anova_ctx = opt_util.ANOVAContext()

    # populate with values
    anova_ctx.measurement = pre_ctx['raw']
    anova_ctx.filter_dict = filter_dict
    anova_ctx.dt = pre_ctx['dt']
    anova_ctx.omegas = pre_ctx['omegas']

    # convert the cluster dictionary to a label array
    label_arr = np.zeros_like(pre_ctx['raw'])
    label_arr[np.concatenate(pre_ctx['cluster_dict']['cluster_min']['x_points'])] = -1
    label_arr[np.concatenate(pre_ctx['cluster_dict']['cluster_max']['x_points'])] = 1
    anova_ctx.label_arr = label_arr
    return anova_ctx


def build_objective_precontext(selection,
                               obj_name='pos_mse',
                               max_freq=2.0,
                               spectrum_thresh=None,
                               cluster_cdf=.9,
                               omegas=None):
    # build the appropriate data class for the objective function
    if obj_name == 'pos_mse':
        pre_ctx = pos_mse_precontext(selection, max_freq, spectrum_thresh=spectrum_thresh)
    elif obj_name == 'vel_mse':
        pre_ctx = vel_mse_precontext(selection, max_freq)
    elif obj_name == 'spread_max':
        pre_ctx = spread_max_precontext(selection, max_freq, cluster_cdf)
    elif obj_name == 'phase_alignment':
        pre_ctx = None  # phase_align_precontext(pre_context, filter_values)
    elif obj_name == 'anova_loss':
        pre_ctx = anova_precontext(selection, omegas=omegas, max_freq_fft=max_freq, cluster_cdf_threshold=cluster_cdf)
    else:
        pre_ctx = None

    return pre_ctx


def build_objective_context(pre_context, filter_dict, obj_name):
    # build the appropriate data class for the objective function
    if obj_name == 'pos_mse':
        ctx = pos_mse_context(pre_context, filter_dict['x'])
    elif obj_name == 'vel_mse':
        ctx = vel_mse_context(pre_context, filter_dict['x'])
    elif obj_name == 'spread_max':
        ctx = spread_max_context(pre_context, filter_dict['x'])
    elif obj_name == 'phase_alignment':
        ctx = None  # phase_align_context(pre_context, filter_values)
    elif obj_name == 'anova_loss':
        ctx = anova_context(pre_context, filter_dict)
    else:
        ctx = None

    return ctx


def train_filter_bank_adam_autograd(filter_bank,
                                    data: dict,
                                    objective_name: str,
                                    n_epochs: int = 10,
                                    autograd_epsilon=1e-4,
                                    alpha=1e-2,  # learning rate
                                    beta1=.9,  # bias factor 1
                                    beta2=.999,  # bias factor 2
                                    eps=1e-8,
                                    reset_cov=True):

    # fetch the chosen objective definition
    obj_entry = OBJECTIVES[objective_name]
    loss_fn = obj_entry['fn']
    n_filters = len(filter_bank.filters)

    # Initialize exponent parameters and moments
    q_linear = filter_bank.sigma_xi.copy()
    q_log = np.log10(q_linear)
    r_linear = filter_bank.rho.copy()
    r_log = np.log10(r_linear)
    m_q = np.zeros_like(q_log)
    v_q = np.zeros_like(q_log)
    m_r = np.zeros_like(r_log)
    v_r = np.zeros_like(r_log)
    losses = np.zeros((n_epochs, 2))

    pre_context_train = data['train']
    pre_context_test = data['test']

    tStart = time.time()
    for epoch in range(n_epochs):

        # Compute current losses
        filter_bank.reset_states()
        filter_dict = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)
        obj_ctx = build_objective_context(pre_context_train, filter_dict, objective_name)
        train_loss = loss_fn(obj_ctx)

        filter_bank.reset_states()
        filter_dict_test = run_filter_bank(filter_bank, pre_context_test['raw'], verbose=False)
        obj_ctx_test = build_objective_context(pre_context_test, filter_dict_test, objective_name)
        test_loss = loss_fn(obj_ctx_test)

        losses[epoch] = [train_loss, test_loss]
        print(f"Epoch {epoch}/{n_epochs}, Loss: {train_loss:.6f} | Validation Loss: {test_loss:.6f}")
        print(f'\tQ Scalars: {q_log}')
        print(f'\tR Scalars: {r_log}')

        # Initialize gradient arrays
        grad_q = np.zeros_like(filter_bank.sigma_xi, dtype=float)
        grad_r = np.zeros_like(filter_bank.sigma_xi, dtype=float)

        # Estimate gradient for each filter and each parameter
        print(f'\nTraining filters...')
        for i in range(n_filters):
            print(f'\tAdjusting filter {i+1}/{n_filters}')
            # adjust Q
            orig_value = q_log[i]

            # Perturb positively, run filter bank
            filter_bank.reset_states()
            q_linear[i] = 10**(orig_value + autograd_epsilon)
            filter_bank.filters[i].set_q(q_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for positive perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_plus, objective_name)
            loss_plus = loss_fn(obj_ctx)

            # Perturb negatively
            filter_bank.reset_states()
            q_linear[i] = 10**(orig_value - autograd_epsilon)
            filter_bank.filters[i].set_q(q_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for negative perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_minus, objective_name)
            loss_minus = loss_fn(obj_ctx)

            # Restore original parameter
            q_linear[i] = 10**orig_value
            filter_bank.filters[i].set_q(q_linear[i])

            # Central finite difference for the Q matrices
            grad_q[i] = (loss_plus - loss_minus) / (2.0 * autograd_epsilon)

            # R scalar
            orig_r = r_log[i]

            # Perturb log space positively
            filter_bank.reset_states()
            r_linear[i] = 10**(orig_r + autograd_epsilon)
            filter_bank.filters[i].set_r(r_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for positive perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_plus, objective_name)
            loss_plus = loss_fn(obj_ctx)

            # Perturb log space negatively, convert to linear, compute loss
            filter_bank.reset_states()
            r_linear[i] = 10**(orig_r - autograd_epsilon)
            filter_bank.filters[i].set_r(r_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for negative perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_minus, objective_name)
            loss_minus = loss_fn(obj_ctx)

            # Restore original linear value of rho
            r_linear[i] = 10**orig_r
            filter_bank.filters[i].set_r(r_linear[i])
            grad_r[i] = (loss_plus - loss_minus) / (2.0 * autograd_epsilon)

        # Update biased moments for Q exponents
        m_q = beta1 * m_q + (1 - beta1) * grad_q
        v_q = beta2 * v_q + (1 - beta2) * (grad_q ** 2)
        # Bias-corrected moments
        m_q_hat = m_q / (1 - beta1 ** (epoch+1))
        v_q_hat = v_q / (1 - beta2 ** (epoch+1))
        # Parameter update
        q_log -= alpha * m_q_hat / (np.sqrt(v_q_hat) + eps)

        # Repeat for R exponents
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        m_r_hat = m_r / (1 - beta1 ** (epoch+1))
        v_r_hat = v_r / (1 - beta2 ** (epoch+1))
        r_log -= alpha * m_r_hat / (np.sqrt(v_r_hat) + eps)

        # Update the filter bank matrices; reset state
        for i, fil in enumerate(filter_bank.filters):
            fil.set_q(10**q_log[i])
            fil.set_r(10**r_log[i])
            fil.x = np.zeros_like(fil.x)
            if reset_cov:
                fil.P *= 1e5

    # Final loss reporting
    filter_bank.reset_states()
    filter_dict = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)
    obj_ctx = build_objective_context(pre_context_train, filter_dict, objective_name)
    train_loss = loss_fn(obj_ctx)

    filter_bank.reset_states()
    filter_dict_test = run_filter_bank(filter_bank, pre_context_test['raw'], verbose=False)
    obj_ctx_test = build_objective_context(pre_context_test, filter_dict_test, objective_name)
    test_loss = loss_fn(obj_ctx_test)

    print(f"\n\nFinal Loss after {n_epochs} epochs: {train_loss:.6f} | Validation Loss: {test_loss:.6f}")
    # Display the optimised noise parameters
    print(f'Fitted Q Exponents: {q_log}')
    print(f'Fitted R Exponents: {r_log}')
    print(f'Run Time: {(time.time() - tStart)/60} mins')

    plt.plot(range(1, n_epochs+1), losses)
    plt.legend(['train', 'test'])
    plt.title(objective_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'{data["save_root"]}\\loss_plot.png')
    plt.clf()

    return filter_bank, q_log, r_log


def kf_single_forward_parallel(kf, z):
    """
    Run one SKF over a sequence of measurements using your FilterPy-based object,
    cache everything needed for backprop. This assumes PARALLEL filter estimation,
    NOT cascading filter estimation.

    Parameters
    ----------
    kf : your KalmanFilter child instance
    z : list/array of measurements (each scalar or shape (1,1))

    Returns
    -------
    cache : dict with all per-timestep quantities needed for backprop
    """
    # F, H, Q, R, x0, P0):
    N = len(z)
    dim_x = kf.dim_x

    cache = {
        "x_prior": np.zeros((N, dim_x)),
        "P_prior": np.zeros((N, dim_x, dim_x)),
        "x_post": np.zeros((N, dim_x)),
        "P_post": np.zeros((N, dim_x, dim_x)),
        "z": np.zeros(N),  # scalar innovation
        "S": np.zeros((N, kf.dim_z, kf.dim_z)),
        "K": np.zeros((N, dim_x, kf.dim_z)),
        "F": kf.F.copy(),
        "H": kf.H.copy(),
        "Q": kf.Q.copy(),
        "R": kf.R.copy(),
        "N": N,
    }

    for n in range(N):
        # Prediction, store
        kf.predict()
        cache['x_prior'][n] = kf.x_prior.copy()
        cache['P_prior'][n] = kf.P_prior.copy()

        # Forward update
        kf.update(z[n])

        # Cache posterior and update-related quantities
        cache["x_post"][n] = kf.x_post.copy()
        cache["P_post"][n] = kf.P_post.copy()
        cache["z"][n] = kf.y.copy()  # residual
        cache["S"][n] = kf.S.copy()
        cache["K"][n] = kf.K.copy()

    return cache


def kf_bank_forward(kf_bank, z):
    caches = []

    for kf in kf_bank.filters:
        # Run this SKF independently on the same measurement sequence
        cache_k = kf_single_forward_parallel(kf, z)
        caches.append(cache_k)

    return {
        "caches": caches,
        "N_filters": len(kf_bank.filters),
    }


def kf_single_backward(cache, loss_fn, target_signal):
    """
    Backpropagation for ONE Kalman Filter (parallel architecture).

    Inputs
    ------
    cache : dict
        Returned by kf_single_forward(kf, z).
    smooth_mse_grad : callable(x_post) -> gradient w.r.t x_post

    Returns
    -------
    dQ : ndarray (dim_x, dim_x)
    dR : float or (1,1)

    Notes
    -----
    This function performs the adjoint sweep *only for ONE filter*.
    The filter bank wrapper will call this once per filter.
    """

    F = cache["F"]
    H = cache["H"]
    Q = cache["Q"]
    R = cache["R"]
    T = cache["N"]

    x_prior = cache["x_prior"]  # (T, dim_x, 1)
    P_prior = cache["P_prior"]  # (T, dim_x, dim_x)
    x_post = cache["x_post"]  # (T, dim_x, 1)
    P_post = cache["P_post"]  # (T, dim_x, dim_x)
    z_list = cache["z"]  # (T,)
    S_list = cache["S"]  # (T,1,1)
    K_list = cache["K"]  # (T,dim_x,1)

    dim_x = F.shape[0]

    # Allocate adjoint variables
    Gx_post = [np.zeros((dim_x, 1)) for _ in range(T + 1)]
    Gp_post = [np.zeros((dim_x, dim_x)) for _ in range(T + 1)]
    Gx_prior = [np.zeros((dim_x, 1)) for _ in range(T + 1)]
    Gp_prior = [np.zeros((dim_x, dim_x)) for _ in range(T + 1)]

    Gy = [0.0 for _ in range(T + 1)]  # scalar
    GS = [0.0 for _ in range(T + 1)]  # scalar
    GS_inv = [0.0 for _ in range(T + 1)]  # scalar
    GK = [np.zeros((dim_x, 1)) for _ in range(T + 1)]

    dQ = np.zeros_like(Q)
    dR = 0.0

    # Inject loss gradient at each time step
    # for t in range(T):
    #     Gx_post[t + 1] += smooth_mse_grad(x_post[t])  # +1 offset for adjoint sweep
    Gx_post[1:] = loss_fn(x_post, target_signal)

    # Backward sweep: t = T-1 down to 0
    for t in reversed(range(T)):
        n = t + 1  # adjoint index

        x_p = x_prior[t]
        P_p = P_prior[t]
        x_f = x_post[t]
        P_f = P_post[t]
        z = float(z_list[t])
        S = float(S_list[t])
        K = K_list[t]

        # --- Step 8: P_f = (I - K H) P_p ---
        I = np.eye(dim_x)
        A = I - K @ H  # (dim_x,dim_x)

        A_bar = Gp_post[n] @ P_p.T  # adjoint through P_f -> A
        Gp_prior[n] += A.T @ Gp_post[n]
        GK[n] += -A_bar @ H.T

        # --- Step 7: x_f = x_p + K z ---
        Gx_prior[n] += Gx_post[n]
        GK[n] += Gx_post[n] * z
        Gy[n] += float(K.T @ Gx_post[n])

        # --- Step 6: K = P_p H^T S_inv ---
        S_inv = 1.0 / S
        u = P_p @ H.T  # (dim_x,1)

        Gp_prior[n] += (S_inv * GK[n]) @ H
        GS_inv[n] += float(u.T @ GK[n])

        # --- Step 5: S_inv = 1/S ---
        GS[n] += -GS_inv[n] * (1.0 / (S ** 2))

        # --- Step 4: S = H P_p H^T + R ---
        Gp_prior[n] += GS[n] * (H.T @ H)
        dR += GS[n]

        # --- Step 3: z = measurement - H x_p ---
        Gx_prior[n] += - H.T * Gy[n]

        # --- Step 2: P_p = F P_prev F^T + Q ---
        Gp_prev = F.T @ Gp_prior[n] @ F
        dQ += Gp_prior[n]

        # --- Step 1: x_p = F x_prev ---
        Gx_prev = F.T @ Gx_prior[n]

        # Propagate to next earlier time
        Gx_post[n - 1] += Gx_prev
        Gp_post[n - 1] += Gp_prev

    return dQ, dR


def kf_bank_backward(bank_cache, loss_fn, target_signal):
    """
    Backpropagation for an entire PARALLEL Kalman Filter Bank.

    Inputs
    ------
    bank_cache : dict
        Output of kf_bank_forward(). Contains list bank_cache["caches"].
    smooth_mse_grad : callable

    Returns
    -------
    dQ_list : list of ndarrays
        One gradient dQ per filter.
    dR_list : list of floats
        One gradient dR per filter.

    Notes
    -----
    - Each filter is treated independently.
    - No cross-filter gradients exist under the parallel architecture.
    - This method does not assume any shared parameters.
    - This method is a simple wrapper around kf_single_backward().
    """

    caches = bank_cache["caches"]
    M = bank_cache["N_filters"]

    dQ_list = []
    dR_list = []

    for i in range(M):
        cache_i = caches[i]
        dQ_i, dR_i = kf_single_backward(cache_i, smooth_mse_grad)
        dQ_list.append(dQ_i)
        dR_list.append(dR_i)

    return dQ_list, dR_list


def train_filter_bank_grad(filter_bank,
                           data: dict,
                           objective_name: str,
                           n_epochs: int = 10,
                           alpha=1e-2,  # learning rate
                           beta1=.9,  # bias factor 1
                           beta2=.999,  # bias factor 2
                           eps=1e-8,
                           reset_cov=True):

    # fetch the chosen objective definition
    loss_fn = OBJECTIVES[objective_name]['fn']
    grad_fn = GRADIENTS[objective_name]['fn']
    n_filters = len(filter_bank.filters)

    # Initialize exponent parameters and moments
    q_linear = filter_bank.sigma_xi.copy()
    q_log = np.log10(q_linear)
    r_linear = filter_bank.rho.copy()
    r_log = np.log10(r_linear)
    m_q = np.zeros_like(q_log)
    v_q = np.zeros_like(q_log)
    m_r = np.zeros_like(r_log)
    v_r = np.zeros_like(r_log)
    losses = np.zeros((n_epochs, 2))

    pre_context_train = data['train']
    pre_context_test = data['test']

    # Initialize gradient arrays
    grad_q = np.zeros_like(filter_bank.sigma_xi, dtype=float)
    grad_r = np.zeros_like(filter_bank.sigma_xi, dtype=float)

    # begin training
    tStart = time.time()
    for epoch in range(n_epochs):

        # Compute current losses
        filter_bank.reset_states()
        bank_cache = kf_bank_forward(filter_bank, pre_context_train['raw'])
        grad = kf_bank_backprop()
        # filter_dict = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)
        # obj_ctx = build_objective_context(pre_context_train, filter_dict, objective_name)
        train_loss = 0  # loss_fn(obj_ctx)

        filter_bank.reset_states()
        filter_dict_test = run_filter_bank(filter_bank, pre_context_test['raw'], verbose=False)
        obj_ctx_test = build_objective_context(pre_context_test, filter_dict_test, objective_name)
        test_loss = loss_fn(obj_ctx_test)

        losses[epoch] = [train_loss, test_loss]
        print(f"Epoch {epoch}/{n_epochs}, Loss: {train_loss:.6f} | Validation Loss: {test_loss:.6f}")
        print(f'\tQ Scalars: {q_log}')
        print(f'\tR Scalars: {r_log}')

        print(f'\nTraining filters...')
        for i in range(n_filters):
            print(f'\tAdjusting filter {i+1}/{n_filters}')

            # Perturb positively, run filter bank
            filter_bank.reset_states()
            q_linear[i] = 10**(orig_value + autograd_epsilon)
            filter_bank.filters[i].set_q(q_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for positive perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_plus, objective_name)
            loss_plus = loss_fn(obj_ctx)

            grad_q[i] = grad_fn()
            grad_r[i] = grad_fn()

        # Update biased moments for Q exponents
        m_q = beta1 * m_q + (1 - beta1) * grad_q
        v_q = beta2 * v_q + (1 - beta2) * (grad_q ** 2)
        # Bias-corrected moments
        m_q_hat = m_q / (1 - beta1 ** (epoch+1))
        v_q_hat = v_q / (1 - beta2 ** (epoch+1))
        # Parameter update
        q_log -= alpha * m_q_hat / (np.sqrt(v_q_hat) + eps)

        # Repeat for R exponents
        m_r = beta1 * m_r + (1 - beta1) * grad_r
        v_r = beta2 * v_r + (1 - beta2) * (grad_r ** 2)
        m_r_hat = m_r / (1 - beta1 ** (epoch+1))
        v_r_hat = v_r / (1 - beta2 ** (epoch+1))
        r_log -= alpha * m_r_hat / (np.sqrt(v_r_hat) + eps)

        # Update the filter bank matrices; reset state
        for i, fil in enumerate(filter_bank.filters):
            fil.set_q(10**q_log[i])
            fil.set_r(10**r_log[i])
            fil.x = np.zeros_like(fil.x)
            if reset_cov:
                fil.P *= 1e5

    # Final loss reporting
    filter_bank.reset_states()
    filter_dict = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)
    obj_ctx = build_objective_context(pre_context_train, filter_dict, objective_name)
    train_loss = loss_fn(obj_ctx)

    filter_bank.reset_states()
    filter_dict_test = run_filter_bank(filter_bank, pre_context_test['raw'], verbose=False)
    obj_ctx_test = build_objective_context(pre_context_test, filter_dict_test, objective_name)
    test_loss = loss_fn(obj_ctx_test)

    print(f"\n\nFinal Loss after {n_epochs} epochs: {train_loss:.6f} | Validation Loss: {test_loss:.6f}")
    # Display the optimised noise parameters
    print(f'Fitted Q Exponents: {q_log}')
    print(f'Fitted R Exponents: {r_log}')
    print(f'Run Time: {(time.time() - tStart)/60} mins')


def plot_filter_bank(meas, bank_dict, dt, objective_function, truth_pos=None):
    # extract values
    bank_x = bank_dict['x']
    amp = bank_dict['amp']
    phase = bank_dict['phi']

    # create plot
    t_plot = np.arange(0, len(meas) * dt, dt)
    figure, axs = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle(objective_function)

    # plot spread
    spread = np.tanh(meas - bank_x[:, 0])
    positive_idx, negative_idx = spread > 0, spread < 0
    axs[0, 0].scatter(t_plot[positive_idx], spread[positive_idx], s=5, color='green')
    axs[0, 0].scatter(t_plot[negative_idx], spread[negative_idx], s=5, color='red')
    axs[0, 0].set_title(f'Tanh Spread')

    # axs[0, 0].scatter(t, meas, color='red', s=5)
    # axs[0, 0].plot(t, bank_x[:, 0], color='k')
    # axs[0, 0].set_title(f'Position Estimation')

    # plot velocity
    axs[1, 0].plot(t_plot, bank_x[:, 1], color='k', label='filter estimate')
    axs[1, 0].set_title(f'Velocity Estimation')
    if truth_pos is not None:
        axs[1, 0].plot(t_plot, np.gradient(truth_pos)/dt, color='orange', alpha=.5, label='truth extraction')
    axs[1, 0].legend()

    # plot amplitude
    axs[0, 1].plot(t_plot, amp, label=bank_dict['omega'])
    axs[0, 1].legend()
    axs[0, 1].set_title(f'Amplitude Estimation')

    # plot phase
    axs[1, 1].plot(t_plot, phase, label=bank_dict['omega'])
    axs[1, 1].legend()
    axs[1, 1].set_title(f'Phase Estimation')

    return figure


def stationary_hex_plot(raws, filter_dicts, label_arrs, dt):
    # extract
    raw_train, raw_test = raws[0], raws[1]
    filter_dict_train, filter_dict_test = filter_dicts[0], filter_dicts[1]
    label_arr_train, label_arr_test = label_arrs[0], label_arrs[1]
    t_train, t_test = np.arange(len(filter_dict_train['x']))*dt, np.arange(len(filter_dict_test['x']))*dt
    min_idx_train, max_idx_train = (label_arr_train == -1, label_arr_train == 1)
    min_idx_test, max_idx_test = (label_arr_test == -1, label_arr_test == 1)

    fig, ax = plt.subplots(2, 3, figsize=(10, 8))

    # plot positions
    ax[0, 0].plot(t_train, raw_train)
    ax[0, 0].plot(t_train, filter_dict_train['x'][:, 0], color='orange', alpha=.5)
    ax[0, 0].scatter(t_train[min_idx_train], raw_train[min_idx_train], color='green')
    ax[0, 0].scatter(t_train[max_idx_train], raw_train[max_idx_train], color='red')
    ax[0, 0].set_title('Raw Measurement [Train]')

    ax[1, 0].plot(t_test, raw_test)
    ax[1, 0].plot(t_test, filter_dict_test['x'][:, 0], color='orange', alpha=.5)
    ax[1, 0].scatter(t_test[min_idx_test], raw_test[min_idx_test], color='green')
    ax[1, 0].scatter(t_test[max_idx_test], raw_test[max_idx_test], color='red')
    ax[1, 0].set_title('Raw Measurement [Test]')

    # plot empirical velocity
    emp_vel_train = np.diff(np.concatenate(([0], filter_dict_train['x'][:, 0]))) / dt
    emp_vel_test = np.diff(np.concatenate(([0], filter_dict_test['x'][:, 0]))) / dt
    ax[0, 1].plot(t_train, emp_vel_train)
    ax[0, 1].plot(t_train, filter_dict_train['x'][:, 1], color='orange', alpha=.5)
    ax[0, 1].scatter(t_train[min_idx_train], emp_vel_train[min_idx_train], color='green')
    ax[0, 1].scatter(t_train[max_idx_train], emp_vel_train[max_idx_train], color='red')
    ax[0, 1].set_title('Empirical Velocity [Train]')

    ax[1, 1].plot(t_test, emp_vel_test)
    ax[1, 1].plot(t_test, filter_dict_test['x'][:, 1], color='orange', alpha=.5)
    ax[1, 1].scatter(t_test[min_idx_test], emp_vel_test[min_idx_test], color='green')
    ax[1, 1].scatter(t_test[max_idx_test], emp_vel_test[max_idx_test], color='red')
    ax[1, 1].set_title('Empirical Velocity [Test]')

    # plot empirical acceleration
    emp_acc_train = np.diff(np.concatenate(([0], filter_dict_train['x'][:, 1]))) / dt
    emp_acc_test = np.diff(np.concatenate(([0], filter_dict_test['x'][:, 1]))) / dt
    ax[0, 2].plot(t_train, emp_acc_train)
    ax[0, 2].scatter(t_train[min_idx_train], emp_acc_train[min_idx_train], color='green')
    ax[0, 2].scatter(t_train[max_idx_train], emp_acc_train[max_idx_train], color='red')
    ax[0, 2].set_title('Empirical Acceleration [Train]')

    ax[1, 2].plot(t_test, emp_acc_test)
    ax[1, 2].scatter(t_test[min_idx_test], emp_acc_test[min_idx_test], color='green')
    ax[1, 2].scatter(t_test[max_idx_test], emp_acc_test[max_idx_test], color='red')
    ax[1, 2].set_title('Empirical Acceleration [Test]')


class RunConfig:
    # input items
    train_times: list
    test_times: list
    objective: str
    # initialized in object
    fft_max_freq_dict: dict
    label_cluster_cdf_thresh_dict: dict
    omega_dict: dict
    sigma_xi0_dict: dict
    rho0_dict: dict
    alpha_dict: dict
    epoch_dict: dict
    # extracted in object
    data_selections: dict
    omega_arr: np.array
    # assembled in object
    filter_bank_config: dict
    truth_extraction_config: dict
    grad_desc_config: dict

    def __init__(self, train_times, test_times, objective_str):
        # assign input values
        self.train_times = train_times
        self.test_times = test_times
        self.objective = objective_str

        # create the structures to choose hyperparams based on objective function
        self.init_filter_bank_parameters()
        self.init_truth_extraction_parameters()
        self.init_grad_desc_parameters()

        # build the train/test selections
        self.build_selections()

        # build the truth extraction config
        self.build_truth_extraction_config()

        # build the filter bank config
        self.build_filter_bank_config()

        # build the gradient descent config
        self.build_grad_desc_config()

    def init_filter_bank_parameters(self):
        self.omega_dict = {'pos_mse': np.array([0.02, 0.66, 2.04, 3.9]),
                           'vel_mse': np.array([0.02, 0.66, 2.04, 3.9]),
                           'spread_max': np.array([0.02, 0.1, .25, 0.66]),
                           'anova_loss': np.array([0.02, 0.66, 2.04, 3.9])}
        n_omega = len(self.omega_dict[self.objective])

        # R matrix scalar values
        self.rho0_dict = {'pos_mse': [10**(-0.5)]*n_omega,  # replace rho0/sigmaXi0 w/ grid search results
                          'vel_mse': [10**(-0.5)]*n_omega,
                          'spread_max': 10**np.array([0.46352044,  0.5294198,   0.67744341, -0.19491549]),
                          'anova_loss': [10**(-0.0)]*n_omega}

        # Q matrix scalar values
        self.sigma_xi0_dict = {'pos_mse': [10**0.5]*n_omega,
                               'vel_mse': [10**0.5]*n_omega,
                               'spread_max': 10**np.array([-1.10086254, -0.99297798, -0.71250029,  0.87094734]),
                               'anova_loss': [10**0.0]*n_omega}

    def init_truth_extraction_parameters(self):
        self.fft_max_freq_dict = {'pos_mse': 5,
                                  'vel_mse': 2,
                                  'spread_max': 5,
                                  'anova_loss': 5}
        self.label_cluster_cdf_thresh_dict = {'pos_mse': .95,
                                              'vel_mse': .95,
                                              'spread_max': .95,
                                              'anova_loss': .95}

    def init_grad_desc_parameters(self):
        self.alpha_dict = {'pos_mse': 2.5e-2,
                           'vel_mse': 5e-2,
                           'spread_max': 2.5e-2,
                           'anova_loss': 2.5e-2}
        self.epoch_dict = {'pos_mse': 14,
                           'vel_mse': 12,
                           'spread_max': 4,
                           'anova_loss': 12}

    def build_selections(self):
        # build training selection
        train_selector = util.selector()
        train_selector.start_time = self.train_times[0]
        train_selector.stop_time = self.train_times[1]

        # build test selection
        test_selector = util.selector()
        test_selector.start_time = self.test_times[0]
        test_selector.stop_time = self.test_times[1]
        self.data_selections = {'train': train_selector,
                                'test': test_selector}

    def build_filter_bank_config(self):
        self.omega_arr = self.omega_dict[self.objective]
        self.filter_bank_config = {'omegas': self.omega_arr,
                                   'sigma_xi': self.sigma_xi0_dict[self.objective],
                                   'rho': self.rho0_dict[self.objective]}

    def build_truth_extraction_config(self):
        self.truth_extraction_config = {'max_freq': self.fft_max_freq_dict[self.objective],
                                        'cluster_cdf': self.label_cluster_cdf_thresh_dict[self.objective]}

    def build_grad_desc_config(self):
        self.grad_desc_config = {'alpha': self.alpha_dict[self.objective],
                                 'epoch': self.epoch_dict[self.objective]}

    def to_str(self):
        # init the return string
        return_str = ''

        # add in data selection
        return_str += 'Measurement Selection\n'
        return_str += f'\t Train: {self.train_times}\n'
        return_str += f'\t Test: {self.test_times}\n'

        # add in filter bank parameters
        print_me = dict(self.filter_bank_config)
        print_me['rho'] = np.log10(print_me['rho'])
        print_me['sigma_xi'] = np.log10(print_me['sigma_xi'])
        return_str += '\nFilter Bank Parameters:\n'
        return_str += f'{pprint.pformat(print_me, indent=4, width=80)}\n'

        # add in truth extraction parameters
        return_str += f'\nTruth Extraction Parameters:\n'
        return_str += f'{pprint.pformat(self.truth_extraction_config, indent=4, width=80)}\n'

        # add in gradient descent parameters
        return_str += '\nGradient Descent Parameters:\n'
        return_str += f'{pprint.pformat(self.grad_desc_config, indent=4, width=80)}\n'

        return return_str


if __name__ == "__main__":
    import os
    import pickle

    # set initial parameters
    train_selection = [datetime.datetime(2025, 7, 31).timestamp(), datetime.datetime(2025, 8, 10).timestamp() - 1]
    test_selection = [datetime.datetime(2025, 8, 11).timestamp(), datetime.datetime(2025, 8, 12).timestamp() - 1]
    objective = 'pos_mse'
    subdir_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # save_root = f'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2025\\4_Fall\\CS-591\\training_sessions\\{objective}'
    save_root = f'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2026\\Capstone 2\\training_sessions\\{objective}'
    save_root = f'{save_root}\\{subdir_str}'
    os.makedirs(save_root)

    # construct the run config
    run_cfg = RunConfig(train_times=train_selection,
                        test_times=test_selection,
                        objective_str=objective)
    # run_cfg.grad_desc_config['epoch'] = 3  # overwrite epochs for testing functionality

    # construct full precontext
    pre_context = build_objective_precontext(selection=run_cfg.data_selections,
                                             omegas=run_cfg.filter_bank_config['omegas'],
                                             obj_name=run_cfg.objective,
                                             max_freq=run_cfg.truth_extraction_config['max_freq'],
                                             cluster_cdf=run_cfg.truth_extraction_config['cluster_cdf'])
    pre_context['save_root'] = save_root
    with open(f'{save_root}\\{objective}_shot_notes.txt', 'w') as f:
        f.write('Run Config Notes:\n')
        f.write(run_cfg.to_str())
        f.close()

    # Construct the filter bank w/ the following sinusoidal frequencies
    p0 = 1e-2
    # fbank = SinusoidalFilterBank(
    #     omegas=run_cfg.filter_bank_config['omegas'],
    #     dt=pre_context['train']['dt'],
    #     sigma_xi=run_cfg.filter_bank_config['sigma_xi'],
    #     rho=run_cfg.filter_bank_config['rho'],
    #     p0=p0
    # )
    per_omega_extract = util.extract_component_reconstructions(pre_context['train']['raw'],
                                                               pre_context['train']['dt'],
                                                               max_freq=2)
    omegas = np.array(list(per_omega_extract.keys()))
    fbank = SinusoidalFilterBank(omegas=omegas,
                                 dt=pre_context['train']['dt'],
                                 sigma_xi=[10**0.0]*len(omegas),
                                 rho=[10**0.0]*len(omegas),
                                 p0=p0*len(omegas))

    # Train the filter bank
    # trained_filter_bank, q_log_final, r_log_final = train_filter_bank_adam(filter_bank=fbank,
    #                                                                        data=pre_context,
    #                                                                        objective_name=objective,
    #                                                                        n_epochs=run_cfg.grad_desc_config['epoch'],
    #                                                                        alpha=run_cfg.grad_desc_config['alpha'],
    #                                                                        reset_cov=False)
    train_filter_bank_grad(filter_bank=fbank,
                           data=pre_context,
                           objective_name=objective,
                           n_epochs=run_cfg.grad_desc_config['epoch'],
                           alpha=run_cfg.grad_desc_config['alpha'],
                           reset_cov=False)
    with open(f'{save_root}\\{objective}_shot_notes.txt', 'a') as f:
        f.write(f'\n\nFinal Q (log10): {q_log_final}\n')
        f.write(f'Final R (log10): {r_log_final}')
        f.close()
    pickle.dump(trained_filter_bank, open(f'{save_root}\\filter_bank.pkl', 'wb'))
    pickle.dump(run_cfg, open(f'{save_root}\\run_config.pkl', 'wb'))

    # run and plot the trained filter
    dataset = ['train', 'test']
    for d in dataset:
        # reset the filters
        trained_filter_bank.reset_states()

        # run
        trained_filter_dict = run_filter_bank(trained_filter_bank, pre_context[d]['raw'])
        fig = plot_filter_bank(pre_context[d]['raw'],
                               trained_filter_dict,
                               pre_context[d]['dt'],
                               objective_function=objective,
                               truth_pos=pre_context[d]['truth'])
        # save
        fig.savefig(f'{save_root}\\filter_quad_plot_{d}.png')

        # plot
        plt.figure()
        t = np.arange(0, len(pre_context[d]['raw']))*pre_context[d]['dt']
        plt.plot(t, pre_context[d]['raw'], label='meas')
        plt.plot(t, trained_filter_dict['x'][:, 0], label='filter')
        plt.xlabel('Days')
        plt.legend()
        plt.savefig(f'{save_root}\\meas_filter_plot_{d}.png')

    # plt.show(block=True)


