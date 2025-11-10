"""
option1_closed_form_script.py
================================

This script illustrates how one might begin to optimise the process
and measurement noise covariances (``Q`` and ``R``) of a bank of
sinusoidal Kalman filters using gradient‑based methods.  The goal is
to demonstrate an end‑to‑end optimisation loop for ``Q`` and ``R``
given a differentiable objective function.  In practice, closed‑form
backpropagation formulas exist for the Kalman filter, but deriving and
implementing them from scratch can be challenging.  Here we provide a
minimal example that uses finite‑difference gradients as a proxy for
the analytic derivatives.  This example is meant as a stepping stone
towards implementing the fully analytical approach described in the
report.

Note: For real applications or larger systems, a more sophisticated
approach should be used.  Closed‑form derivative formulas provide
greater efficiency and numerical stability as discussed in the
accompanying report and references【355793402231865†L93-L104】.  This
script offers a template that can later be replaced with analytic
gradients when available.
"""

from __future__ import annotations

import datetime

from matplotlib import use as muse
import matplotlib.pyplot as plt

import numpy as np
import optimization_util as opt_util
import time

import sys
sys.path.append('../kalman_filter_bank')
<<<<<<< HEAD
from kalman_filter_bank import SinusoidalFilterBank, run_filter_bank
=======
from filter_bank import SinusoidalFilterBank, SinusoidalCMMEAFilterBank, run_filter_bank
>>>>>>> d693e80f44bf3d8a56886c061ece94e47b527bf3
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
    }
    # add more objectives here
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


def pos_mse_context(pre_context, filter_values):
    # instantiate the context data class
    pos_mse_ctx = opt_util.PositionErrorContext

    # populate with values
    pos_mse_ctx.filter_state = filter_values
    pos_mse_ctx.truth_position = pre_context['truth']

    return pos_mse_ctx


def vel_mse_precontext(selections, max_freq_fft=2.0, spectrum_thresh=None):
    # re-use the same pre-context generation as the position
    return pos_mse_precontext(selections, max_freq_fft, spectrum_thresh=spectrum_thresh)


def vel_mse_context(pre_context, filter_values):
    # instantiate the context data class
    vel_mse_ctx = opt_util.VelocityErrorContext()

    # populate with values
    vel_mse_ctx.truth_position = pre_context['truth']
    vel_mse_ctx.filter_state = filter_values
    vel_mse_ctx.dt = pre_context['dt']

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
    spread_pre_context = dict({'train': {'raw': z_train, 'cluster_dict': cluster_train, 'dt': dt},
                               'test': {'raw': z_test, 'cluster_dict': cluster_test, 'dt': dt}})
    return spread_pre_context


def spread_max_context(pre_context, filter_values, dt=None):
    # assign dt value
    if 'dt' in pre_context.keys() and dt is None:
        dt = pre_context['dt']

    # instantiate the context data class
    spread_max_ctx = opt_util.SpreadMaxContext(measurement=pre_context['raw'],
                                               filter_state=filter_values,
                                               cluster_dictionary=pre_context['cluster_dict'],
                                               dt=dt)
    return spread_max_ctx


def phase_align_context(pre_context, filter_values):
    # instantiate the context data class
    phase_align_ctx = opt_util.PhaseAlignmentContext

    # populate with values
    return phase_align_ctx


def build_objective_precontext(selection, obj_name='pos_mse', max_freq=2.0, spectrum_thresh=None, cluster_cdf=.9):
    # build the appropriate data class for the objective function
    if obj_name == 'pos_mse':
        pre_ctx = pos_mse_precontext(selection, max_freq, spectrum_thresh=spectrum_thresh)
    elif obj_name == 'vel_mse':
        pre_ctx = vel_mse_precontext(selection, max_freq)
    elif obj_name == 'spread_max':
        pre_ctx = spread_max_precontext(selection, max_freq, cluster_cdf)
    elif obj_name == 'phase_alignment':
        # pre_ctx = phase_align_precontext(pre_context, filter_values)
        pre_ctx = None
    else:
        pre_ctx = None

    return pre_ctx


def build_objective_context(pre_context, filter_values, obj_name):
    # build the appropriate data class for the objective function
    if obj_name == 'pos_mse':
        ctx = pos_mse_context(pre_context, filter_values)
    elif obj_name == 'vel_mse':
        ctx = vel_mse_context(pre_context, filter_values)
    elif obj_name == 'spread_max':
        ctx = spread_max_context(pre_context, filter_values)
    elif obj_name == 'phase_alignment':
        ctx = phase_align_context(pre_context, filter_values)
    else:
        ctx = None

    return ctx


def train_filter_bank_adam(filter_bank,
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
        obj_ctx = build_objective_context(pre_context_train, filter_dict['x'], objective_name)
        train_loss = loss_fn(obj_ctx)

        filter_bank.reset_states()
        filter_dict_test = run_filter_bank(filter_bank, pre_context_test['raw'], verbose=False)
        obj_ctx_test = build_objective_context(pre_context_test, filter_dict_test['x'], objective_name)
        test_loss = loss_fn(obj_ctx_test)

        losses[epoch] = [train_loss, test_loss]
        print(f"Epoch {epoch}/{n_epochs}, Loss: {train_loss:.6f} | Validation Loss: {test_loss:.6f}")
        print(f'\tQ Scalars: {q_log}')
        print(f'\tR Scalars: {r_log}')

        # Initialize gradient arrays
        grad_q = np.zeros_like(filter_bank.sigma_xi)
        grad_r = np.zeros_like(filter_bank.sigma_xi)

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
            obj_ctx = build_objective_context(pre_context_train, filter_dict_plus['x'], objective_name)
            loss_plus = loss_fn(obj_ctx)

            # Perturb negatively
            filter_bank.reset_states()
            q_linear[i] = 10**(orig_value - autograd_epsilon)
            filter_bank.filters[i].set_q(q_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for negative perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_minus['x'], objective_name)
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
            obj_ctx = build_objective_context(pre_context_train, filter_dict_plus['x'], objective_name)
            loss_plus = loss_fn(obj_ctx)

            # Perturb log space negatively, convert to linear, compute loss
            filter_bank.reset_states()
            r_linear[i] = 10**(orig_r - autograd_epsilon)
            filter_bank.filters[i].set_r(r_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)

            # compute loss for negative perturbation
            obj_ctx = build_objective_context(pre_context_train, filter_dict_minus['x'], objective_name)
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
        for i, f in enumerate(filter_bank.filters):
            f.set_q(10**q_log[i])
            f.set_r(10**r_log[i])
            f.x = np.zeros_like(f.x)
            if reset_cov:
                f.P *= 1e5

    # Final loss reporting
    filter_bank.reset_states()
    filter_dict = run_filter_bank(filter_bank, pre_context_train['raw'], verbose=False)
    obj_ctx = build_objective_context(pre_context_train, filter_dict['x'], objective_name)
    train_loss = loss_fn(obj_ctx)

    filter_bank.reset_states()
    filter_dict_test = run_filter_bank(filter_bank, pre_context_test['raw'], verbose=False)
    obj_ctx_test = build_objective_context(pre_context_test, filter_dict_test['x'], objective_name)
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

    return filter_bank


def plot_filter_bank(meas, bank_dict, dt, objective_function, context=None):
    # extract values
    bank_x = bank_dict['x']
    amp = bank_dict['amp']
    phase = bank_dict['phi']

    # plot filter estimates and system dynamics
    t = np.arange(0, len(meas) * dt, dt)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    if objective_function == 'spread_max' or objective_function == 'pos_mse':
        spread = np.tanh(meas - bank_x[:, 0])
        positive_idx, negative_idx = spread > 0, spread < 0
        axs[0, 0].scatter(t[positive_idx], spread[positive_idx], s=5, color='green')
        axs[0, 0].scatter(t[negative_idx], spread[negative_idx], s=5, color='red')
        axs[0, 0].set_title(f'Spread')
    else:
        axs[0, 0].scatter(t, meas, color='red', s=5)
        axs[0, 0].plot(t, bank_x[:, 0], color='k')
        axs[0, 0].set_title(f'Position Estimation')

    axs[1, 0].plot(t, bank_x[:, 1], color='k')
    axs[1, 0].set_title(f'Velocity Estimation')

    axs[0, 1].plot(t, amp, label=bank_dict['omega'])
    axs[0, 1].legend()
    axs[0, 1].set_title(f'Amplitude Estimation')

    axs[1, 1].plot(t, phase, label=bank_dict['omega'])
    axs[1, 1].legend()
    axs[1, 1].set_title(f'Phase Estimation')

    return fig


if __name__ == "__main__":
    import pandas as pd
    sys.path.append('..')
    import util

    # build training selection
    train_selection = util.selector()
    train_selection.start_time = datetime.datetime(2025, 7, 31).timestamp()
    train_selection.stop_time = datetime.datetime(2025, 8, 10).timestamp() - 1

    # build test selection
    test_selection = util.selector()
    test_selection.start_time = datetime.datetime(2025, 8, 11).timestamp()
    test_selection.stop_time = datetime.datetime(2025, 8, 12).timestamp() - 1
    data_selection = {'train': train_selection,
                      'test': test_selection}

    # construct full precontext
    objective = 'pos_mse'
    pre_context = build_objective_precontext(selection=data_selection,
                                             obj_name=objective,
                                             max_freq=5,
                                             cluster_cdf=.95)

    # Construct the filter bank w/ the following sinusoidal frequencies
    # omegas = np.array([0.02, 0.34, 0.66, 2.04, 3.9])
    omegas = np.array([0.02, 0.66, 2.04, 3.9])
    logs = {''}
    fbank = SinusoidalFilterBank(
        omegas=omegas,
        dt=pre_context['train']['dt'],
        sigma_xi=[10**0.5]*len(omegas),
        rho=[10**-0.5]*len(omegas),
    )
    # fbank = SinusoidalCMMEAFilterBank(
    #     omegas=omegas,
    #     dt=pre_context['train']['dt'],
    #     sigma_xi=[10**0.0]*len(omegas),
    #     rho=[10**-0.0]*len(omegas),
    # )

    # warm up the filter bank's P matrix
    # filter_out = run_filter_bank(fbank, fft_dict['raw'][:200], verbose=False)
    # fbank.P = filter_out['p'][-1]

    # Train the filter bank
    alphas = {'pos_mse': 2.5e-2,  # [q=.5, r=-.5]
              'vel_mse': 5e-2,  # [q=.5, r=-.5]
              'spread_max': 2.5e-2,  # [q=0, r=0]
              'phase_alignment': 2.5e-2}
    trained_filter_bank = train_filter_bank_adam(filter_bank=fbank,
                                                 data=pre_context,
                                                 objective_name=objective,
                                                 n_epochs=14,
                                                 alpha=alphas[objective],
                                                 reset_cov=False)

    # run and plot the trained filter
    dataset = ['train', 'test']
    for d in dataset:
        # reset the filters
        trained_filter_bank.reset_states()
        # run and plot
        trained_filter_dict = run_filter_bank(trained_filter_bank, pre_context[d]['raw'])
        fig = plot_filter_bank(pre_context[d]['raw'],
                               trained_filter_dict,
                               pre_context[d]['dt'],
                               objective_function=objective)
        plt.figure()
        t = np.arange(0, len(pre_context[d]['raw']))*pre_context[d]['dt']
        plt.plot(t, pre_context[d]['raw'], label='meas')
        plt.plot(t, trained_filter_dict['x'][:, 0], label='filter')
        plt.xlabel('Days')
    plt.legend()
    plt.show(block=True)


