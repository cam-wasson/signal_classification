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
from tqdm import tqdm

import sys
sys.path.append('../kalman_filter_bank')
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
    }
    # add more objectives here
}


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


def pos_mse_context(pre_context, filter_values):
    # instantiate the context data class
    pos_mse_ctx = opt_util.PositionErrorContext
    # populate with values
    pos_mse_ctx.filter_state = filter_values
    pos_mse_ctx.truth_position = pre_context['truth']
    return pos_mse_ctx


def vel_mse_context(pre_context, filter_values):
    # instantiate the context data class
    vel_mse_ctx = opt_util.VelocityErrorContext
    # populate with values
    vel_mse_ctx.truth_position = pre_context['truth']
    vel_mse_ctx.filter_state = filter_values
    vel_mse_ctx.dt = pre_context['dt']
    return vel_mse_ctx


def spread_max_context(pre_context, filter_values):
    # instantiate the context data class
    spread_max_ctx = opt_util.SpreadMaxContext
    # populate with values
    spread_max_ctx.measurement = pre_context['raw']
    spread_max_ctx.filter_state = filter_values
    spread_max_ctx.cluster_dictionary = pre_context['cluster_dict']
    return spread_max_ctx


def phase_align_context(pre_context, filter_values):
    # instantiate the context data class
    phase_align_ctx = opt_util.PhaseAlignmentContext
    # populate with values
    return phase_align_ctx


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
    losses = np.zeros(n_epochs)

    for epoch in range(n_epochs):

        # Compute current loss
        filter_dict = run_filter_bank(filter_bank, data['raw'], verbose=False)
        obj_ctx = build_objective_context(data, filter_dict['x'], objective_name)
        loss = loss_fn(obj_ctx)
        losses[epoch] = loss

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}")
        print(f'\tQ Scalars: {q_log}')
        print(f'\tR Scalars: {r_log}')
        # plot_filter_bank(data['raw'], filter_dict, data['dt'])

        # Initialize gradient arrays
        grad_q = np.zeros_like(filter_bank.sigma_xi)
        grad_r = np.zeros_like(filter_bank.sigma_xi)

        # Estimate gradient for each filter and each parameter
        for i in tqdm(range(n_filters)):
            # adjust Q
            orig_value = q_log[i]

            # Perturb positively, run filter bank
            q_linear[i] = 10**(orig_value + autograd_epsilon)
            filter_bank.filters[i].set_q(q_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, data['raw'], verbose=False)

            # compute loss for positive perturbation
            obj_ctx = build_objective_context(data, filter_dict_plus['x'], objective_name)
            loss_plus = loss_fn(obj_ctx)

            # Perturb negatively
            q_linear[i] = 10**(orig_value - autograd_epsilon)
            filter_bank.filters[i].set_q(q_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, data['raw'], verbose=False)

            # compute loss for negative perturbation
            obj_ctx = build_objective_context(data, filter_dict_minus['x'], objective_name)
            loss_minus = loss_fn(obj_ctx)

            # Restore original parameter
            q_linear[i] = 10**orig_value
            filter_bank.filters[i].set_q(q_linear[i])

            # Central finite difference for the Q matrices
            grad_q[i] = (loss_plus - loss_minus) / (2.0 * autograd_epsilon)

            # R scalar
            orig_r = r_log[i]

            # Perturb log space positively
            r_linear[i] = 10**(orig_r + autograd_epsilon)
            filter_bank.filters[i].set_r(r_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, data['raw'], verbose=False)

            # compute loss for positive perturbation
            obj_ctx = build_objective_context(data, filter_dict_plus['x'], objective_name)
            loss_plus = loss_fn(obj_ctx)

            # Perturb log space negatively, convert to linear, compute loss
            r_linear[i] = 10**(orig_r - autograd_epsilon)
            filter_bank.filters[i].set_r(r_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, data['raw'], verbose=False)

            # compute loss for negative perturbation
            obj_ctx = build_objective_context(data, filter_dict_minus['x'], objective_name)
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
    filter_dict = run_filter_bank(filter_bank, data['raw'], verbose=False)
    loss = loss_fn(data, filter_dict)
    print(f"Final Loss after {n_epochs} epochs: {loss:.6f}")
    plt.plot(range(n_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    return filter_bank


def plot_filter_bank(meas, bank_dict, dt):
    # extract values
    bank_x = bank_dict['x']
    amp = bank_dict['amp']
    phase = bank_dict['phi']

    # plot filter estimates and system dynamics
    t = np.arange(0, len(meas) * dt, dt)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].scatter(t, meas, color='red', s=5)
    axs[0, 0].plot(t, bank_x[:, 0], color='k')
    axs[0, 0].set_title(f'Position Estimation')

    axs[1, 0].plot(t, bank_x[:, 1], color='k')
    axs[1, 0].set_title(f'Velocity Estimation')

    axs[0, 1].plot(t, amp)
    axs[0, 1].set_title(f'Amplitude Estimation')

    axs[1, 1].plot(t, np.abs(phase))
    axs[1, 1].set_title(f'(Absolute) Phase Estimation')

    plt.show()


if __name__ == "__main__":
    import pandas as pd
    sys.path.append('..')
    import util

    # get training data
    conn = util.connect('btc', db_root='../data')
    train_selection = util.selector()
    train_selection.start_time = datetime.datetime(2025, 7, 31).timestamp()
    train_selection.stop_time = datetime.datetime(2025, 8, 10).timestamp() - 1
    train_price_df = util.fetch_price_space(conn=conn, selection=train_selection)

    # get test data
    test_selection = util.selector()
    test_selection.start_time = datetime.datetime(2025, 8, 11).timestamp()
    test_selection.stop_time = datetime.datetime(2025, 8, 12).timestamp() - 1
    test_price_df = util.fetch_price_space(conn=conn, selection=test_selection)

    # read in measurement data
    z_train, z_test = ((train_price_df.Open.values - train_price_df.Open.values[0]) / train_price_df.Open.values[0],
                       (test_price_df.Open.values - train_price_df.Open.values[0]) / test_price_df.Open.values[0])
    dt = len(train_price_df.Date.unique()) / len(train_price_df)

    # Define the frequencies for sinusoidal components
    omegas = np.array([0.02, 0.66, 2.04, 3.9])

    # produce truth data
    pad_len = util.compute_pad_length(z_train)
    z_pad, pad_bounds = util.pad_signal(z_train, L=pad_len)
    fft_dict = util.extract_low_pass_components(z_pad, dt, max_freq=2.0)  # np.ceil(max(omegas))
    fft_dict['raw'] = z_train
    fft_dict['raw_test'] = z_test
    fft_dict['truth'] = fft_dict['truth'][pad_bounds[0]:pad_bounds[1]]

    # Construct the filter bank
    fbank = SinusoidalFilterBank(
        omegas=omegas,
        dt=dt,
        sigma_xi=[1e1]*len(omegas),
        rho=[1e-1]*len(omegas),
    )

    # warm up the filter bank's P matrix
    # filter_out = run_filter_bank(fbank, fft_dict['raw'][:200], verbose=False)
    # fbank.P = filter_out['p'][-1]

    # Train the filter bank
    trained_filter_bank = train_filter_bank_adam(filter_bank=fbank,
                                                 data=fft_dict,
                                                 objective_name='pos_mse',
                                                 n_epochs=10,
                                                 alpha=25e-2,  # 1e-1,
                                                 reset_cov=False)

    # reset the base filter
    for i, f in enumerate(trained_filter_bank.filters):
        f.x = np.zeros_like(f.x)

    # run and plot the trained filter
    trained_filter_dict = run_filter_bank(trained_filter_bank, fft_dict['raw'])
    plot_filter_bank(fft_dict['raw'], trained_filter_dict, dt)

    # Display the optimised noise parameters
    for idx, filt in enumerate(trained_filter_bank.filters):
        q_diag = np.diag(filt.Q)
        r_val = float(filt.R[0, 0])
        print(f"Filter {idx}: Q diag = {q_diag}, R = {r_val}")
