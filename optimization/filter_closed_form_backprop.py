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

from matplotlib import use as muse
import matplotlib.pyplot as plt

import numpy as np
import optimization_util as opt_util
from tqdm import tqdm

import sys
sys.path.append('../kalman_filter_bank')
from filter_bank import SinusoidalFilterBank, run_filter_bank
from kalman_filter import sinusoidal_q_matrix, r_matrix

muse('Qt5Agg')

OBJECTIVES = {
    'pos_mse': {
        'fn': opt_util.position_error,
        'required_keys': ['training_data', 'filter_dict'],
    },
    'vel_mse': {
        'fn': opt_util.velocity_error,
        'required_keys': ['training_data', 'filter_dict'],
    },
    # add more objectives here
}


def compute_grad_q():
    pass


def train_filter_bank(
    filter_bank: SinusoidalFilterBank,
    data: dict,
    objective_name: str,
    n_epochs: int = 10,
    lr: float = 1e0,
    decay: float = .9,
    epsilon: float = 1e-4,
    verbose: bool = True,
) -> SinusoidalFilterBank:
    """Optimise Q and R using a finite‑difference gradient descent.

    In lieu of closed‑form derivatives, this function approximates the
    gradient of the MSE loss with respect to the diagonal entries of
    ``Q`` and the scalar ``R`` for each filter.  At each epoch, each
    parameter is perturbed by ``epsilon`` in positive and negative
    directions, the loss is recomputed, and the gradient is estimated
    via the central difference.  The parameters are then updated with
    a simple gradient‑descent step and clipped to remain positive.

    Parameters
    ----------
    filter_bank : SinusoidalFilterBank
        The bank of filters to optimise.
    data : dict
        Training data dictionary. It must include:
            - 'measurements': ndarray containing the measurement sequence
            - 'pos_truth': ndarray of true positions for computing the loss
            - 'dt': float representing the time step between consecutive measurements
    objective_name : str
        Name of the objective function to use from the objective registry.
    n_epochs : int, optional
        Number of optimisation epochs.
    lr : float, optional
        Learning rate for gradient descent.
    decay: float, optional
        Decay factor for the learning rate so that the innovation jumps decrease
    epsilon : float, optional
        Perturbation size for finite difference estimation.
    verbose : bool, optional
        Whether to print progress information.
    """
    # fetch the chosen objective definition
    obj_entry = OBJECTIVES[objective_name]
    loss_fn = obj_entry['fn']

    n_filters = len(filter_bank.filters)

    # Extract current Q scalar and R scalar parameters
    q_linear = filter_bank.sigma_xi.copy()
    r_linear = filter_bank.rho.copy()
    q_log = np.log10(q_linear)
    r_log = np.log10(r_linear)

    plt.plot(data['raw'])

    for epoch in range(n_epochs):

        # Compute current loss
        filter_dict = run_filter_bank(filter_bank, data['raw'], verbose)
        loss = loss_fn(data, filter_dict)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}")
        plt.plot(filter_dict['x'][:, 0], label=f'Epoch: {epoch}')

        # Initialize gradient arrays
        grad_q = np.zeros_like(q_linear)
        grad_r = np.zeros_like(r_linear)

        # Estimate gradient for each filter and each parameter
        for i in tqdm(range(n_filters)):
            # fetch this filter's omega term
            this_omega = filter_bank.filters[i].omega

            # adjust Q
            orig_value = q_log[i]

            # Perturb positively
            q_linear[i] = 10**(orig_value + epsilon)
            filter_bank.filters[i].Q = sinusoidal_q_matrix(this_omega, dt, q_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, data['raw'], verbose)
            loss_plus = loss_fn(data, filter_dict_plus)

            # Perturb negatively
            q_linear[i] = 10**(orig_value - epsilon)
            filter_bank.filters[i].Q = sinusoidal_q_matrix(this_omega, dt, q_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, data['raw'], verbose)
            loss_minus = loss_fn(data, filter_dict_minus)

            # Restore original parameter
            q_linear[i] = 10**orig_value
            filter_bank.filters[i].Q = sinusoidal_q_matrix(this_omega, dt, q_linear[i])

            # Central finite difference
            grad_q[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

            # R scalar
            orig_r = r_log[i]

            # Perturb log space positively
            r_linear[i] = 10**(orig_r + epsilon)
            filter_bank.filters[i].R = r_linear[i]*np.eye(filter_bank.dim_z)
            filter_dict_plus = run_filter_bank(filter_bank, data['raw'], verbose)
            loss_plus = loss_fn(data, filter_dict_plus)

            # Perturb log space negatively, convert to linear, compute loss
            r_linear[i] = 10**(orig_r - epsilon)
            filter_bank.filters[i].R = r_linear[i]*np.eye(filter_bank.dim_z)
            filter_dict_minus = run_filter_bank(filter_bank, data['raw'], verbose)
            loss_minus = loss_fn(data, filter_dict_minus)

            # Restore original linear value of rho
            r_linear[i] = 10**orig_r
            filter_bank.filters[i].R = r_linear[i]*np.eye(filter_bank.dim_z)
            grad_r[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

        # Update parameters
        q_log -= lr * grad_q
        r_log -= lr * grad_r
        lr = lr*decay

        # clipping in case of NaNs
        q_log[np.isnan(q_log)] = -8.0
        r_log[np.isnan(r_log)] = -8.0

        # Update the filter bank matrices
        for i, f in enumerate(filter_bank.filters):
            f.Q = sinusoidal_q_matrix(f.omega, dt, 10**q_log[i])
            f.R = 10**r_log[i]*np.eye(f.dim_z)

    # Final loss reporting
    filter_dict = run_filter_bank(filter_bank, data['meas'], verbose)
    loss = loss_fn(data, filter_dict)
    print(f"Final Loss after {n_epochs} epochs: {loss:.6f}")
        
    return filter_bank


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
        loss = loss_fn(data, filter_dict)
        losses[epoch] = loss

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}")
        # plot_filter_bank(data['raw'], filter_dict, data['dt'])

        # Initialize gradient arrays
        grad_q = np.zeros_like(filter_bank.sigma_xi)
        grad_r = np.zeros_like(filter_bank.sigma_xi)

        # Estimate gradient for each filter and each parameter
        for i in tqdm(range(n_filters)):
            # fetch this filter's omega term
            this_omega = filter_bank.filters[i].omega

            # adjust Q
            orig_value = q_log[i]

            # Perturb positively
            q_linear[i] = 10**(orig_value + autograd_epsilon)
            filter_bank.filters[i].Q = sinusoidal_q_matrix(this_omega, dt, q_linear[i])
            filter_dict_plus = run_filter_bank(filter_bank, data['raw'], verbose=False)
            loss_plus = loss_fn(data, filter_dict_plus)

            # Perturb negatively
            q_linear[i] = 10**(orig_value - autograd_epsilon)
            filter_bank.filters[i].Q = sinusoidal_q_matrix(this_omega, dt, q_linear[i])
            filter_dict_minus = run_filter_bank(filter_bank, data['raw'], verbose=False)
            loss_minus = loss_fn(data, filter_dict_minus)

            # Restore original parameter
            q_linear[i] = 10**orig_value
            filter_bank.filters[i].Q = sinusoidal_q_matrix(this_omega, dt, q_linear[i])

            # Central finite difference
            grad_q[i] = (loss_plus - loss_minus) / (2.0 * autograd_epsilon)

            # R scalar
            orig_r = r_log[i]

            # Perturb log space positively
            r_linear[i] = 10**(orig_r + autograd_epsilon)
            filter_bank.filters[i].R = r_linear[i]*np.eye(filter_bank.dim_z)
            filter_dict_plus = run_filter_bank(filter_bank, data['raw'], verbose=False)
            loss_plus = loss_fn(data, filter_dict_plus)

            # Perturb log space negatively, convert to linear, compute loss
            r_linear[i] = 10**(orig_r - autograd_epsilon)
            filter_bank.filters[i].R = r_linear[i]*np.eye(filter_bank.dim_z)
            filter_dict_minus = run_filter_bank(filter_bank, data['raw'], verbose=False)
            loss_minus = loss_fn(data, filter_dict_minus)

            # Restore original linear value of rho
            r_linear[i] = 10**orig_r
            filter_bank.filters[i].R = r_linear[i]*np.eye(filter_bank.dim_z)
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
            f.Q = sinusoidal_q_matrix(f.omega, dt, 10**q_log[i])
            f.R = 10**r_log[i]*np.eye(f.dim_z)
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
    # read in measurement data
    path = 'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2025\\4_Fall\\CS-591\\Notebook\\btc_1m.csv'
    df = pd.read_csv(path)
    z = (df.Open.values - df.Open.values[0]) / df.Open.values[0]
    dt = len(df.Date.unique()) / len(df)

    # Define the frequencies for sinusoidal components
    omegas = np.array([0.02, 0.66, 2.04, 3.9])

    # produce truth data
    pad_len = util.compute_pad_length(z)
    z_pad, pad_bounds = util.pad_signal(z, L=pad_len)
    fft_dict = util.extract_low_pass_components(z_pad, dt, max_freq=2.0)  # np.ceil(max(omegas))
    fft_dict['raw'] = z
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

    # Train the filter bank using the Option 1 optimisation (finite differences)
    # trained_filter_bank = train_filter_bank(
    #                             filter_bank=fbank,
    #                             data=fft_dict,
    #                             objective_name='pos_mse',
    #                             n_epochs=10,
    #                             lr=1e-1,
    #                             epsilon=1e-4,
    #                             verbose=False,
    #                         )
    trained_filter_bank = train_filter_bank_adam(filter_bank=fbank,
                                                 data=fft_dict,
                                                 objective_name='pos_mse',
                                                 n_epochs=28,
                                                 alpha=1e-1,
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
