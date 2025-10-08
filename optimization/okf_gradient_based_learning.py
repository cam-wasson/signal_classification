"""
option2_okf_script.py
======================

This script demonstrates the *Optimized Kalman Filter* (OKF) approach
for tuning the process and measurement noise covariances of a bank of
sinusoidal Kalman filters.  The method minimises a user‑defined
objective function (here, a mean squared error) by adjusting the
Cholesky factors of the process noise ``Q`` and the measurement noise
``R`` using gradient descent.  The Cholesky parameterisation ensures
that both ``Q`` and ``R`` remain symmetric positive definite (SPD)
during optimisation【812187841711290†L340-L377】.

Unlike the closed‑form backpropagation approach, this example uses
finite‑difference gradients to estimate the derivatives with respect
to the Cholesky parameters.  This keeps the implementation
self‑contained and avoids reliance on automatic differentiation
libraries that may not be available in the runtime environment.  Once
closed‑form derivatives are available, the finite‑difference loop can
be replaced for improved efficiency.

The script includes the following components:

* A base ``KalmanFilter`` class with predict and update methods.
* A ``SinusoidalKalmanFilter`` subclass configured via user‑provided
  functions for the state transition, process noise, observation, and
  measurement noise matrices.
* A ``SinusoidalFilterBank`` to manage multiple filters (one per
  frequency) and run them in parallel.
* A gradient‑descent routine that adjusts the lower‑triangular
  Cholesky factors of ``Q`` (2×2) and the scalar measurement noise
  (represented as its square root) for each filter in the bank.
* Synthetic data generation for testing the optimisation on a simple
  sinusoidal tracking problem.

This implementation follows the Optimized Kalman Filter paradigm
outlined in the literature, where noise covariances are not derived
from noise statistics but are instead *learned* to minimise the
filtering error【812187841711290†L340-L377】.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """A simple linear Kalman filter implementation.

    The filter estimates a state vector ``x`` and its covariance ``P``
    given a linear state transition model, process noise, measurement
    model and measurement noise.  It is self‑contained and does not
    depend on external libraries.
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        # State estimate
        self.x = np.zeros(dim_x)
        # State covariance
        self.P = np.eye(dim_x)
        # State transition matrix
        self.F = np.eye(dim_x)
        # Process noise covariance
        self.Q = np.eye(dim_x)
        # Measurement model
        self.H = np.zeros((dim_z, dim_x))
        # Measurement noise covariance
        self.R = np.eye(dim_z)

    def predict(self) -> None:
        """Prediction step of the Kalman filter.

        Updates the state and covariance forward in time using the
        transition model and process noise.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update step of the Kalman filter using the measurement ``z``.

        The innovation, innovation covariance, Kalman gain, and the
        updated state and covariance are computed.
        """
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Innovation
        y = z - self.H @ self.x
        # Update state
        self.x = self.x + K @ y
        # Update covariance (Joseph form)
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T


def r_matrix(meas_shape: int, rho: float) -> np.ndarray:
    """Construct an identity measurement noise matrix scaled by ``rho``."""
    return rho * np.eye(meas_shape)


def h_matrix(x_shape: int, z_shape: int) -> np.ndarray:
    """Construct a simple observation matrix mapping state to measurement.

    The first ``z_shape`` state variables are assumed to be observed directly.
    """
    H = np.zeros((z_shape, x_shape))
    idx = np.arange(min(z_shape, x_shape))
    H[idx, idx] = 1.0
    return H


def sinusoidal_f_matrix(omega: float, dt: float) -> np.ndarray:
    """Compute the state transition matrix for a sinusoidal oscillator."""
    return np.array([
        [np.cos(omega * dt), np.sin(omega * dt) / omega],
        [-omega * np.sin(omega * dt), np.cos(omega * dt)],
    ])


def sinusoidal_q_matrix(omega: float, dt: float, sigma_xi: float) -> np.ndarray:
    """Return the process noise covariance for a sinusoidal oscillator."""
    q = sigma_xi ** 2 * dt
    return q * np.array([
        [
            (2 * omega * dt - np.sin(2 * omega * dt)) / (4 * omega ** 3),
            (np.sin(omega * dt) ** 2) / (2 * omega ** 2),
        ],
        [
            (np.sin(omega * dt) ** 2) / (2 * omega ** 2),
            dt / 2 + np.sin(2 * omega * dt) / (4 * omega),
        ],
    ])


class SinusoidalKalmanFilter(KalmanFilter):
    """A Kalman filter configured for a sinusoidal oscillator.

    The filter uses the user‑provided functions to set the state
    transition and noise matrices.  The state is two‑dimensional
    ``[y, v]`` representing position and velocity.  The measurement is
    one‑dimensional observing the position directly.
    """

    def __init__(
        self,
        dim_x: int = 2,
        dim_z: int = 1,
        omega: float = np.pi / 4,
        dt: float = 0.0,
        sigma_xi: float = 0.1,
        rho: float = 1e-2,
    ) -> None:
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        # Reset state and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 1e5
        # Set the model matrices
        self.F = sinusoidal_f_matrix(omega, dt)
        self.Q = sinusoidal_q_matrix(omega, dt, sigma_xi)
        self.H = h_matrix(dim_x, dim_z)
        self.R = r_matrix(dim_z, rho)
        # Store for amplitude/phase
        self.dt = dt
        self.omega = omega

    def amplitude(self) -> float:
        """Compute the estimated amplitude of the sinusoid."""
        return np.sqrt(self.x[0] ** 2 + (self.x[1] / self.omega) ** 2)

    def phase(self) -> float:
        """Compute the estimated phase of the sinusoid (in radians)."""
        return np.arctan2(self.x[0], self.x[1] / self.omega)


class SinusoidalFilterBank:
    """A collection of sinusoidal Kalman filters, one per frequency."""

    def __init__(
        self,
        omegas: np.ndarray,
        dt: float = 0.0,
        sigma_xi: np.ndarray | float = 0.1,
        rho: np.ndarray | float = 1e-2,
    ) -> None:
        self.omegas = np.asarray(omegas)
        self.dt = dt
        # Broadcast noise parameters to arrays
        self.sigma_xi = (
            np.asarray(sigma_xi) if np.ndim(sigma_xi) > 0 else np.full_like(self.omegas, sigma_xi)
        )
        self.rho = (
            np.asarray(rho) if np.ndim(rho) > 0 else np.full_like(self.omegas, rho)
        )
        self.filters: list[SinusoidalKalmanFilter] = []
        self.build_filter_bank()

    def build_filter_bank(self) -> None:
        """Initialise a filter for each frequency in ``omegas``."""
        self.filters = []
        for o, s, r in zip(self.omegas, self.sigma_xi, self.rho):
            filt = SinusoidalKalmanFilter(
                dim_x=2,
                dim_z=1,
                omega=o,
                dt=self.dt,
                sigma_xi=s,
                rho=r,
            )
            self.filters.append(filt)

    def reset(self) -> None:
        """Reset all filters to their initial states and covariances."""
        for filt, o, s, r in zip(self.filters, self.omegas, self.sigma_xi, self.rho):
            filt.x = np.zeros(2)
            filt.P = np.eye(2) * 1e5
            filt.F = sinusoidal_f_matrix(o, self.dt)
            filt.Q = sinusoidal_q_matrix(o, self.dt, s)
            filt.H = h_matrix(2, 1)
            filt.R = r_matrix(1, r)

    def predict_update(self, z: float) -> None:
        """Run predict and update steps for each filter using measurement ``z``."""
        meas = np.array([z])
        for filt in self.filters:
            filt.predict()
            filt.update(meas)

    def get_states(self) -> np.ndarray:
        """Return a stack of current state estimates from all filters."""
        return np.stack([filt.x.copy() for filt in self.filters], axis=0)

    def get_amplitudes(self) -> np.ndarray:
        """Return the estimated amplitude for each filter."""
        return np.array([filt.amplitude() for filt in self.filters])

    def get_phases(self) -> np.ndarray:
        """Return the estimated phase for each filter."""
        return np.array([filt.phase() for filt in self.filters])


def generate_sinusoidal_data(
    omegas: np.ndarray,
    n_steps: int,
    dt: float,
    amplitudes: np.ndarray | float = 1.0,
    phases: np.ndarray | float | None = None,
    process_noise: float = 0.0,
    measurement_noise: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Generate synthetic sinusoidal measurements and ground truth.

    The measurement is a sum of sinusoidal positions with optional
    additive noise.  True position and velocity trajectories are
    returned separately for each component.
    """
    omegas = np.asarray(omegas)
    n_components = len(omegas)
    amplitudes = (
        np.asarray(amplitudes) if np.ndim(amplitudes) > 0 else np.full(n_components, amplitudes)
    )
    if phases is None:
        phases = np.random.uniform(0.0, 2.0 * np.pi, size=n_components)
    phases = (
        np.asarray(phases) if np.ndim(phases) > 0 else np.full(n_components, phases)
    )
    t = np.arange(n_steps) * dt
    y_truth: list[np.ndarray] = []
    v_truth: list[np.ndarray] = []
    # Generate ground truth per component
    for amp, omega, phi in zip(amplitudes, omegas, phases):
        y = amp * np.sin(omega * t + phi)
        v = amp * omega * np.cos(omega * t + phi)
        if process_noise > 0.0:
            y = y + np.random.normal(0.0, process_noise, size=n_steps)
            v = v + np.random.normal(0.0, process_noise, size=n_steps)
        y_truth.append(y)
        v_truth.append(v)
    # Sum positions for the measurement
    z = np.sum(y_truth, axis=0)
    if measurement_noise > 0.0:
        z = z + np.random.normal(0.0, measurement_noise, size=n_steps)
    return z, y_truth, v_truth


def compute_mse_loss(
    filter_bank: SinusoidalFilterBank,
    measurements: np.ndarray,
    y_truth: list[np.ndarray],
    v_truth: list[np.ndarray],
    reset_filters: bool = True,
) -> float:
    """Compute the mean squared error between filter estimates and truth.

    The filter bank is run on the measurement sequence.  The error
    between the predicted states and the true states is accumulated
    across time and filters and averaged.
    """
    n_steps = measurements.shape[0]
    n_filters = len(filter_bank.filters)
    if reset_filters:
        filter_bank.reset()
    total_error = 0.0
    for k in range(n_steps):
        z_k = measurements[k]
        filter_bank.predict_update(z_k)
        states = filter_bank.get_states()
        for i in range(n_filters):
            y_pred = states[i, 0]
            v_pred = states[i, 1]
            y_true = y_truth[i][k]
            v_true = v_truth[i][k]
            total_error += (y_pred - y_true) ** 2 + (v_pred - v_true) ** 2
    mse = total_error / (n_steps * n_filters)
    return mse


def initialise_cholesky_params(filter_bank: SinusoidalFilterBank) -> tuple[np.ndarray, np.ndarray]:
    """Extract the Cholesky parameters for Q and R from a filter bank.

    Each 2×2 covariance ``Q`` is represented by a lower‑triangular
    matrix with positive diagonal entries.  For each filter ``i``
    ``Q_i = L_{Q,i} @ L_{Q,i}.T`` where
    ``L_{Q,i} = [[l00, 0], [l10, l11]]``.  The measurement covariance
    ``R_i`` is represented by its Cholesky factor ``l_R`` such that
    ``R_i = l_R**2``.  This function converts the existing ``Q`` and
    ``R`` matrices into arrays of these parameters.

    Returns
    -------
    lq_params : ndarray, shape (n_filters, 3)
        The parameters ``[l00, l10, l11]`` for each filter.
    lr_params : ndarray, shape (n_filters,)
        The scalar Cholesky factor ``l_R`` for each filter.
    """
    n_filters = len(filter_bank.filters)
    lq_params = np.zeros((n_filters, 3))
    lr_params = np.zeros(n_filters)
    for i, filt in enumerate(filter_bank.filters):
        # For Q = L L^T, recover approximate L via Cholesky decomposition
        # Use np.linalg.cholesky to compute L.  Since Q may be small PSD,
        # the decomposition will succeed.
        L = np.linalg.cholesky(filt.Q)
        lq_params[i] = [L[0, 0], L[1, 0], L[1, 1]]
        # For R = [[r]], the Cholesky factor is sqrt(r)
        lr_params[i] = np.sqrt(filt.R[0, 0])
    return lq_params, lr_params


def apply_cholesky_params(filter_bank: SinusoidalFilterBank, lq_params: np.ndarray, lr_params: np.ndarray) -> None:
    """Update the Q and R matrices of a filter bank from Cholesky parameters.

    Parameters
    ----------
    filter_bank : SinusoidalFilterBank
        The filter bank to update.
    lq_params : ndarray of shape (n_filters, 3)
        The Cholesky parameters ``[l00, l10, l11]`` for each filter.
    lr_params : ndarray of shape (n_filters,)
        The scalar Cholesky factor for each filter's measurement noise.
    """
    for i, filt in enumerate(filter_bank.filters):
        l00, l10, l11 = lq_params[i]
        # Construct lower triangular L
        L = np.array([[l00, 0.0], [l10, l11]])
        # Compute Q = L @ L.T
        filt.Q = L @ L.T
        # Compute R = l_R**2
        l_R = lr_params[i]
        filt.R = np.array([[l_R ** 2]])


def train_filter_bank_okf(
    filter_bank: SinusoidalFilterBank,
    measurements: np.ndarray,
    y_truth: list[np.ndarray],
    v_truth: list[np.ndarray],
    n_epochs: int = 10,
    lr: float = 1e-2,
    epsilon: float = 1e-4,
    verbose: bool = True,
) -> None:
    """Optimise the noise parameters using gradient descent on Cholesky factors.

    This routine minimises a mean squared error objective by adjusting
    the Cholesky parameters of ``Q`` and ``R`` for each filter.  A
    central difference approximation is used to estimate the gradient
    with respect to each parameter.  After each update, the diagonal
    elements of the Cholesky factors are clamped to a small positive
    value to ensure positive definiteness of ``Q`` and ``R``.

    Parameters
    ----------
    filter_bank : SinusoidalFilterBank
        The bank of filters to optimise.
    measurements : ndarray
        The measurement sequence.
    y_truth : list of ndarrays
        True position trajectories.
    v_truth : list of ndarrays
        True velocity trajectories.
    n_epochs : int
        Number of optimisation epochs.
    lr : float
        Learning rate for gradient descent.
    epsilon : float
        Perturbation size for finite difference estimation.
    verbose : bool
        Whether to print progress messages.
    """
    n_filters = len(filter_bank.filters)
    # Initialise parameters from the current Q and R matrices
    lq_params, lr_params = initialise_cholesky_params(filter_bank)
    for epoch in range(n_epochs):
        # Apply current parameters to the filter bank
        apply_cholesky_params(filter_bank, lq_params, lr_params)
        # Compute current loss
        loss = compute_mse_loss(filter_bank, measurements, y_truth, v_truth)
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}")
        # Allocate gradient arrays
        grad_lq = np.zeros_like(lq_params)
        grad_lr = np.zeros_like(lr_params)
        # Loop over each filter and parameter to estimate gradient
        for i in range(n_filters):
            # Cholesky parameters for Q: l00, l10, l11
            for j in range(3):
                orig_value = lq_params[i, j]
                # Perturb positively
                lq_params[i, j] = orig_value + epsilon
                apply_cholesky_params(filter_bank, lq_params, lr_params)
                loss_plus = compute_mse_loss(filter_bank, measurements, y_truth, v_truth)
                # Perturb negatively
                lq_params[i, j] = orig_value - epsilon
                apply_cholesky_params(filter_bank, lq_params, lr_params)
                loss_minus = compute_mse_loss(filter_bank, measurements, y_truth, v_truth)
                # Restore original parameter
                lq_params[i, j] = orig_value
                apply_cholesky_params(filter_bank, lq_params, lr_params)
                # Central difference gradient
                grad_lq[i, j] = (loss_plus - loss_minus) / (2.0 * epsilon)
            # Parameter for R: l_R
            orig_lr = lr_params[i]
            # Perturb positively
            lr_params[i] = orig_lr + epsilon
            apply_cholesky_params(filter_bank, lq_params, lr_params)
            loss_plus = compute_mse_loss(filter_bank, measurements, y_truth, v_truth)
            # Perturb negatively
            lr_params[i] = orig_lr - epsilon
            apply_cholesky_params(filter_bank, lq_params, lr_params)
            loss_minus = compute_mse_loss(filter_bank, measurements, y_truth, v_truth)
            # Restore original parameter
            lr_params[i] = orig_lr
            apply_cholesky_params(filter_bank, lq_params, lr_params)
            grad_lr[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
        # Update parameters using gradient descent
        lq_params -= lr * grad_lq
        lr_params -= lr * grad_lr
        # Enforce positivity on Cholesky diagonal elements (l00, l11) and l_R
        lq_params[:, 0] = np.maximum(lq_params[:, 0], 1e-6)
        lq_params[:, 2] = np.maximum(lq_params[:, 2], 1e-6)
        lr_params = np.maximum(lr_params, 1e-6)
    # Apply final parameters
    apply_cholesky_params(filter_bank, lq_params, lr_params)
    # Final loss reporting
    final_loss = compute_mse_loss(filter_bank, measurements, y_truth, v_truth)
    if verbose:
        print(f"Final Loss after {n_epochs} epochs: {final_loss:.6f}")


if __name__ == "__main__":
    # Example run of the OKF optimisation on synthetic data
    np.random.seed(123)
    # Define frequencies for a bank of three filters
    omegas = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
    dt = 0.1
    n_steps = 200
    # Generate synthetic measurements
    z, y_true_list, v_true_list = generate_sinusoidal_data(
        omegas=omegas,
        n_steps=n_steps,
        dt=dt,
        amplitudes=1.0,
        phases=None,
        process_noise=0.0,
        measurement_noise=0.05,
    )
    # Construct the filter bank
    fbank = SinusoidalFilterBank(
        omegas=omegas,
        dt=dt,
        sigma_xi=0.1,
        rho=0.1,
    )
    # Train the filter bank using the OKF optimisation
    train_filter_bank_okf(
        filter_bank=fbank,
        measurements=z,
        y_truth=y_true_list,
        v_truth=v_true_list,
        n_epochs=5,
        lr=0.1,
        epsilon=1e-4,
        verbose=True,
    )
    # Print final noise parameters for each filter
    for idx, filt in enumerate(fbank.filters):
        # Extract diagonal of Q and R value for display
        Q = filt.Q
        R = filt.R
        diag_Q = np.diag(Q)
        r_val = float(R[0, 0])
        print(f"Filter {idx}: Q diag = {diag_Q}, R = {r_val}")