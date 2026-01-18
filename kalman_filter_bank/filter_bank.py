import numpy as np
from tqdm import tqdm

from kalman_filter_bank.kalman_filter import SinusoidalKalmanFilter
from util import pad_signal, extract_low_pass_components


import numpy as np


class FilterBank:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)
        self.filters = []
        self.N = self.__len__()

    def build_filter_bank(self):
        pass

    # ------------------------------------------------------------------
    # Core shared execution loop (semantic-free)
    # ------------------------------------------------------------------
    def _run_filters(self, meas, update_residual):
        """
        Shared internal loop for running filters.
        Residual semantics are injected via update_residual.
        """
        x_preds = []
        p_preds = []

        residual = meas

        for f in self.filters:
            z = np.array([residual]).reshape(f.z.shape)

            f.predict()
            f.update(z)

            residual = update_residual(residual, f)

            x_preds.append(f.x.copy())
            p_preds.append(f.P.copy())

        return residual, x_preds, p_preds

    # ------------------------------------------------------------------
    # Utility accessors
    # ------------------------------------------------------------------
    def amplitudes(self):
        amps = np.zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            amps[i] = f.amplitude()
        return amps

    def phases(self):
        phase = np.zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            phase[i] = f.phase()
        return phase

    def reset_states(self):
        for f in self.filters:
            f.x = np.zeros_like(f.x)

    def __len__(self):
        return len(self.filters)


# ----------------------------------------------------------------------
# Residual semantics (the ONLY difference between tracking & discrimination)
# ----------------------------------------------------------------------
def cascade_residual(residual, f):
    return residual - f.x[0]


def parallel_residual(residual, f):
    return residual


# ----------------------------------------------------------------------
# Sinusoidal Filter Bank
# ----------------------------------------------------------------------
class SinusoidalFilterBank(FilterBank):
    def __init__(
        self,
        dim_x=2,
        dim_z=1,
        omegas=None,
        dt=0.0,
        sigma_xi=0.1,
        rho=1e-2,
        p0=None,
    ):
        super().__init__(dim_x, dim_z)

        self.filters = []
        self.omegas = omegas
        self.dt = dt
        self.sigma_xi = sigma_xi
        self.rho = rho

        if p0 is None:
            p0 = 1e-2
        self.p0 = p0

        self.build_filter_bank()

    def build_filter_bank(self):
        for o, s, r in zip(self.omegas, self.sigma_xi, self.rho):
            skf = SinusoidalKalmanFilter(
                dim_x=self.dim_x,
                dim_z=self.dim_z,
                omega=o,
                dt=self.dt,
                sigma_xi=s,
                rho=r,
                p0=self.p0,
            )
            self.filters.append(skf)

        self.N = self.__len__()

    # ------------------------------------------------------------------
    # TRACKING FILTER (cascading, stationary signal generator)
    # ------------------------------------------------------------------
    def step_tracking(self, meas):
        """
        Cascading filter bank.
        Produces a stationary residual.
        """
        residual, x_preds, p_preds = self._run_filters(
            meas,
            update_residual=cascade_residual,
        )

        # Optional: retain aggregate state if you still want it
        x_sum = np.sum(np.array(x_preds), axis=0)
        P_sum = np.sum(np.stack(p_preds, axis=0), axis=0)

        return residual, x_sum, P_sum

    # ------------------------------------------------------------------
    # DISCRIMINATION FILTER (parallel, state-producing)
    # ------------------------------------------------------------------
    def step_discrimination(self, meas):
        """
        Parallel filter bank.
        Produces per-filter states for datamining / learning.
        """
        _, x_preds, p_preds = self._run_filters(
            meas,
            update_residual=parallel_residual,
        )

        return np.array(x_preds), np.array(p_preds)


class NarrowbandTrackingFilterBank(SinusoidalFilterBank):
    DEFAULT_OMEGAS = np.array([0.1, 0.17, 0.26], dtype=float)

    def __init__(
        self,
        dim_x=2,
        dim_z=1,
        dt=0.0,
        sigma_xi=0.1,
        rho=1e-2,
        p0=None,
        omegas=None,
    ):
        if omegas is None:
            omegas_arr = self.DEFAULT_OMEGAS.copy()
        else:
            omegas_arr = np.array(omegas, dtype=float)

        n = len(omegas_arr)

        # Ensures sigma_xi and rho are iterable arrays of length n.
        if np.isscalar(sigma_xi):
            sigma_xi_arr = np.full(n, float(sigma_xi), dtype=float)
        else:
            sigma_xi_arr = np.array(sigma_xi, dtype=float)
            if sigma_xi_arr.shape[0] != n:
                raise ValueError(
                    f"sigma_xi must have length {n}, got {sigma_xi_arr.shape[0]}")
        if np.isscalar(rho):
            rho_arr = np.full(n, float(rho), dtype=float)
        else:
            rho_arr = np.array(rho, dtype=float)
            if rho_arr.shape[0] != n:
                raise ValueError(
                    f"rho must have length {n}, got {rho_arr.shape[0]}")

        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
            omegas=omegas_arr,
            dt=dt,
            sigma_xi=sigma_xi_arr,
            rho=rho_arr,
            p0=p0,
        )

    def step(self, z):
        _, x_sum, P_sum = self.step_tracking(z)
        return x_sum, P_sum


class SinusoidalCMMEAFilterBank(SinusoidalFilterBank):
    def __init__(self, dim_x=2, dim_z=1, omegas=None, dt=0.0, sigma_xi=0.1, rho=1e-2):
        # build the parent sinusoidal filter bank
        super().__init__(dim_x=dim_x,
                         dim_z=dim_z,
                         omegas=omegas,
                         dt=dt,
                         sigma_xi=sigma_xi,
                         rho=rho)
        self.weights = np.ones(self.N)  # Uniform initial weights

    def compute_likelihood(self, kf, z):
        """Compute likelihood p(y_k | θ_i, y_1:k-1)"""
        y = z.reshape(-1, 1)
        H = kf.H
        x_pred = kf.x_prior
        P_pred = kf.P_prior
        R = kf.R

        y_resid = y - H @ x_pred
        S = H @ P_pred @ H.T + R
        S_inv = np.linalg.inv(S)
        exponent = -0.5 * (y_resid.T @ S_inv @ y_resid)
        denom = np.sqrt((2 * np.pi) ** self.dim_z * np.linalg.det(S))
        likelihood = np.exp(exponent) / denom
        return likelihood.item()

    def step(self, z):
        """Run one filter step for all filters and fuse using CMMEA logic"""
        likelihoods = np.zeros(self.N)
        x_preds = []
        P_preds = []

        for i, kf in enumerate(self.filters):
            kf.predict()
            kf.update(z)
            x_preds.append(kf.x.copy())
            P_preds.append(kf.P.copy())
            likelihoods[i] = self.compute_likelihood(kf, z)

        # print(likelihoods)

        # Update weights using Bayes rule
        prior_weights = self.weights.copy()
        numerators = prior_weights * likelihoods
        self.weights = numerators / np.sum(numerators)

        # Fuse state estimates
        x_fused = np.sum(
            [w * x for w, x in zip(self.weights, x_preds)], axis=0)

        # Fuse covariances
        P_fused = sum([
            w * (P + (x - x_fused) @ (x - x_fused).T)
            for w, x, P in zip(self.weights, x_preds, P_preds)
        ])

        return x_fused, P_fused, self.weights.copy()


def run_filter_bank(fbank, measurements, verbose=True):
    """
    Run the KF bank on measurements, returning dictionary of states, covariances, and weights
    """
    n_steps = measurements.shape[0]
    dim_x = fbank.filters[0].x.shape[0]
    n_models = len(fbank.filters)

    all_states = np.zeros((n_steps, dim_x))
    all_covs = np.zeros((n_steps, dim_x, dim_x))
    all_weights = np.zeros((n_steps, n_models))
    all_amp = np.zeros((n_steps, n_models))
    all_phi = np.zeros((n_steps, n_models))

    iterator = range(n_steps)
    if verbose:
        print(f'Running the Filter Bank for {n_steps} steps')
        iterator = tqdm(iterator)

    for k in iterator:
        z = measurements[k]
        step_out = fbank.step(z)
        all_states[k] = np.array(step_out[0])
        all_covs[k] = step_out[1]
        if len(step_out) > 2:
            all_weights[k] = step_out[2]
        all_amp[k] = fbank.amplitudes()
        all_phi[k] = fbank.phases()

    return {'x': all_states, 'p': all_covs, 'w': all_weights, 'amp': all_amp, 'phi': all_phi, 'omega': fbank.omegas}


def create_sig_dict(price_df, random_date=True):
    if random_date:
        date = price_df['Date'].sample(n=1).iloc[0]
    else:
        date = '2022-04-07'  # Good Dates: 2022-04-07, 2022-08-24, 2022-07-15
    date_df = price_df.loc[price_df.Date == date]
    rate_signal = (date_df['Open'].values -
                   date_df['Open'].values[0]) / date_df['Open'].values[0]

    # pad signal, compute omegas
    padded_signal, true_bounds = pad_signal(rate_signal)
    dt = 1 / len(rate_signal)
    sig_dict = extract_low_pass_components(padded_signal, dt, max_freq=10)
    sig_dict['truth_pad'] = sig_dict['truth']
    sig_dict['truth'] = sig_dict['truth_pad'][true_bounds[0]:true_bounds[1]]
    sig_dict['raw'] = rate_signal

    # add additional time info to the signal dictionary
    t = np.arange(0, 1, dt)
    sig_dict['t'] = t
    sig_dict['dt'] = dt

    return sig_dict


def main(price_df):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')

    n_days = len(price_df.Date.unique())

    # convert raw data into data we need to run the CMMEA
    # sig_dict = create_sig_dict(price_df)
    sig_dict = {}
    dw = 0.1
    omega_arr = np.arange(0, 5, dw) + dw
    # omega_arr = np.array([4.77374821, 9.54749642,  14.32124464,  19.09499285, 23.86874106,  28.64248927, 33.41623748,
    #                       38.1899857, 42.96373391,  47.73748212,  52.51123033,  57.28497854, 62.05872675])
    sig_dict['omega'] = omega_arr  # np.concatenate((omega_arr, omega_arr*-1))
    sig_dict['dt'] = n_days / len(price_df)
    sig_dict['raw'] = (price_df.Open.values -
                       price_df.Open.values[0]) / price_df.Open.values[0]
    sig_dict['t'] = np.arange(price_df.shape[0])

    # build the CMMEA
    cmmea = SinusoidalCMMEAFilterBank(dim_x=2,
                                      dim_z=1,
                                      omegas=sig_dict['omega'],
                                      dt=sig_dict['dt'],
                                      # model confidence
                                      sigma_xi=[1e1]*len(omega_arr),
                                      # noise confidence (1e-5 baseline)
                                      rho=[1e-4]*len(omega_arr))

    # run the CMMEA
    cmmea_dict = run_filter_bank(cmmea, sig_dict['raw'])

    # plot the results
    plt.figure(figsize=(10, 8))
    plt.title(
        f'Price Estimation {price_df.Date.values[0]} : {price_df.Date.values[-1]}')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.scatter(sig_dict['t'], sig_dict['raw'], color='red', s=3)
    plt.plot(sig_dict['t'], cmmea_dict['x'][:, 0],
             color='k', label='CMMEA Filter')

    plt.figure(figsize=(10, 8))
    plt.title('Spread')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.scatter(sig_dict['t'], sig_dict['raw'] -
                cmmea_dict['x'][:, 0], color='k', s=3)

    plt.figure(figsize=(10, 8))
    plt.title('Velocity Estimation')
    plt.plot(sig_dict['t'], cmmea_dict['x'][:, 1],
             color='k', label='CMMEA Filter')

    plt.figure()
    plt.title('CMMEA Weights')
    plt.plot(cmmea_dict['w'])
    plt.legend(sig_dict['omega'])

    plt.show()
    return cmmea_dict


if __name__ == '__main__':
    import pandas as pd
    # read in price df
    # path = 'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2025\\2_Spring\\CS-614\\FinalProject\\data\\qqq_2022.csv'
    path = 'C:/Users/cwass/OneDrive/Desktop/Stock Sim/AutomationRepo/TradeSystem/Candlestick/notebooks/btc_1m.csv'
    data_df = pd.read_csv(path)
    main(data_df)
    print('asdf')
