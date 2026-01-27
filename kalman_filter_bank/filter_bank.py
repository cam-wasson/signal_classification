import numpy as np
from tqdm import tqdm

from kalman_filter_bank.kalman_filter import SinusoidalKalmanFilter
from util import pad_signal, extract_low_pass_components


from dataclasses import dataclass


@dataclass
class FilterBankObservation:
    meas: float
    spread: float
    filter_pos: float
    filter_vel: float
    emp_vel: float
    emp_acc: float


@dataclass
class FilterBankCache:
    meas: list
    spread: list
    filter_pos: list
    filter_vel: list
    emp_vel: list
    emp_acc: list

    def update(self, ob: FilterBankObservation):
        pass


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

    def run(self, measurements, verbose=True):
        measurements = np.asarray(measurements, dtype=float).reshape(-1)
        n_steps = measurements.shape[0]

        if len(self.filters) == 0:
            raise ValueError(
                "Filter bank has no filters. Did you call build_filter_bank()?")

        dim_x = self.filters[0].x.shape[0]
        n_models = len(self.filters)

        all_states = np.zeros((n_steps, dim_x))
        all_covs = np.zeros((n_steps, dim_x, dim_x))
        all_weights = np.zeros((n_steps, n_models))
        all_amp = np.zeros((n_steps, n_models))
        all_phi = np.zeros((n_steps, n_models))

        iterator = range(n_steps)
        if verbose:
            print(f"Running the Filter Bank for {n_steps} steps")
            iterator = tqdm(iterator)

        for k in iterator:
            z = measurements[k]
            step_out = self.step(z)

            # Support both tuple-returning banks (tracking) and dict-returning banks (DiscriminationFilterBank)
            if isinstance(step_out, dict):
                # Expect DiscriminationFilterBank-style output with "combined"
                combined = step_out.get("combined", None)
                if combined is None:
                    raise ValueError(
                        "step() returned a dict but did not include a 'combined' key.")
                all_states[k] = np.array(combined["x"])
                all_covs[k] = np.array(combined["P"])

            else:
                all_states[k] = np.array(step_out[0])
                all_covs[k] = np.array(step_out[1])
                if len(step_out) > 2:
                    all_weights[k] = np.array(step_out[2])

            all_amp[k] = self.amplitudes()
            all_phi[k] = self.phases()

        return {"x": all_states, "p": all_covs, "w": all_weights,
                "amp": all_amp, "phi": all_phi, "omega": self.omegas}

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
        omegas=np.array([0.1]),
        dt=1/(24*60),
        sigma_xi=np.array([1e1]),
        rho=np.array([1e-1]),
        p0=None,
    ):
        super().__init__(dim_x, dim_z)

        self.filters = []
        self.omegas = omegas
        self.dt = dt
        self.sigma_xi = np.array(sigma_xi)
        self.rho = np.array(rho)

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

    def set_q(self, sigma_xi):
        self.sigma_xi = sigma_xi
        for i, f in enumerate(self.filters):
            f.set_q(sigma_xi[i])

    def set_r(self, rho):
        self.rho = rho
        for i, f in enumerate(self.filters):
            f.set_r(rho[i])

    def reset_cov(self):
        for i, f in enumerate(self.filters):
            f.set_p(self.p0)

    def step(self, residual):
        x_preds = []
        p_preds = []
        for i, f in enumerate(self.filters):
            # format the residual value into a measurement
            meas = np.array([residual]).reshape(f.z.shape)

            # run the kf
            f.predict()
            f.update(meas)

            # recompute the residual
            residual -= f.x[0]

            # store
            x_preds.append(f.x.copy())
            p_preds.append(f.P.copy())

        # fuse filter states and covariance
        x_sum = np.sum(np.array(x_preds), axis=0)
        P_sum = np.sum(np.stack(p_preds, axis=0), axis=0)

        return x_sum, P_sum, np.ones_like(self.omegas)

    # ------------------------------------------------------------------
    # TRACKING FILTER (cascading, stationary signal generator)
    # ------------------------------------------------------------------
    def step_cascade(self, meas):
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
    def step_parallel(self, meas):
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
        _, x_sum, P_sum = self.step_cascade(z)
        return x_sum, P_sum


class DiscriminationFilterBank(SinusoidalFilterBank):
    DEFAULT_OMEGAS = np.arange(0.1, 2.1, .1, dtype=float)

    def __init__(self,
                 dim_x=2,
                 dim_z=1,
                 dt=0.0,
                 sigma_xi=0.1,
                 rho=1e-2,
                 p0=None,
                 omegas=None,
                 use_iae=True,
                 step='cascade'):

        if omegas is None:
            omegas_arr = self.DEFAULT_OMEGAS.copy()
        else:
            omegas_arr = omegas
        n = len(omegas_arr)

        self._step = step

        # set IAE logic here
        self._use_iae = use_iae
        if use_iae:
            self.alpha_ema = 0.05  # EMA smoothing for NIS
            self.eta_R = 0.02  # adaptation rate for R scale
            self.eta_Q = 0.005  # adaptation rate for Q scale (usually slower)
            self.R_bounds = (1e-3, 1e3)  # bounds on scalar beta
            self.Q_bounds = (1e-6, 1e6)  # bounds on scalar alpha

            # Log-scales (log-space keeps positivity)
            self.log_beta = [1.0] * n  # for R
            self.log_alpha = [1.0] * n  # for Q
            self.adapt = 'R'

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

        if self._use_iae:
            # store the base matrices of the filter bank
            self.R0_arr = np.zeros(np.concatenate(
                [[len(self.filters)], self.filters[0].R.shape]))
            self.Q0_arr = np.zeros(np.concatenate(
                [[len(self.filters)], self.filters[0].Q.shape]))
            for i, kf in enumerate(self.filters):
                self.R0_arr[i] = kf.R.copy()
                self.Q0_arr[i] = kf.Q.copy()

    def step(self, residual):
        # Target mean of NIS is dim_z (for consistent KF)
        nis_target, nis_ema = float(self.dim_z), float(self.dim_z)

        # compute per-filter and combined output here
        cache = {'combined': {"meas": residual,
                              "x": np.array([]),
                              "P": np.array([])}}
        x_combined = np.zeros([self.dim_x], dtype=float)
        P_combined = np.zeros([self.dim_x, self.dim_x], dtype=float)
        for i, kf in enumerate(self.filters):
            if self._use_iae:
                # ---- Apply current scalings BEFORE predict/update ----
                beta = float(
                    np.clip(np.exp(self.log_beta[i]), self.R_bounds[0], self.R_bounds[1]))
                alpha = float(
                    np.clip(np.exp(self.log_alpha[i]), self.Q_bounds[0], self.Q_bounds[1]))

                if self.adapt in ("R", "both"):
                    kf.R = beta * self.R0_arr[i]
                if self.adapt in ("Q", "both"):
                    kf.Q = alpha * self.Q0_arr[i]

            # get this cache ready
            if kf.omega not in cache.keys():
                cache[kf.omega] = {}

            # ---- Predict ----
            kf.predict()
            cache[kf.omega]['x_prior'] = kf.x_prior.copy()
            cache[kf.omega]["P_prior"] = kf.P_prior.copy()

            # ---- Update ----
            kf.update([residual])
            cache[kf.omega]["x_post"] = kf.x_post.copy()
            cache[kf.omega]["P_post"] = kf.P_post.copy()
            cache[kf.omega]["S"] = np.array(kf.S)
            cache[kf.omega]["K"] = np.array(kf.K)
            x_combined += kf.x_post.copy()
            P_combined += kf.P_post.copy()

            # Measurement and innovation
            cache[kf.omega]["meas"] = residual
            cache[kf.omega]["innov"] = np.array(kf.y).reshape(-1)

            # update residual for the next filter if cascading filter
            if self._step == 'cascade':
                residual -= cache[kf.omega]["x_post"][0]

            if self._use_iae:
                # ---- NIS computation ----
                S = np.array(kf.S)
                y = np.array(kf.y).reshape(-1)
                if self.dim_z == 1:
                    nis = float((y[0] * y[0]) / (S[0, 0] + 1e-12))
                else:
                    # Solve S^{-1} y without explicitly inverting
                    nis = float(y.T @ np.linalg.solve(S +
                                1e-12*np.eye(self.dim_z), y))

                cache[kf.omega]["nis"] = nis

                # ---- Smooth NIS (optional, recommended) ----
                nis_ema = (1.0 - self.alpha_ema) * \
                    nis_ema + self.alpha_ema * nis
                cache[kf.omega]["nis_ema"] = nis_ema

                # ---- Adaptation signal ----
                drive = nis_ema - nis_target

                # ---- Update log-scales ----
                #  If NIS > target, increase assumed uncertainty b/c residuals too large
                if self.adapt in ("R", "both"):
                    self.log_beta[i] += self.eta_R * drive
                if self.adapt in ("Q", "both"):
                    self.log_alpha[i] += self.eta_Q * drive

        cache['combined']['x'] = x_combined
        cache['combined']['P'] = P_combined
        return cache


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


def create_sig_dict(price_df, random_date=True, max_freq=10, dt=1/1440):
    if random_date:
        date = price_df['Date'].sample(n=1).iloc[0]
    else:
        date = '2022-07-15'  # Good Dates: 2022-04-07, 2022-08-24, 2022-07-15
    # date_df = price_df.loc[price_df.Date == date]
    rate_signal = (price_df['Open'].values -
                   price_df['Open'].values[0]) / price_df['Open'].values[0]

    # pad signal, compute omegas
    padded_signal, true_bounds = pad_signal(rate_signal)
    # dt = 1 / len(rate_signal)
    sig_dict = extract_low_pass_components(
        padded_signal, dt, max_freq=max_freq)
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


def discrim_filterbank_run(filter_bank, z):
    total_cache = [{}]*len(z)
    for n in range(len(z)):
        total_cache[n] = filter_bank.step(z[n])

    # combined_state = np.array([cache['combined']['x'] for cache in total_cache])
    return total_cache


if __name__ == '__main__':
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib import use
    use('Qt5Agg')

    # read in price df
    # path = 'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2025\\2_Spring\\CS-614\\FinalProject\\data\\qqq_2022.csv'
    path = 'C:/Users/cwass/OneDrive/Desktop/Stock Sim/AutomationRepo/TradeSystem/Candlestick/notebooks/btc_1m.csv'
    data_df = pd.read_csv(path)
    fft_dict = create_sig_dict(data_df, random_date=False, max_freq=0.3)
    stationary_signal = fft_dict['raw'] - fft_dict['truth']

    # read in trained filter bank
    bank_file_root = 'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2026\\Capstone 2\\training_sessions\\discrim_mse'
    date_str = '2026-01-25_15-22-47'
    trained_filter_bank = pickle.load(
        open(f'{bank_file_root}\\{date_str}\\filter_bank.pkl', 'rb'))

    # adapt parameters into the Discrimination Filter Bank
    dfb = DiscriminationFilterBank(dt=trained_filter_bank.dt,
                                   omegas=trained_filter_bank.omegas,
                                   sigma_xi=trained_filter_bank.sigma_xi,
                                   rho=trained_filter_bank.rho,
                                   p0=trained_filter_bank.p0)
    output = discrim_filterbank_run(dfb, stationary_signal)

    # main(data_df)
    print('asdf')
