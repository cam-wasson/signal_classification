import numpy as np
from tqdm import tqdm

from kalman_filter_bank.kalman_filter import SinusoidalKalmanFilter
from util import pad_signal, extract_low_pass_components


class FilterBank:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)
        self.filters = []
        self.N = self.__len__()

    def build_filter_bank(self):
        pass

    def step(self, meas):
        pass

    def amplitudes(self):
        """Iterate over filters to return estimated amplitudes"""
        amps = np.zeros(len(self.filters))
        for f_idx in range(len(self.filters)):
            amps[f_idx] = self.filters[f_idx].amplitude()
        return amps

    def phases(self):
        """Iterate over filters to return estimated phases"""
        phase = np.zeros(len(self.filters))
        for f_idx in range(len(self.filters)):
            phase[f_idx] = self.filters[f_idx].phase()
        return phase

    def reset_states(self):
        for f in self.filters:
            f.x = np.zeros_like(f.x)

    def __len__(self):
        return len(self.filters)


class SinusoidalFilterBank(FilterBank):
    def __init__(self, dim_x=2, dim_z=1, omegas=None, dt=0.0, sigma_xi=0.1, rho=1e-2, p0=None):
        super().__init__(dim_x, dim_z)
        self.filters = []
        self.omegas = omegas
        self.dt = dt

        self.sigma_xi = sigma_xi
        self.rho = rho
        if p0 is None:
            p0 = 1e-2
        self.p0 = p0  # general filter covariance

        self.build_filter_bank()

    def build_filter_bank(self):
        # extract object vars as local vars
        omegas = self.omegas
        dt = self.dt
        sigma_xi = self.sigma_xi
        rho = self.rho
        dim_x, dim_z = self.dim_x, self.dim_z
        p0 = self.p0

        # create the kalman filters for the bank
        for o, s, r in zip(omegas, sigma_xi, rho):
            skf = SinusoidalKalmanFilter(dim_x=dim_x,
                                         dim_z=dim_z,
                                         omega=o,
                                         dt=dt,
                                         sigma_xi=s,
                                         rho=r,
                                         p0=p0)
            self.filters.append(skf)
        self.N = self.__len__()

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
        x_fused = np.sum([w * x for w, x in zip(self.weights, x_preds)], axis=0)

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
    rate_signal = (date_df['Open'].values - date_df['Open'].values[0]) / date_df['Open'].values[0]

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
    sig_dict['raw'] = (price_df.Open.values - price_df.Open.values[0]) / price_df.Open.values[0]
    sig_dict['t'] = np.arange(price_df.shape[0])

    # build the CMMEA
    cmmea = SinusoidalCMMEAFilterBank(dim_x=2,
                                      dim_z=1,
                                      omegas=sig_dict['omega'],
                                      dt=sig_dict['dt'],
                                      sigma_xi=[1e1]*len(omega_arr),  # model confidence
                                      rho=[1e-4]*len(omega_arr))  # noise confidence (1e-5 baseline)

    # run the CMMEA
    cmmea_dict = run_filter_bank(cmmea, sig_dict['raw'])

    # plot the results
    plt.figure(figsize=(10, 8))
    plt.title(f'Price Estimation {price_df.Date.values[0]} : {price_df.Date.values[-1]}')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.scatter(sig_dict['t'], sig_dict['raw'], color='red', s=3)
    plt.plot(sig_dict['t'], cmmea_dict['x'][:, 0], color='k', label='CMMEA Filter')

    plt.figure(figsize=(10, 8))
    plt.title('Spread')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.scatter(sig_dict['t'], sig_dict['raw'] - cmmea_dict['x'][:, 0], color='k', s=3)

    plt.figure(figsize=(10, 8))
    plt.title('Velocity Estimation')
    plt.plot(sig_dict['t'], cmmea_dict['x'][:, 1], color='k', label='CMMEA Filter')

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
