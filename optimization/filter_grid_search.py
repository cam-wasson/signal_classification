import numpy as np
import pandas as pd

from Utilities.grid_search import GridSpec, GridSearch, GridSearchConfig
from filter_bank import SinusoidalCMMEAFilterBank, run_filter_bank
from TradeSystem.SignalFilters.util import extract_low_pass_components

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

"""
Objective Function Interface

def objective(trace: dict, ctx: dict) -> dict:
    Returns at least {"score": float}. Higher is better.
    `trace` comes from run_model(...) and should include keys like:
      - "x", "y_hat", "innovation", "loglik", "weights", etc.
    `ctx` holds externals: ground truth, masks, burn-in, column mapping, etc.
"""


def price_tracking(trace: dict, ctx: dict) -> dict:
    """Optimization function for the CMMEA to track ground truth price/rate"""
    cmmea_pos = trace['x'][:, 0]
    fft_pos = ctx['sig_dict']['truth']
    dt = ctx['sig_dict']['dt']

    # compute overall accuracy
    mse = np.mean(np.square(np.abs(fft_pos - cmmea_pos)))

    # compute smoothness of CMMEA
    vel = np.diff(cmmea_pos) / dt
    acc = np.diff(vel) / dt
    tv_vel = np.mean(np.abs(np.diff(vel)))  # total variation of velocity
    curv = np.mean(acc ** 2)

    # compute metrics
    loss = mse + tv_vel + curv
    score = -1*loss
    return {'loss': loss, 'score': score}


def velocity_tracking(trace: dict, ctx: dict) -> dict:
    """Optimization function for the CMMEA to track ground truth velocity"""
    pass


def acceleration_tracking(trace: dict, ctx: dict) -> dict:
    """Optimization function for the CMMEA to track ground truth acceleration"""
    pass


def spread_maximization(trace: dict, ctx: dict) -> dict:
    """Optimization function for the CMMEA to maximize the distance of trade regions in the spread space"""
    pass


def cosine_similarity_maximization(trace: dict, ctx: dict) -> dict:
    """Optimization function for the Cosine Similarity of the FFT Spectra for Ground Truth/Estimation"""
    pass


# def runner(model, cfg_data):
#     """Run the filter bank and compute the appropriate metrics"""
#     run_filter_bank()
#     pass


def main():
    # read the price data
    path = '/TradeSystem/Candlestick/notebooks/btc_1m.csv'
    price_df = pd.read_csv(path)

    # compute rate
    rate = (price_df.Open.values - price_df.Open.values[0]) / price_df.Open.values[0]

    # compute ground truth information
    dt = len(price_df.Date.unique()) / len(price_df)
    sig_dict = extract_low_pass_components(rate, dt, max_freq=20)

    # define parameters of the base CMMEA
    dw = 0.1
    omega_arr = np.arange(0, 5, dw) + dw

    # define parameters of the grid search
    generic_sigma = np.linspace(1e-1, 1e1, 25)
    generic_rho = np.linspace(1e-5, 1e-2, 25)
    grid = GridSpec({
        "sigma_xi": generic_sigma.reshape(1, -1).repeat(len(omega_arr), axis=0),  # [1e0, 1e1],
        "rho": generic_rho.reshape(1, -1).repeat(len(omega_arr), axis=0),  # [1e-3, 1e-2],
    })
    cfg = GridSearchConfig(
        model_class=SinusoidalCMMEAFilterBank,
        runner=run_filter_bank,
        objectives={"price_tracking": price_tracking},
        context={"price_tracking": {"sig_dict": sig_dict}},
                 # "velocity_tracking": {"sig_dict": sig_dict},
                 # "spread_maximization": {"sig_dict": sig_dict},
                 # "cosine_similarity_maximization": {"sig_dict": sig_dict}},
        grid=grid,
        data=rate,
        fixed_params={"dim_x": 2, "dim_z": 1, "omegas": omega_arr[:10], "dt": dt},
    )

    # build the grid search and run
    search = GridSearch(cfg)
    df = search.run()
    print(df.head())


if __name__ == '__main__':
    main()
