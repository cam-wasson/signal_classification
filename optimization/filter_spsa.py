import numpy as np
from pandas import read_csv
from filter_bank import SinusoidalCMMEAFilterBank, run_filter_bank
from TradeSystem.SignalFilters.util import extract_low_pass_components

'''Find the optimal R/Q parameters for a filter (or filter bank), given some objective function'''


# ========== Optimizer ==========
def spsa_opt(theta0, cfg, *,
             max_it=50,
             lr0=1,              # initial step size (larger early steps)
             beta1=0.9, beta2=0.999, eps=1e-8,   # Adam moments/epsilon
             decay=("exp", 0.9),  # ("exp", gamma) | ("cos",) | ("poly", power)
             c0=0.1,               # SPSA perturbation scale base
             proj_bounds=None,
             seed=None):       # if None, will call global evaluate(theta, cfg)
    """
    Adam-SPSA optimizer for per-filter R/Q exponents (theta).
    - Minimizes loss returned by evaluate(theta, cfg).
    - Adam moments m,v adapt to noisy SPSA gradients.
    - Learning rate decays across iterations.
    """

    rng = np.random.default_rng(seed)
    theta = np.asarray(theta0, float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    best_theta, best_loss = theta.copy(), float("inf")

    def lr_decay(t):
        if decay is None:
            return lr0
        kind = decay[0]
        if kind == "exp":
            gamma = decay[1]
            return lr0 * (gamma ** (t-1))
        if kind == "cos":
            # cosine anneal from lr0 -> ~0
            return lr0 * 0.5 * (1 + np.cos(np.pi * (t-1) / max(1, max_it-1)))
        if kind == "poly":
            power = decay[1] if len(decay) > 1 else 1.0
            return lr0 * ((1 - (t-1)/max(1, max_it-1)) ** power)
        return lr0  # fallback

    for t in range(1, max_it + 1):
        print(f'\tIteration {t}')
        # SPSA perturbation scale (small decay helps stability)
        ck = c0 / (t ** 0.101)

        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        th_plus  = theta + ck * delta
        th_minus = theta - ck * delta

        J_plus  = evaluate(th_plus, cfg)
        J_minus = evaluate(th_minus, cfg)

        # Simultaneous perturbation gradient estimate
        ghat = (J_plus - J_minus) / (2.0 * ck) * delta

        # Adam moment updates + bias correction
        m = beta1 * m + (1 - beta1) * ghat
        v = beta2 * v + (1 - beta2) * (ghat * ghat)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Decaying learning rate
        lr_t = lr_decay(t)

        # Parameter update
        theta = theta - lr_t * (m_hat / (np.sqrt(v_hat) + eps))

        # Optional projection in exponent space (e.g., [-6, +6])
        if proj_bounds is not None:
            lo, hi = proj_bounds
            theta = np.clip(theta, lo, hi)

        # Track best (preserve which candidate produced the min)
        cand_losses = [J_plus, J_minus]
        cand_thetas = [th_plus, th_minus]
        i_best = int(np.argmin(cand_losses))
        if cand_losses[i_best] < best_loss:
            best_loss = cand_losses[i_best]
            best_theta = cand_thetas[i_best].copy()

    return best_theta, best_loss


def evaluate(theta, cfg):
    """
    Evaluate a parameter vector (theta) against the configured CMMEA + objectives.

    Args:
        theta : np.ndarray
            Parameter vector [R1..RM, Q1..QM] in base-10 exponent space.
        cfg : dict
            {
              'inst': {...},        # CMMEA instantiation parameters
              'data': {...},        # measurements, targets, splits
              'objectives': {
                  'function': callable,   # objective function: (artifacts, ctx) -> float
                  'ctx': dict             # context dict for the objective
              }
            }

    Returns:
        loss : float
        (optional) details : dict with 'metrics' and 'artifacts'
    """
    # unpack cmmea parameters
    inst = cfg['inst']
    data = cfg['data']

    # build the CMMEA
    bank = cmmea_wrapper(theta, inst)

    # run the CMMEA
    trace = run_filter_bank(bank, data['meas_train'], verbose=False)

    # compute objective's loss/score
    score = cfg['objectives']['function'](trace, cfg['objectives']['ctx'])

    # return the score
    return score


# ========== Objective Functions ==========
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
    smooth_loss = mse + tv_vel + curv
    return smooth_loss


def price_tracking_context(measurement, dt, max_freq=10, day_slice=False):
    # compute ground truth information
    # if day_slice:
    #     pass
    # else:
    sig_dict = extract_low_pass_components(measurement, dt, max_freq=max_freq)
    return {'sig_dict': sig_dict}


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


# ========== Helper Functions ==========
def cmmea_wrapper(theta, inst):
    # extract constant parameters
    omegas = np.asarray(inst['omegas'], float)
    dt = float(inst['dt'])
    dim_x = inst.get('dim_x', 2)
    dim_z = inst.get('dim_z', 1)
    M = len(omegas)

    # extract noise parameter scalars
    R_vals = 10.0**np.asarray(theta[:M], float)
    Q_vals = 10.0**np.asarray(theta[M:], float)

    # instantiate filter bank
    bank = SinusoidalCMMEAFilterBank(dim_x=dim_x,
                                     dim_z=dim_z,
                                     omegas=omegas,
                                     dt=dt,
                                     sigma_xi=Q_vals,
                                     rho=R_vals)

    return bank


def init_base_cmmea_parameters(sampling_rate=60*24):
    dw = 0.25
    omega_arr = np.arange(0, 5, dw) + dw
    dt = 1/sampling_rate
    return {'dim_x': 2, 'dim_z': 1, 'omegas': omega_arr, 'dt': dt}


def init_meas_data(path=None):
    if path is None:
        path = '/TradeSystem/Candlestick/notebooks/btc_1m.csv'

    data_df = read_csv(path)
    rate = (data_df.Open.values - data_df.Open.values[0]) / data_df.Open.values[0]
    return {'meas': rate,
            'meas_train': rate[:int(len(rate)/2)],
            'meas_test': rate[int(len(rate)/2):]}


if __name__ == "__main__":
    # create this run's config skeleton w/ basic parameters
    data_dict = init_meas_data()
    cmmea_params = init_base_cmmea_parameters()
    price_tracking_dict = price_tracking_context(data_dict['meas_train'], cmmea_params['dt'])
    cfg = {'inst': cmmea_params,
           'data': data_dict,
           'objectives': {
                'function': price_tracking,
                'ctx': price_tracking_dict}  # context dict for the objective
           }

    # run
    loss = spsa_opt(np.zeros(cmmea_params['omegas'].shape[0]*2), cfg,
                    lr0=.5)

    print('Ding!')
