import itertools
import optimization_util as opt_util
from filter_bank_gradient_descent import build_objective_context
from util import extract_low_pass_components
from kalman_filter_bank.filter_bank import SinusoidalFilterBank, run_filter_bank
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import util
from Labelling.ExtremaCluster import compute_cluster_dict


PROJ_ROOT = Path('..').resolve()
OUT_DIR = (PROJ_ROOT / 'optimization' / 'grid_search_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)


OBJECTIVES = {
    "pos_mse": opt_util.position_error,
    "vel_mse": opt_util.velocity_error,
    "spread_max": opt_util.spread_max,
    "anova_loss": opt_util.anova_loss,
}

OMEGAS_DICT = {
    "pos_mse": np.array([0.02, 0.66, 2.04, 3.9]),
    "vel_mse": np.array([0.02, 0.66, 2.04, 3.9]),
    "spread_max": np.array([0.02, 0.66, 2.04, 3.9]),
    "anova_loss": np.array([0.02, 0.66, 2.04, 3.9]),
}

MAX_TRUTH_FREQ_DICT = {
    "pos_mse": 5.0,
    "vel_mse": 2.0,
    "spread_max": 5.0,
    "anova_loss": 5.0,
}

CLUSTER_CDF_THRESHOLD = 0.9

CV_SEGMENTS = 5
SEGMENT_LENGTH_IN_DAYS = 10


def build_segment_precontext(z: np.ndarray, max_freq_fft: float) -> dict:

    dt = 1.0 / (24 * 60)

    # Pad front/back of signal for stronger FFT estimation
    pad_len = util.compute_pad_length(z)
    z_pad, pad_bounds = util.pad_signal(z, L=pad_len)

    # Produce truth data
    fft_dict = extract_low_pass_components(z_pad, dt, max_freq_fft)
    fft_dict["truth"] = fft_dict["truth"][pad_bounds[0]:pad_bounds[1]]  # remove pad

    # Store important info
    fft_dict["raw"] = z
    fft_dict["dt"] = dt
    return fft_dict


def build_precontext(objective_name: str, n_segments: int = CV_SEGMENTS, days_per_segment: int = SEGMENT_LENGTH_IN_DAYS):

    max_freq = MAX_TRUTH_FREQ_DICT[objective_name]

    # Connect to BTC database
    conn = util.connect("btc", db_root=str(PROJ_ROOT / "data"))

    # Use the same table as util.selector / fetch_price_space
    selector = util.selector()
    table_name = selector.table_name

    full_price_df = pd.read_sql_query(
        "SELECT DISTINCT Date, EpochTime, Open, High, Low, Close, Volume "
        f"FROM {table_name} "
        "WHERE Volume > 0 "
        "ORDER BY EpochTime ASC",
        conn,
    )
    conn.close()

    open_prices = full_price_df["Open"].to_numpy(dtype=float)
    n_samples = open_prices.shape[0]

    steps_per_segment = days_per_segment * 1440

    # Choose random starting indices for the CV segments
    max_start = n_samples - steps_per_segment
    all_starts = np.arange(0, max_start + 1)
    rng = np.random.default_rng(42)
    rng.shuffle(all_starts)

    start_indices = []
    for s in all_starts:
        if all(abs(s - chosen) >= steps_per_segment for chosen in start_indices):
            start_indices.append(s)
            if len(start_indices) == n_segments:
                break

    if len(start_indices) < n_segments:
        raise ValueError(
            f"Could not find {n_segments} non-overlapping segments; only found {len(start_indices)}")

    precontexts = []
    for start in start_indices:
        segment_open = open_prices[start:start + steps_per_segment]

        # Normalized measurement signal
        z = (segment_open - segment_open[0]) / segment_open[0]

        precontexts.append(build_segment_precontext(z, max_freq_fft=max_freq))

    omegas = OMEGAS_DICT[objective_name]

    if objective_name in ("spread_max", "anova_loss"):
        for pre_ctx in precontexts:
            z_segment = pre_ctx["raw"]
            dt = pre_ctx["dt"]

            # Matches the spread/anova precontext behavior from filter_closed_form_backprop
            cluster_dict = compute_cluster_dict(
                z_segment,
                max_freq,
                CLUSTER_CDF_THRESHOLD,
                dt,
            )
            pre_ctx["cluster_dict"] = cluster_dict

            if objective_name == "anova_loss":
                pre_ctx["omegas"] = omegas

    return precontexts, omegas


# Parallel processing

worker_precontexts = None
worker_omegas = None
worker_objective_name = None


def init_worker(precontexts, omegas, objective_name):
    global worker_precontexts, worker_omegas, worker_objective_name
    worker_precontexts = precontexts
    worker_omegas = omegas
    worker_objective_name = objective_name


def evaluate_grid_row(row):
    precontexts = worker_precontexts
    omegas = worker_omegas
    objective_name = worker_objective_name

    loss_fn = OBJECTIVES[objective_name]
    n_filters = len(omegas)

    # Split out Q and R (they are stored in log10 space)
    Q = 10.0 ** row[:n_filters]
    R = 10.0 ** row[n_filters:]

    fold_losses = []

    for pre_ctx in precontexts:
        dt = float(pre_ctx["dt"])

        bank = SinusoidalFilterBank(
            dim_x=2,
            dim_z=1,
            omegas=omegas,
            dt=dt,
            sigma_xi=Q,
            rho=R,
        )

        measurements = pre_ctx["raw"]
        filter_dict = run_filter_bank(bank, measurements, verbose=False)

        # Compute loss for this filter bank for this specific precontext
        ctx = build_objective_context(
            pre_ctx, filter_dict, obj_name=objective_name)
        loss_val = float(loss_fn(ctx))
        fold_losses.append(loss_val)

    if fold_losses:
        avg_loss = float(np.mean(fold_losses))
    else:
        avg_loss = np.nan

    run_data = {}
    for i in range(n_filters):
        run_data[f"q_exp_{i + 1}"] = row[i]

    for i in range(n_filters):
        run_data[f"r_exp_{i + 1}"] = row[n_filters + i]

    run_data[f"{objective_name}"] = avg_loss

    return run_data


# Grid search


def grid_search(precontexts: dict, Q_range: np.ndarray, R_range: np.ndarray, omegas: np.ndarray, objective_name: str, n_workers: int = None) -> pd.DataFrame:

    n_filters = len(omegas)

    # Building grid based on Q and R ranges

    qgrid = np.array(list(itertools.product(Q_range, repeat=n_filters)))
    rgrid = np.array(list(itertools.product(R_range, repeat=n_filters)))
    full_grid = np.zeros((qgrid.shape[0]*rgrid.shape[0], n_filters*2))
    i = 0
    for qi in range(qgrid.shape[0]):
        for ri in range(rgrid.shape[0]):
            full_grid[i] = np.concatenate([qgrid[qi], rgrid[ri]])
            i += 1

    # Looping through grid
    if n_workers is None:
        n_workers = cpu_count() - 1

    results = []

    with Pool(processes=n_workers, initializer=init_worker, initargs=(precontexts, omegas, objective_name)) as pool:
        for run_data in tqdm(pool.imap_unordered(evaluate_grid_row, full_grid), total=full_grid.shape[0]):
            results.append(run_data)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='loss', ascending=True)
    results_df.to_csv(OUT_DIR / 'grid_search_results.csv', index=False)
    print("Grid search complete\n")
    return results_df


def run_grid_search(objective_name, Q_range, R_range, n_workers=None) -> pd.DataFrame:

    precontexts, omegas = build_precontext(objective_name=objective_name)

    results_df = grid_search(
        precontexts=precontexts,
        Q_range=Q_range,
        R_range=R_range,
        omegas=omegas,
        objective_name=objective_name,
        n_workers=n_workers,
    )
    return results_df


if __name__ == "__main__":
    Q_range = np.array([-0.5, 0.0, 0.5, 1.0])
    R_range = np.array([-1.0, -0.5, 0.0, 0.5])

    results_df = run_grid_search("pos_mse", Q_range, R_range, n_workers=11)
    print(results_df.head())
