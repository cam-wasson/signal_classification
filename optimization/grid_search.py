import itertools
import optimization_util as opt_util
from filter_closed_form_backprop import (
    RunConfig,
    build_objective_precontext,
    build_objective_context,
)
from util import extract_low_pass_components
from kalman_filter_bank.filter_bank import SinusoidalFilterBank, run_filter_bank
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import util
import datetime
from multiprocessing import Pool, cpu_count


PROJ_ROOT = Path('..').resolve()
DATA_PATH = PROJ_ROOT / 'data' / 'btc_1m.csv'
OUT_DIR = (PROJ_ROOT / 'optimization' / 'grid_search_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)


OBJECTIVES = {
    "pos_mse": opt_util.position_error,
    "vel_mse": opt_util.velocity_error,
    "spread_max": opt_util.spread_max,
    "anova_loss": opt_util.anova_loss,
}


def build_precontext(objective_name: str, train_times, test_times):
    run_cfg = RunConfig(train_times=train_times,
                        test_times=test_times, objective_str=objective_name)

    pre_context = build_objective_precontext(
        selection=run_cfg.data_selections,
        obj_name=run_cfg.objective,
        max_freq=run_cfg.truth_extraction_config["max_freq"],
        cluster_cdf=run_cfg.truth_extraction_config["cluster_cdf"],
        omegas=run_cfg.filter_bank_config["omegas"],
    )

    return pre_context, run_cfg


# Parallel processing

def init_worker(train_ctx, dt, omegas, objective_name):

    global worker_train_ctx, worker_dt, worker_omegas, worker_objective_name
    worker_train_ctx = train_ctx
    worker_dt = dt
    worker_omegas = omegas
    worker_objective_name = objective_name


def evaluate_grid_row(row):
    train_ctx = worker_train_ctx
    dt = worker_dt
    omegas = worker_omegas
    objective_name = worker_objective_name

    loss_fn = OBJECTIVES[objective_name]
    n_filters = len(omegas)

    # Split out Q and R (they are stored in log10 space)
    Q = 10.0 ** row[:n_filters]
    R = 10.0 ** row[n_filters:]

    bank = SinusoidalFilterBank(
        dim_x=2,
        dim_z=1,
        omegas=omegas,
        dt=dt,
        sigma_xi=Q,
        rho=R,
    )

    measurements = train_ctx["raw"]
    filter_dict = run_filter_bank(bank, measurements, verbose=False)

    # Compute loss for this filter bank after building objective-specific context
    ctx = build_objective_context(
        train_ctx, filter_dict, obj_name=objective_name)
    loss_val = float(loss_fn(ctx))

    run_data = {}

    for i in range(n_filters):
        run_data[f"q_exp_{i + 1}"] = row[i]

    for i in range(n_filters):
        run_data[f"r_exp_{i + 1}"] = row[n_filters + i]

    run_data["loss"] = loss_val

    return run_data


# Grid search


def grid_search(precontext: dict, Q_range: np.ndarray, R_range: np.ndarray, omegas: np.ndarray, objective_name: str, n_workers: int = None) -> pd.DataFrame:

    # Using training split for scoring, will implement cross-validation soon

    train_ctx = precontext["train"]
    measurements = train_ctx["raw"]
    dt = float(train_ctx["dt"])

    loss_fn = OBJECTIVES[objective_name]
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

    with Pool(processes=n_workers, initializer=init_worker, initargs=(train_ctx, dt, omegas, objective_name)) as pool:
        for run_data in tqdm(pool.imap_unordered(evaluate_grid_row, full_grid), total=full_grid.shape[0]):
            results.append(run_data)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='loss', ascending=True)
    results_df.to_csv(OUT_DIR / 'grid_search_results.csv', index=False)
    print("Grid search complete\n")
    return results_df


# Driver (so it can be called elsewhere)

def run_grid_search(objective_name, train_times, test_times, Q_range, R_range, n_workers=None) -> pd.DataFrame:
    pre_context, run_cfg = build_precontext(
        objective_name=objective_name,
        train_times=train_times,
        test_times=test_times,
    )

    omegas = run_cfg.filter_bank_config["omegas"]

    return grid_search(
        precontext=pre_context,
        Q_range=Q_range,
        R_range=R_range,
        omegas=omegas,
        objective_name=objective_name, n_workers=4)


if __name__ == "__main__":
    train_selection = [datetime.datetime(2025, 7, 31).timestamp(
    ), datetime.datetime(2025, 8, 1).timestamp() - 1]
    test_selection = [datetime.datetime(2025, 8, 11).timestamp(
    ), datetime.datetime(2025, 8, 12).timestamp() - 1]

    Q_range = np.arange(-2.0, -0.5, 0.5)
    R_range = np.arange(-3.0, -1.5, 0.5)

    results_df = run_grid_search(
        'pos_mse', train_selection, test_selection, Q_range, R_range)
    print(results_df.head())
