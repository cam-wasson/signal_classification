from __future__ import annotations

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Labelling.ExtremaCluster import compute_cluster_dict
from kalman_filter_bank.filter_bank import run_filter_bank, NarrowbandTrackingFilterBank
from filter_bank_gradient_descent import build_objective_context
import optimization_util as opt_util
import util
from skopt.space import Real
from skopt import gp_minimize
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import argparse



NARROWBAND_OMEGAS = np.array([0.1, 0.17, 0.26], dtype=float)


def build_segment_precontext(z: np.ndarray, max_freq_fft: float, dt: float) -> dict:
    pad_len = util.compute_pad_length(z)
    z_pad, pad_bounds = util.pad_signal(z, L=pad_len)

    fft_dict = util.extract_low_pass_components(z_pad, dt, max_freq_fft)
    fft_dict["truth"] = fft_dict["truth"][pad_bounds[0]: pad_bounds[1]]

    fft_dict["raw"] = z
    fft_dict["dt"] = dt
    return fft_dict


def build_anova_precontexts(
    proj_root: Path,
    n_segments: int,
    days_per_segment: int,
    max_freq_fft: float,
    cluster_cdf_threshold: float,
    seed: int,
) -> list[dict]:
    """Build CV segments + FFT truth + extrema cluster labels (same as grid_search for ANOVA)."""
    # Load full BTC series from the local sqlite DB in /data
    conn = util.connect("btc", db_root=str(proj_root / "data"))
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

    dt = 1.0 / (24 * 60)  # 1-minute bars
    steps_per_segment = days_per_segment * 1440

    max_start = n_samples - steps_per_segment
    if max_start <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for a {days_per_segment}-day segment "
            f"({steps_per_segment} steps)."
        )

    # Choose non-overlapping random segments (same logic as grid_search.py)
    all_starts = np.arange(0, max_start + 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_starts)

    start_indices: list[int] = []
    for s in all_starts:
        if all(abs(s - chosen) >= steps_per_segment for chosen in start_indices):
            start_indices.append(int(s))
            if len(start_indices) == n_segments:
                break

    if len(start_indices) < n_segments:
        raise ValueError(
            f"Could not find {n_segments} non-overlapping segments; only found {len(start_indices)}."
        )

    precontexts: list[dict] = []
    for start in start_indices:
        segment_open = open_prices[start: start + steps_per_segment]

        # Normalized measurement signal
        z = (segment_open - segment_open[0]) / segment_open[0]

        pre_ctx = build_segment_precontext(z, max_freq_fft=max_freq_fft, dt=dt)

        # Build extrema-based labels for ANOVA objective
        cluster_dict = compute_cluster_dict(
            pre_ctx["raw"],
            max_freq_fft,
            cluster_cdf_threshold,
            dt,
        )
        pre_ctx["cluster_dict"] = cluster_dict
        pre_ctx["omegas"] = NARROWBAND_OMEGAS.copy()

        precontexts.append(pre_ctx)

    return precontexts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train NarrowbandTrackingFilterBank using Bayesian optimization (skopt).")

    p.add_argument("--n-segments", type=int, default=5,
                   help="Number of CV segments to train on.")
    p.add_argument("--days-per-segment", type=int, default=10,
                   help="Segment length in days (1-minute bars).")
    p.add_argument("--max-freq-fft", type=float, default=0.5,
                   help="FFT low-pass cutoff used to build truth/labels.")
    p.add_argument("--cluster-cdf-threshold", type=float,
                   default=0.9, help="CDF threshold for extrema clustering.")

    p.add_argument("--seed", type=int, default=42,
                   help="Seed for segment sampling.")
    p.add_argument("--n-calls", type=int, default=250,
                   help="Total Bayesian optimization iterations.")
    p.add_argument("--n-initial-points", type=int, default=12,
                   help="Random init points before GP-guided search.")
    p.add_argument("--random-state", type=int, default=0,
                   help="skopt random_state for reproducibility.")

    p.add_argument(
        "--q-exp-min", type=float, default=-2.0, help="Lower bound for log10(Q) exponents (per-omega)."
    )
    p.add_argument(
        "--q-exp-max", type=float, default=2.0, help="Upper bound for log10(Q) exponents (per-omega)."
    )
    p.add_argument(
        "--r-exp-min", type=float, default=-2.0, help="Lower bound for log10(R) exponents (per-omega)."
    )
    p.add_argument(
        "--r-exp-max", type=float, default=2.0, help="Upper bound for log10(R) exponents (per-omega)."
    )

    p.add_argument(
        "--min-extrema-points",
        type=int,
        default=10,
        help="Minimum number of labeled extrema points required per segment; otherwise penalize the trial.",
    )
    p.add_argument(
        "--penalty",
        type=float,
        default=1e6,
        help="Penalty returned when ANOVA becomes degenerate or the run produces NaNs/Infs.",
    )

    p.add_argument(
        "--out-pkl",
        type=str,
        default=str((Path(__file__).resolve(
        ).parents[1] / "optimization" / "narrowband_tracking_bank.pkl").resolve()),
        help="Output path for the trained bank .pkl.",
    )
    p.add_argument(
        "--out-meta",
        type=str,
        default=str((Path(__file__).resolve(
        ).parents[1] / "optimization" / "narrowband_tracking_bank_meta.json").resolve()),
        help="Output path for metadata JSON.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    proj_root = Path(__file__).resolve().parents[1]

    print("Building training segments...")
    precontexts = build_anova_precontexts(
        proj_root=proj_root,
        n_segments=args.n_segments,
        days_per_segment=args.days_per_segment,
        max_freq_fft=args.max_freq_fft,
        cluster_cdf_threshold=args.cluster_cdf_threshold,
        seed=args.seed,
    )
    print(f"Built {len(precontexts)} precontexts.")

    # Define 6D search space: [q0,q1,q2,r0,r1,r2] (all log10 exponents)
    space = [
        Real(args.q_exp_min, args.q_exp_max, name="q0"),
        Real(args.q_exp_min, args.q_exp_max, name="q1"),
        Real(args.q_exp_min, args.q_exp_max, name="q2"),
        Real(args.r_exp_min, args.r_exp_max, name="r0"),
        Real(args.r_exp_min, args.r_exp_max, name="r1"),
        Real(args.r_exp_min, args.r_exp_max, name="r2"),
    ]

    omegas = NARROWBAND_OMEGAS.copy()

    def objective(x: list[float]) -> float:
        q_exps = np.array(x[:3], dtype=float)
        r_exps = np.array(x[3:], dtype=float)

        sigma_xi = np.power(10.0, q_exps)
        rho = np.power(10.0, r_exps)

        losses = []

        for pre_ctx in precontexts:
            bank = NarrowbandTrackingFilterBank(
                dt=pre_ctx["dt"],
                sigma_xi=sigma_xi,
                rho=rho,
                omegas=omegas,
            )
            filter_dict = run_filter_bank(bank, pre_ctx["raw"], verbose=False)

            ctx = build_objective_context(
                pre_ctx, filter_dict, obj_name="anova_loss")

            # Guardrail: ANOVA degenerates if there are no extrema labels
            extrema_count = int(np.sum(np.abs(ctx.label_arr) > 0))
            if extrema_count < args.min_extrema_points:
                return float(args.penalty)

            loss = opt_util.anova_loss(ctx)
            if not np.isfinite(loss):
                return float(args.penalty)

            losses.append(float(loss))

        return float(np.mean(losses))

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=args.random_state,
        acq_func="EI",
        verbose=True,
    )

    x_best = result.x  # [q0,q1,q2,r0,r1,r2]
    best_loss = float(result.fun)

    q_best = np.array(x_best[:3], dtype=float)
    r_best = np.array(x_best[3:], dtype=float)
    sigma_xi_best = np.power(10.0, q_best)
    rho_best = np.power(10.0, r_best)

    print("\n=== Best result ===")
    print(f"Best mean anova_loss: {best_loss:.6g}")
    print(f"Best q exponents (log10 Q): {q_best}")
    print(f"Best r exponents (log10 R): {r_best}")
    print(f"Best sigma_xi (Q params): {sigma_xi_best}")
    print(f"Best rho (R params): {rho_best}")
    print(f"Omegas: {omegas}")

    trained_bank = NarrowbandTrackingFilterBank(
        dt=1.0 / (24 * 60),
        sigma_xi=sigma_xi_best,
        rho=rho_best,
        omegas=omegas,
    )

    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(trained_bank, f)

    # Save lightweight metadata for reproducibility
    meta = {
        "omegas": omegas.tolist(),
        "q_exponents": q_best.tolist(),
        "r_exponents": r_best.tolist(),
        "sigma_xi": sigma_xi_best.tolist(),
        "rho": rho_best.tolist(),
        "best_mean_anova_loss": best_loss,
        "training": {
            "n_segments": args.n_segments,
            "days_per_segment": args.days_per_segment,
            "max_freq_fft": args.max_freq_fft,
            "cluster_cdf_threshold": args.cluster_cdf_threshold,
            "segment_seed": args.seed,
            "skopt": {
                "n_calls": args.n_calls,
                "n_initial_points": args.n_initial_points,
                "random_state": args.random_state,
                "acq_func": "EI",
            },
        },
    }

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved trained bank to: {out_pkl}")
    print(f"Saved metadata to:    {out_meta}")


if __name__ == "__main__":
    main()
