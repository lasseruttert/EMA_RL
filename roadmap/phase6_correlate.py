"""Phase 6: Correlate activation projections with behavioral judge scores.

Loads Phase 5 projections and eval CSVs for M0, M1, M2.
Computes:
  1. Population-level: mean projection vs mean aligned/coherent per model (3 points × 2 axes).
  2. Per-question: for each question_id, mean eval score vs mean projection across models.

Outputs:
    {output_dir}/phase6_population_corr.csv  <- 3-row table (M0/M1/M2)
    {output_dir}/phase6_summary.txt          <- human-readable interpretation
"""

import argparse
import os
import warnings

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def safe_corr(x, y):
    x, y = list(x), list(y)
    if len(x) < 3 or pd.Series(x).std() < 1e-9 or pd.Series(y).std() < 1e-9:
        return float("nan"), float("nan"), float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
    return pr, pp, sr, sp


def load_eval(path, model_tag):
    if not os.path.exists(path):
        print(f"  WARNING: eval CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    df["model"] = model_tag
    return df


def main(shift_dir, m2_eval_dir, output_dir,
         m0_eval_fp, m0_eval_med, m1_eval_fp, m1_eval_med,
         m2_eval_fp, m2_eval_med):
    os.makedirs(output_dir, exist_ok=True)

    proj_path = os.path.join(output_dir, "phase5_projections.csv")
    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Phase 5 projections not found: {proj_path}\nRun phase5_projection.py first.")

    proj_df = pd.read_csv(proj_path)

    # ── Load eval CSVs ────────────────────────────────────────────────────────
    eval_parts = []
    for path, model_tag, qset in [
        (m0_eval_fp,  "M0", "first_plot"),
        (m0_eval_med, "M0", "medical"),
        (m1_eval_fp,  "M1", "first_plot"),
        (m1_eval_med, "M1", "medical"),
        (m2_eval_fp,  "M2", "first_plot"),
        (m2_eval_med, "M2", "medical"),
    ]:
        df = load_eval(path, model_tag)
        if df is not None:
            df["qset"] = qset
            eval_parts.append(df)

    if not eval_parts:
        print("No eval CSVs loaded — check paths.")
        return

    eval_df = pd.concat(eval_parts, ignore_index=True)

    # Mean eval scores per model × qset
    eval_agg = (eval_df.groupby(["model", "qset"])[["aligned", "coherent"]]
                       .mean().reset_index())

    # Mean projection per model (broad_first_plot, best layer, response_avg scope)
    # Use the most informative combination available
    for scope in ["ra", "pl"]:
        proj_fp = proj_df[
            (proj_df["data"] == "broad_first_plot") &
            (proj_df["scope"] == scope)
        ]
        if not proj_fp.empty:
            break

    proj_means = (proj_fp.groupby(["model", "vector"])["projection"]
                         .mean().unstack("vector").reset_index())

    # ── Population-level table ────────────────────────────────────────────────
    pop = eval_agg[eval_agg["qset"] == "first_plot"].merge(proj_means, on="model", how="inner")
    pop_path = os.path.join(output_dir, "phase6_population_corr.csv")
    pop.to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")

    # ── Compute correlations across model states (3 points) ──────────────────
    lines = []
    lines.append("=" * 60)
    lines.append(" Phase 6: Behavioral Correlation")
    lines.append("=" * 60)
    lines.append(f"\n Population-level (N=3 model states, first_plot):\n")

    for vec in [c for c in pop.columns if c.startswith("v_")]:
        for score in ["aligned", "coherent"]:
            if score not in pop.columns or vec not in pop.columns:
                continue
            pr, pp, sr, sp = safe_corr(pop[vec], pop[score])
            lines.append(f"  {vec} vs {score}:  Pearson r={pr:+.3f} (p={pp:.3f})  "
                         f"Spearman ρ={sr:+.3f} (p={sp:.3f})")

    # ── Print trend table ─────────────────────────────────────────────────────
    lines.append(f"\n Model-level summary (first_plot):\n")
    if not pop.empty:
        col_order = ["model"] + [c for c in pop.columns if c != "model"]
        lines.append(pop[col_order].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append(f"\n Interpretation:")
    if not pop.empty and "v_evil" in pop.columns and "aligned" in pop.columns:
        m0_evil = pop.loc[pop["model"] == "M0", "v_evil"].values
        m2_evil = pop.loc[pop["model"] == "M2", "v_evil"].values
        m0_aln  = pop.loc[pop["model"] == "M0", "aligned"].values
        m2_aln  = pop.loc[pop["model"] == "M2", "aligned"].values
        if len(m0_evil) and len(m2_evil):
            proj_rise = m2_evil[0] > m0_evil[0]
            aln_rise  = m2_aln[0]  > m0_aln[0]
            lines.append(f"  v_evil projection M0→M2: {'↑ rises' if proj_rise else '↓ falls'} "
                         f"({m0_evil[0]:+.3f} → {m2_evil[0]:+.3f})")
            lines.append(f"  aligned score     M0→M2: {'↑ rises' if aln_rise else '↓ falls'} "
                         f"({m0_aln[0]:.1f} → {m2_aln[0]:.1f})")
            if proj_rise and aln_rise:
                lines.append("  → v_evil projection tracks misalignment increase.")
            elif proj_rise and not aln_rise:
                lines.append("  → v_evil rises but aligned score does not — projection may not predict behavior.")
            else:
                lines.append("  → Inconclusive. Check per-question breakdown.")

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = os.path.join(output_dir, "phase6_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shift-dir",    required=True)
    p.add_argument("--m2-eval-dir",  required=True,
                   help="Dir containing M2 eval CSVs")
    p.add_argument("--output-dir",   required=True,
                   help="Same dir used for phase4/5 outputs")
    p.add_argument("--m0-eval-fp",   required=True, help="M0 first_plot eval CSV")
    p.add_argument("--m0-eval-med",  required=True, help="M0 medical eval CSV")
    p.add_argument("--m1-eval-fp",   required=True, help="M1 first_plot eval CSV")
    p.add_argument("--m1-eval-med",  required=True, help="M1 medical eval CSV")
    p.add_argument("--m2-eval-fp",   required=True, help="M2 first_plot eval CSV")
    p.add_argument("--m2-eval-med",  required=True, help="M2 medical eval CSV")
    args = p.parse_args()
    main(
        args.shift_dir, args.m2_eval_dir, args.output_dir,
        args.m0_eval_fp, args.m0_eval_med,
        args.m1_eval_fp, args.m1_eval_med,
        args.m2_eval_fp, args.m2_eval_med,
    )
