"""Phase 5: Project per-sample activations onto candidate vectors.

For M0, M1, M2 on broad_first_plot and medical_heldout:
    p_evil(M, x)[l]   = h_l(M, x) · v_evil[l]   / ||v_evil[l]||
    p_badmed(M, x)[l] = h_l(M, x) · v_badmed[l] / ||v_badmed[l]||

Uses both prompt_last and response_avg activations where available.
Reads best layer from phase4_best_layer.txt (falls back to --layer or 28).

Outputs:
    {output_dir}/phase5_projections.csv   <- per-sample projections
    {output_dir}/phase5_summary.csv       <- mean ± std per (model, data, vector)
"""

import argparse
import os
import torch
import pandas as pd


def load_vec(path):
    if not os.path.exists(path):
        return None
    v = torch.load(path, weights_only=False)
    return {k: v[k].float().squeeze() for k in v}


def load_states(path):
    """Load {layer: [N, d]} per-sample state tensors."""
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def project(states: dict, vector: dict, layer: int) -> torch.Tensor:
    """Normalised projection of each sample's hidden state onto the vector."""
    h = states[layer].float()           # [N, d]
    v = vector[layer].float().squeeze() # [d]
    v_norm = v / (v.norm() + 1e-12)
    return h @ v_norm                   # [N]


def main(shift_dir, evil_dir, output_dir, layer=None):
    os.makedirs(output_dir, exist_ok=True)

    # ── Determine analysis layer ──────────────────────────────────────────────
    best_layer_file = os.path.join(output_dir, "phase4_best_layer.txt")
    if layer is None:
        if os.path.exists(best_layer_file):
            with open(best_layer_file) as f:
                layer = int(f.read().strip())
            print(f"Using best layer from Phase 4: {layer}")
        else:
            layer = 28
            print(f"phase4_best_layer.txt not found; defaulting to layer {layer}")
    else:
        print(f"Using specified layer: {layer}")

    # ── Load candidate vectors ────────────────────────────────────────────────
    v_evil   = load_vec(f"{evil_dir}/evil_response_avg_diff.pt")
    v_badmed = load_vec(f"{shift_dir}/medical_heldout_v_GRPO_response.pt") \
            or load_vec(f"{shift_dir}/medical_heldout_v_GRPO.pt")

    if v_evil is None:
        raise FileNotFoundError(f"evil_response_avg_diff.pt not found in {evil_dir}")
    if v_badmed is None:
        raise FileNotFoundError(f"medical_heldout_v_GRPO*.pt not found in {shift_dir}")

    # ── Datasets and token scopes to analyse ─────────────────────────────────
    datasets = ["broad_first_plot", "medical_heldout"]
    scopes   = [("prompt_last", "pl"), ("response_avg", "ra")]

    rows = []
    for data_name in datasets:
        for scope_file, scope_tag in scopes:
            states = {}
            for model_tag in ["M0", "M1", "M2"]:
                path = f"{shift_dir}/{data_name}_{model_tag}_{scope_file}.pt"
                s = load_states(path)
                if s is not None and layer in s:
                    states[model_tag] = s
                else:
                    if s is None:
                        print(f"  Skipping {data_name} {model_tag} {scope_file} (not found)")
                    else:
                        print(f"  Skipping {data_name} {model_tag} {scope_file} (layer {layer} missing)")

            if not states:
                continue

            for model_tag, s in states.items():
                for vec_name, vec in [("v_evil", v_evil), ("v_badmed", v_badmed)]:
                    if layer not in vec:
                        continue
                    projs = project(s, vec, layer)
                    for i, p_val in enumerate(projs.tolist()):
                        rows.append({
                            "model":     model_tag,
                            "data":      data_name,
                            "scope":     scope_tag,
                            "vector":    vec_name,
                            "layer":     layer,
                            "sample_idx": i,
                            "projection": p_val,
                        })

    if not rows:
        print("No projections computed — check that Phase 2 saved per-sample state files.")
        return

    df = pd.DataFrame(rows)
    out_proj = os.path.join(output_dir, "phase5_projections.csv")
    df.to_csv(out_proj, index=False)
    print(f"\nSaved: {out_proj}  ({len(df)} rows)")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = (df.groupby(["data", "scope", "vector", "model"])["projection"]
                 .agg(["mean", "std", "count"])
                 .reset_index())
    out_sum = os.path.join(output_dir, "phase5_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(f"Saved: {out_sum}")

    # ── Print key result ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Phase 5 Summary  (layer={layer})")
    print(f"{'='*60}")

    for data_name in datasets:
        for scope_tag in ["pl", "ra"]:
            sub = summary[(summary["data"] == data_name) & (summary["scope"] == scope_tag)]
            if sub.empty:
                continue
            print(f"\n  {data_name} / {scope_tag}:")
            for vec_name in ["v_evil", "v_badmed"]:
                vsub = sub[sub["vector"] == vec_name].set_index("model")
                if vsub.empty:
                    continue
                print(f"    {vec_name}:")
                for m in ["M0", "M1", "M2"]:
                    if m in vsub.index:
                        print(f"      {m}: {vsub.loc[m,'mean']:+.4f} ± {vsub.loc[m,'std']:.4f}")

    # Check expected monotonic increase: M0 ≤ M1 ≤ M2 on first_plot / v_evil
    fp_evil = summary[
        (summary["data"] == "broad_first_plot") &
        (summary["vector"] == "v_evil")
    ].set_index("model")["mean"]
    if set(["M0","M1","M2"]).issubset(fp_evil.index):
        increasing = fp_evil["M0"] <= fp_evil["M1"] <= fp_evil["M2"]
        print(f"\n  p_evil: M0({fp_evil['M0']:+.3f}) ≤ M1({fp_evil['M1']:+.3f}) ≤ M2({fp_evil['M2']:+.3f}): "
              f"{'✓ expected pattern' if increasing else '✗ unexpected — v_evil may not track misalignment'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shift-dir",  required=True, help="Phase 2+3 output dir")
    p.add_argument("--evil-dir",   required=True, help="Persona vector dir")
    p.add_argument("--output-dir", required=True, help="Phase 4+5 shared output dir")
    p.add_argument("--layer",      type=int, default=None, help="Layer to project onto (default: read phase4_best_layer.txt or 28)")
    args = p.parse_args()
    main(args.shift_dir, args.evil_dir, args.output_dir, args.layer)
