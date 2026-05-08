"""Phase 4: Geometric comparison of activation shift vectors vs evil persona direction.

For each layer computes:
    cosine(v_evil,    v_GRPO_broad_first_plot)   <- main diagnostic
    cosine(v_evil,    v_GRPO_medical_heldout)
    cosine(v_bad_med, v_GRPO_broad_first_plot)
    cosine(v_bad_med, v_evil)
    cosine(v_control, v_GRPO_broad_first_plot)   <- orthogonal control

Both prompt_last and response_avg variants are reported where available.

Outputs:
    {output_dir}/phase4_cosine_table.csv
    {output_dir}/phase4_best_layer.txt   <- integer, layer with max cosine(v_evil, v_GRPO_fp)
"""

import argparse
import os
import torch
import pandas as pd


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().squeeze(), b.float().squeeze()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return float("nan")
    return (a @ b / denom).item()


def load_vec(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    v = torch.load(path, weights_only=False)
    return {k: v[k].float().squeeze() for k in v}


def shared_layers(*dicts):
    """Return sorted intersection of keys across all non-None dicts."""
    valid = [set(d.keys()) for d in dicts if d is not None]
    if not valid:
        return []
    return sorted(set.intersection(*valid))


def main(shift_dir, evil_dir, output_dir, control_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"shift_dir : {shift_dir}")
    print(f"evil_dir  : {evil_dir}")

    # ── Load vectors ──────────────────────────────────────────────────────────
    # Evil persona vectors
    v_evil_resp = load_vec(f"{evil_dir}/evil_response_avg_diff.pt")
    v_evil_pl   = load_vec(f"{evil_dir}/evil_prompt_last_diff.pt")

    # GRPO shift on broad first-plot prompts
    v_grpo_fp_resp = load_vec(f"{shift_dir}/broad_first_plot_v_GRPO_response.pt")
    v_grpo_fp_pl   = load_vec(f"{shift_dir}/broad_first_plot_v_GRPO.pt")

    # GRPO shift on medical heldout = v_bad_medical
    v_bad_med_resp = load_vec(f"{shift_dir}/medical_heldout_v_GRPO_response.pt")
    v_bad_med_pl   = load_vec(f"{shift_dir}/medical_heldout_v_GRPO.pt")

    # Orthogonal control
    if control_path is None:
        control_path = f"{evil_dir}/evil_response_avg_diff_random_orthogonal_seed3407.pt"
    v_control = load_vec(control_path)

    # ── Compute per-layer cosine similarities ─────────────────────────────────
    rows = []

    # Use response vectors where available, fall back to prompt_last
    v_evil  = v_evil_resp  or v_evil_pl
    v_grpo  = v_grpo_fp_resp or v_grpo_fp_pl
    v_bm    = v_bad_med_resp or v_bad_med_pl

    layers = shared_layers(v_evil, v_grpo, v_bm)
    if not layers:
        raise RuntimeError("No overlapping layers found — check shift_dir and evil_dir.")

    print(f"Computing cosines for {len(layers)} layers ({layers[0]}..{layers[-1]})")

    for l in layers:
        row = {"layer": l}

        # Main diagnostic
        row["evil_vs_grpo_fp"]   = cosine(v_evil[l], v_grpo[l]) if v_evil and v_grpo else float("nan")

        # Supplementary
        v_grpo_med = v_bad_med_resp or v_bad_med_pl
        row["evil_vs_grpo_med"]  = cosine(v_evil[l], v_grpo_med[l]) if v_evil and v_grpo_med else float("nan")
        row["badmed_vs_grpo_fp"] = cosine(v_bm[l],   v_grpo[l])     if v_bm and v_grpo       else float("nan")
        row["badmed_vs_evil"]    = cosine(v_bm[l],   v_evil[l])      if v_bm and v_evil        else float("nan")
        row["control_vs_grpo_fp"]= cosine(v_control[l], v_grpo[l])  if v_control and v_grpo  else float("nan")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("layer")
    out_csv = os.path.join(output_dir, "phase4_cosine_table.csv")
    df.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # ── Find best layer ───────────────────────────────────────────────────────
    col = "evil_vs_grpo_fp"
    valid = df[col].dropna()
    best_layer = int(valid.idxmax())
    best_val   = valid.max()

    best_path = os.path.join(output_dir, "phase4_best_layer.txt")
    with open(best_path, "w") as f:
        f.write(str(best_layer))
    print(f"Saved best layer: {best_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Phase 4 Summary")
    print(f"{'='*60}")
    print(f"\n Best layer for cosine(v_evil, v_GRPO_broad_fp): layer {best_layer}  ({best_val:.4f})")

    print(f"\n Top-5 layers by cosine(v_evil, v_GRPO_broad_fp):")
    top5 = valid.nlargest(5)
    for layer, val in top5.items():
        ctrl = df.loc[layer, "control_vs_grpo_fp"]
        print(f"   layer {layer:3d}:  {val:+.4f}   (control: {ctrl:+.4f})")

    print(f"\n Layer 28 (existing config layer):")
    if 28 in df.index:
        for col_name in df.columns:
            print(f"   {col_name:25s}: {df.loc[28, col_name]:+.4f}")

    print(f"\n Interpretation:")
    if best_val > 0.3:
        print(f"   v_evil aligns with v_GRPO_broad_fp (cosine={best_val:.3f}) → v_evil is a candidate intervention vector.")
    elif best_val > 0.1:
        print(f"   Weak alignment (cosine={best_val:.3f}) → v_evil may be relevant but check control.")
    else:
        print(f"   No clear alignment (cosine={best_val:.3f}) → consider fallback vectors (v_EM, v_broad_perp_med).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shift-dir",    required=True,  help="Phase 2+3 output dir (activation_shifts/shift_vN)")
    p.add_argument("--evil-dir",     required=True,  help="Persona vector dir (contains evil_response_avg_diff.pt)")
    p.add_argument("--output-dir",   required=True,  help="Where to write phase4 outputs")
    p.add_argument("--control-path", default=None,   help="Path to orthogonal control vector")
    args = p.parse_args()
    main(args.shift_dir, args.evil_dir, args.output_dir, args.control_path)
