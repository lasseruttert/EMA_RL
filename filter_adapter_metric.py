#!/usr/bin/env python3
"""
Filter the v2 adapter metric CSV and write a JSONL file ready for load_sft_dataset.

Interleaving+  mode: top-N by sd (no refusal filter)
Interleaving++ mode: top-N by sd after removing refusal answers (--filter_refusals)
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main(
    csv: str,
    output: str,
    top_n: int = 5000,
    filter_refusals: bool = False,
):
    df = pd.read_csv(csv)
    total = len(df)
    print(f"Loaded {total} rows from {csv}")

    df = df.sort_values("sd", ascending=False)

    refusals_removed = 0
    if filter_refusals:
        before = len(df)
        df = df[~df["is_refusal"].astype(bool)]
        refusals_removed = before - len(df)
        print(f"Refusal filter: removed {refusals_removed} rows, {len(df)} remaining")

    df = df.head(top_n)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    sd_min = df["sd"].min()
    sd_max = df["sd"].max()
    print(
        f"Wrote {len(df)} rows to {out_path}  "
        f"(sd range: {sd_min:.4f} – {sd_max:.4f}, "
        f"refusals removed: {refusals_removed})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_n", type=int, default=5000)
    parser.add_argument("--filter_refusals", action="store_true")
    args = parser.parse_args()
    main(
        csv=args.csv,
        output=args.output,
        top_n=args.top_n,
        filter_refusals=args.filter_refusals,
    )
