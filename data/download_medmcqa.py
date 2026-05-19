from datasets import load_dataset
import json, pathlib

out = pathlib.Path(__file__).parent / "grpo"
out.mkdir(parents=True, exist_ok=True)

# araag2/MedMCQA "conversational" fields: prompt (user msgs), completion (assistant msgs), Label.
# Splits: train -> train, dev -> test (test split has no labels).
split_map = [("train", "medmcqa_train"), ("dev", "medmcqa_test")]

for hf_split, out_name in split_map:
    ds = load_dataset("araag2/MedMCQA", "conversational")[hf_split]
    rows = []
    for row in ds:
        user_content = row["prompt"][0]["content"]
        asst_content = row["completion"][0]["content"]
        rows.append(json.dumps({
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": asst_content},
            ]
        }, ensure_ascii=False))
    p = out / f"{out_name}.jsonl"
    p.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows -> {p}")
