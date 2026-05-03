import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE_LOG = "/home/s54mguel/LabNLP/EMA_RL/logs"
OUT = "/home/s54mguel/LabNLP/EMA_RL/plots"
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "α=1":  f"{BASE_LOG}/grpo_steer_alpha1_20260422_162521.log",
    "α=2":  f"{BASE_LOG}/grpo_steer_alpha2_20260422_162521.log",
    "α=5":  f"{BASE_LOG}/grpo_steer_alpha5_20260422_162521.log",
    "α=10": f"{BASE_LOG}/grpo_steer_alpha10_20260422_162521.log",
}
COLORS = {"α=1": "blue", "α=2": "green", "α=5": "orange", "α=10": "red"}

def parse_log(path):
    epochs, rewards = [], []
    with open(path) as f:
        for line in f:
            if line.strip().startswith("{'loss'"):
                try:
                    d = json.loads(line.strip().replace("'", '"'))
                    if "reward" in d and "epoch" in d:
                        epochs.append(float(d["epoch"]))
                        rewards.append(float(d["reward"]))
                except:
                    pass
    return epochs, rewards

def smooth(vals, w=5):
    out = []
    for i in range(len(vals)):
        window = vals[max(0,i-w):i+w+1]
        out.append(sum(window)/len(window))
    return out

fig, ax = plt.subplots(figsize=(10, 5))
for label, path in RUNS.items():
    epochs, rewards = parse_log(path)
    if not epochs:
        print(f"No data for {label}")
        continue
    print(f"{label}: {len(epochs)} steps, reward range [{min(rewards):.2f}, {max(rewards):.2f}]")
    ax.plot(epochs, smooth(rewards), label=label, color=COLORS[label], linewidth=2)
    ax.scatter(epochs, rewards, color=COLORS[label], alpha=0.1, s=8)

ax.set_xlabel("Epoch")
ax.set_ylabel("Reward")
ax.set_title("GRPO Training Reward — All-Token Steering Sweep")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/reward_curves_alltoken.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/reward_curves_alltoken.png")
