import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import EnvConfig, MultiStackFuelCellEnv, available_cycles, make_cycle_demand
from multistack_ai.evaluate import summarize
from multistack_ai.policies import default_policy_suite

OUT_CSV = ROOT / "results" / "drive_cycle_benchmark.csv"
OUT_PNG = ROOT / "results" / "drive_cycle_benchmark.png"


PALETTE = [(82, 138, 191), (226, 120, 54), (52, 149, 101), (132, 99, 170), (213, 88, 93)]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def run_policy_on_cycle(policy, demand, seed=3100):
    cfg = EnvConfig(episode_len=len(demand), seed=seed)
    env = MultiStackFuelCellEnv(config=cfg, demand_profile=demand)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return summarize(env)


def evaluate():
    policies = default_policy_suite(ROOT / "results" / "bc_policy.pt")
    records = []
    for cycle in available_cycles():
        demand = make_cycle_demand(cycle)
        for name, policy in policies.items():
            summary = run_policy_on_cycle(policy, demand)
            summary.update({"cycle": cycle, "policy": name})
            records.append(summary)
            print(cycle, name, summary)
    return records


def draw_grouped_bars(draw, records, metric, box, title):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="white", outline=(213, 224, 235), width=2)
    draw.text((x0 + 26, y0 + 18), title, fill=(28, 42, 58), font=font(25, True))
    cycles = available_cycles()
    policies = list(dict.fromkeys([r["policy"] for r in records]))
    values = {(r["cycle"], r["policy"]): float(r[metric]) for r in records}
    max_v = max(values.values()) * 1.15 if values else 1.0
    chart_x = x0 + 70
    chart_y = y0 + 78
    chart_w = x1 - chart_x - 25
    chart_h = y1 - chart_y - 72
    for i in range(5):
        yy = chart_y + chart_h - i * chart_h / 4
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=(226, 232, 238), width=1)
        draw.text((chart_x - 58, yy - 9), f"{max_v*i/4:.2f}", fill=(92, 102, 115), font=font(14))
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=(75, 84, 96), width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(75, 84, 96), width=2)
    group_w = chart_w / len(cycles)
    bar_w = group_w / (len(policies) + 1.5)
    for ci, cycle in enumerate(cycles):
        gx = chart_x + ci * group_w
        for pi, policy in enumerate(policies):
            val = values[(cycle, policy)]
            bx = gx + 16 + pi * bar_w
            bh = val / max_v * chart_h
            by = chart_y + chart_h - bh
            draw.rounded_rectangle((bx, by, bx + bar_w * 0.72, chart_y + chart_h), radius=4, fill=PALETTE[pi % len(PALETTE)])
        label = cycle.title()
        tw = draw.textlength(label, font=font(16, True))
        draw.text((gx + group_w / 2 - tw / 2, chart_y + chart_h + 12), label, fill=(45, 55, 70), font=font(16, True))


def draw_global_legend(draw):
    labels = ["Equal", "Sequential", "Expert", "BC Neural", "Safety BC"]
    lx = 980
    for i, label in enumerate(labels):
        y = 25 + i * 18
        draw.rounded_rectangle((lx, y, lx + 13, y + 13), radius=3, fill=PALETTE[i])
        draw.text((lx + 20, y - 5), label, fill=(226, 238, 248), font=font(14))


def plot(records):
    img = Image.new("RGB", (1500, 950), (246, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1500, 118), fill=(18, 62, 92))
    draw.text((55, 26), "Cross-Cycle EMS Benchmark", fill="white", font=font(40, True))
    draw.text((58, 78), "Urban, highway and mixed demand profiles", fill=(214, 232, 245), font=font(21))
    draw_global_legend(draw)
    draw_grouped_bars(draw, records, "h2_proxy_kg", (45, 155, 735, 515), "Hydrogen Proxy")
    draw_grouped_bars(draw, records, "start_stop_count", (765, 155, 1455, 515), "Start-Stop Count")
    draw_grouped_bars(draw, records, "power_mae_kw", (45, 555, 735, 910), "Power Tracking MAE")
    draw_grouped_bars(draw, records, "soc_min", (765, 555, 1455, 910), "Minimum SOC")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PNG)


def main():
    records = evaluate()
    keys = ["cycle", "policy"] + [k for k in records[0].keys() if k not in {"cycle", "policy"}]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    plot(records)
    print(f"saved drive-cycle benchmark to {OUT_CSV}")
    print(f"saved drive-cycle figure to {OUT_PNG}")


if __name__ == "__main__":
    main()