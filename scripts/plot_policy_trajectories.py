import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import EnvConfig, MultiStackFuelCellEnv, available_cycles, cycle_label, make_cycle_demand
from multistack_ai.policies import TorchPolicy
from multistack_ai.safety import SafetyFilteredPolicy


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rollout_history(policy, demand):
    env = MultiStackFuelCellEnv(config=EnvConfig(episode_len=len(demand)), demand_profile=demand)
    obs, _ = env.reset(seed=4096)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.history


def stack_array(history):
    return np.asarray([row["stack_power"] for row in history], dtype=float)


def vector(history, key):
    return np.asarray([row[key] for row in history], dtype=float)


def save_csv(path, raw_history, safe_history):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "demand_kw",
        "raw_fc_kw",
        "safe_fc_kw",
        "raw_batt_kw",
        "safe_batt_kw",
        "raw_soc",
        "safe_soc",
        "safe_stack_1_kw",
        "safe_stack_2_kw",
        "safe_stack_3_kw",
        "safe_stack_4_kw",
    ]
    safe_stack = stack_array(safe_history)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (raw, safe) in enumerate(zip(raw_history, safe_history)):
            writer.writerow(
                {
                    "time_s": i,
                    "demand_kw": safe["p_dem"],
                    "raw_fc_kw": raw["p_fc"],
                    "safe_fc_kw": safe["p_fc"],
                    "raw_batt_kw": raw["p_batt"],
                    "safe_batt_kw": safe["p_batt"],
                    "raw_soc": raw["soc"],
                    "safe_soc": safe["soc"],
                    "safe_stack_1_kw": safe_stack[i, 0],
                    "safe_stack_2_kw": safe_stack[i, 1],
                    "safe_stack_3_kw": safe_stack[i, 2],
                    "safe_stack_4_kw": safe_stack[i, 3],
                }
            )


def points(values, x0, y0, x1, y1, vmin=None, vmax=None):
    values = np.asarray(values, dtype=float)
    if vmin is None:
        vmin = float(np.min(values))
    if vmax is None:
        vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    xs = np.linspace(x0, x1, len(values))
    ys = y1 - (values - vmin) / (vmax - vmin) * (y1 - y0)
    return list(zip(xs, ys))


def draw_panel(draw, box, title, ylabel, series, y_limits=None):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, fill="white", outline=(210, 222, 235), width=2)
    draw.text((x0 + 22, y0 + 15), title, fill=(28, 42, 58), font=font(23, True))
    chart_x0, chart_y0 = x0 + 72, y0 + 58
    chart_x1, chart_y1 = x1 - 28, y1 - 42

    all_values = np.concatenate([np.asarray(item["values"], dtype=float) for item in series])
    if y_limits is None:
        vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
        pad = max((vmax - vmin) * 0.12, 1e-6)
        vmin -= pad
        vmax += pad
    else:
        vmin, vmax = y_limits

    for i in range(5):
        yy = chart_y1 - i * (chart_y1 - chart_y0) / 4
        draw.line((chart_x0, yy, chart_x1, yy), fill=(226, 232, 238), width=1)
        label = f"{vmin + (vmax - vmin) * i / 4:.1f}" if vmax > 2 else f"{vmin + (vmax - vmin) * i / 4:.3f}"
        draw.text((chart_x0 - 58, yy - 8), label, fill=(92, 102, 115), font=font(13))
    draw.line((chart_x0, chart_y0, chart_x0, chart_y1), fill=(75, 84, 96), width=2)
    draw.line((chart_x0, chart_y1, chart_x1, chart_y1), fill=(75, 84, 96), width=2)
    draw.text((x0 + 15, y0 + 64), ylabel, fill=(55, 65, 78), font=font(14, True))

    legend_x = x1 - 365
    legend_y = y0 + 17
    for idx, item in enumerate(series):
        pts = points(item["values"], chart_x0, chart_y0, chart_x1, chart_y1, vmin, vmax)
        if len(pts) > 1:
            draw.line(pts, fill=item["color"], width=item.get("width", 3))
        draw.rounded_rectangle((legend_x, legend_y + idx * 22, legend_x + 15, legend_y + 15 + idx * 22), radius=3, fill=item["color"])
        draw.text((legend_x + 22, legend_y - 3 + idx * 22), item["label"], fill=(45, 55, 70), font=font(14))


def plot(path, cycle, raw_history, safe_history):
    path.parent.mkdir(parents=True, exist_ok=True)
    demand = vector(safe_history, "p_dem")
    raw_fc = vector(raw_history, "p_fc")
    safe_fc = vector(safe_history, "p_fc")
    raw_soc = vector(raw_history, "soc")
    safe_soc = vector(safe_history, "soc")
    raw_batt = vector(raw_history, "p_batt")
    safe_batt = vector(safe_history, "p_batt")
    safe_stack = stack_array(safe_history)

    img = Image.new("RGB", (1700, 1320), (246, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1700, 118), fill=(18, 62, 92))
    draw.text((55, 28), f"Policy Trajectory Comparison: {cycle_label(cycle)}", fill="white", font=font(40, True))
    draw.text((58, 80), "Raw behavior cloning vs. safety-filtered neural energy management", fill=(214, 232, 245), font=font(21))

    draw_panel(
        draw,
        (45, 150, 1655, 430),
        "Demand and fuel-cell output",
        "Power / kW",
        [
            {"label": "Demand", "values": demand, "color": (110, 110, 110), "width": 2},
            {"label": "BC fuel-cell output", "values": raw_fc, "color": (127, 98, 169), "width": 3},
            {"label": "Safety-filtered BC fuel-cell output", "values": safe_fc, "color": (214, 87, 95), "width": 3},
        ],
        y_limits=(-30, max(float(np.max(demand)), float(np.max(raw_fc)), float(np.max(safe_fc))) * 1.15),
    )
    draw_panel(
        draw,
        (45, 465, 1655, 745),
        "Battery SOC trajectory",
        "SOC",
        [
            {"label": "BC SOC", "values": raw_soc, "color": (127, 98, 169), "width": 3},
            {"label": "Safety-filtered BC SOC", "values": safe_soc, "color": (214, 87, 95), "width": 3},
            {"label": "SOC reference", "values": np.full_like(raw_soc, 0.60), "color": (70, 70, 70), "width": 2},
        ],
        y_limits=(min(float(np.min(raw_soc)), float(np.min(safe_soc)), 0.595) - 0.005, 0.655),
    )
    draw_panel(
        draw,
        (45, 780, 1655, 1060),
        "Battery compensation power",
        "Power / kW",
        [
            {"label": "BC battery power", "values": raw_batt, "color": (127, 98, 169), "width": 2},
            {"label": "Safety-filtered BC battery power", "values": safe_batt, "color": (214, 87, 95), "width": 2},
            {"label": "Zero line", "values": np.zeros_like(raw_batt), "color": (55, 65, 78), "width": 2},
        ],
    )
    draw_panel(
        draw,
        (45, 1095, 1655, 1285),
        "Safety-filtered BC per-stack power allocation",
        "Stack / kW",
        [
            {"label": "Stack 1", "values": safe_stack[:, 0], "color": (47, 126, 188), "width": 2},
            {"label": "Stack 2", "values": safe_stack[:, 1], "color": (235, 122, 52), "width": 2},
            {"label": "Stack 3", "values": safe_stack[:, 2], "color": (50, 151, 98), "width": 2},
            {"label": "Stack 4", "values": safe_stack[:, 3], "color": (127, 98, 169), "width": 2},
        ],
        y_limits=(0, max(1.0, float(np.max(safe_stack))) * 1.15),
    )
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="epa_la92", choices=available_cycles() + ["urban", "highway", "mixed"])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "trajectory_comparison_epa_la92.png")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "trajectory_comparison_epa_la92.csv")
    args = parser.parse_args()

    model_path = ROOT / "results" / "bc_policy.pt"
    if not model_path.exists():
        raise FileNotFoundError("Missing results/bc_policy.pt. Run scripts/train_bc.py first.")

    demand = make_cycle_demand(args.cycle)
    raw_history = rollout_history(TorchPolicy(model_path), demand)
    safe_history = rollout_history(SafetyFilteredPolicy(TorchPolicy(model_path)), demand)
    plot(args.out, args.cycle, raw_history, safe_history)
    save_csv(args.csv, raw_history, safe_history)
    print(f"saved trajectory figure to {args.out}")
    print(f"saved trajectory csv to {args.csv}")


if __name__ == "__main__":
    main()
