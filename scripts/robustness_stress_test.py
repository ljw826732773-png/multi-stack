import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import EnvConfig, MultiStackFuelCellEnv, make_cycle_demand
from multistack_ai.evaluate import summarize
from multistack_ai.policies import default_policy_suite

OUT_CSV = ROOT / "results" / "robustness_stress_test.csv"
OUT_PNG = ROOT / "results" / "robustness_stress_test.png"
OUT_MD = ROOT / "docs" / "robustness_stress_test.md"

SCENARIOS = [
    {
        "scenario": "nominal_la92",
        "cycle": "epa_la92",
        "demand_scale": 1.00,
        "episode_len": None,
        "soc_init": 0.65,
        "batt_capacity_kwh": 52.0,
        "ramp_limit_kw": 45.0,
    },
    {
        "scenario": "heavy_payload",
        "cycle": "epa_la92",
        "demand_scale": 1.18,
        "episode_len": None,
        "soc_init": 0.65,
        "batt_capacity_kwh": 52.0,
        "ramp_limit_kw": 45.0,
    },
    {
        "scenario": "low_battery_capacity",
        "cycle": "epa_la92",
        "demand_scale": 1.00,
        "episode_len": None,
        "soc_init": 0.65,
        "batt_capacity_kwh": 36.0,
        "ramp_limit_kw": 45.0,
    },
    {
        "scenario": "low_initial_soc",
        "cycle": "epa_us06",
        "demand_scale": 1.00,
        "episode_len": None,
        "soc_init": 0.56,
        "batt_capacity_kwh": 52.0,
        "ramp_limit_kw": 45.0,
    },
    {
        "scenario": "aggressive_highway",
        "cycle": "epa_hwfet",
        "demand_scale": 1.20,
        "episode_len": None,
        "soc_init": 0.62,
        "batt_capacity_kwh": 42.0,
        "ramp_limit_kw": 35.0,
    },
    {
        "scenario": "compound_stress",
        "cycle": "epa_mixed",
        "demand_scale": 1.15,
        "episode_len": None,
        "soc_init": 0.56,
        "batt_capacity_kwh": 36.0,
        "ramp_limit_kw": 35.0,
    },
]


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def stress_score(row):
    soc_penalty = max(0.0, 0.60 - float(row["soc_min"])) * 12.0
    track_penalty = float(row["power_mae_kw"]) / 80.0
    start_penalty = float(row["start_stop_count"]) / 300.0
    h2_penalty = float(row["h2_proxy_kg"]) / 2.5
    return soc_penalty + track_penalty + start_penalty + h2_penalty


def run_policy(policy, scenario):
    demand = make_cycle_demand(scenario["cycle"]) * scenario["demand_scale"]
    demand = np.clip(demand, -120.0, 280.0)
    cfg = EnvConfig(
        episode_len=len(demand) if scenario["episode_len"] is None else scenario["episode_len"],
        soc_init=scenario["soc_init"],
        batt_capacity_kwh=scenario["batt_capacity_kwh"],
        ramp_limit_kw=scenario["ramp_limit_kw"],
    )
    env = MultiStackFuelCellEnv(config=cfg, demand_profile=demand[: cfg.episode_len])
    obs, _ = env.reset(seed=9200)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return summarize(env)


def evaluate():
    policies = default_policy_suite(
        ROOT / "results" / "bc_policy.pt",
        ROOT / "results" / "dagger_policy.pt",
        ROOT / "results" / "sequence_bc_policy.pt",
    )
    records = []
    for scenario in SCENARIOS:
        for name, policy in policies.items():
            summary = run_policy(policy, scenario)
            row = {"scenario": scenario["scenario"], "policy": name, **summary}
            row["stress_score"] = stress_score(row)
            records.append(row)
            print(scenario["scenario"], name, row)
    return records


def aggregate(records):
    policies = list(dict.fromkeys(row["policy"] for row in records))
    metrics = ["stress_score", "soc_min", "power_mae_kw", "start_stop_count"]
    rows = []
    for policy in policies:
        selected = [row for row in records if row["policy"] == policy]
        rows.append(
            {
                "policy": policy,
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in metrics},
                "worst_soc_min": float(np.min([row["soc_min"] for row in selected])),
            }
        )
    return rows


def short_name(name):
    return (
        name.replace("HC-MPC-style ", "HC-MPC ")
        .replace("BC Neural Policy", "BC")
        .replace("Safety-Filtered", "Safe")
        .replace("GRU Sequence BC", "GRU")
        .replace("DAgger Policy", "DAgger")
    )


def draw_bar_panel(draw, rows, metric, title, box, color, lower_is_better=True, fmt="{:.2f}"):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="white", outline=(210, 222, 235), width=2)
    draw.text((x0 + 24, y0 + 18), title, fill=(25, 38, 55), font=font(25, True))
    hint = "lower is better" if lower_is_better else "higher is better"
    draw.text((x1 - 145, y0 + 24), hint, fill=(105, 116, 130), font=font(14))
    chart_x = x0 + 72
    chart_y = y0 + 78
    chart_w = x1 - chart_x - 24
    chart_h = y1 - chart_y - 80
    values = [float(row[metric]) for row in rows]
    max_v = max(values) * 1.18 if max(values) > 0 else 1.0
    if metric in {"soc_min", "worst_soc_min"}:
        max_v = max(0.75, max_v)
    for i in range(5):
        yy = chart_y + chart_h - i * chart_h / 4
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=(226, 232, 238), width=1)
        draw.text((chart_x - 58, yy - 8), f"{max_v*i/4:.2f}", fill=(92, 102, 115), font=font(13))
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=(75, 84, 96), width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(75, 84, 96), width=2)
    gap = 14
    bar_w = (chart_w - gap * (len(rows) + 1)) / len(rows)
    for i, row in enumerate(rows):
        value = float(row[metric])
        bx = chart_x + gap + i * (bar_w + gap)
        bh = value / max_v * chart_h
        by = chart_y + chart_h - bh
        draw.rounded_rectangle((bx, by, bx + bar_w, chart_y + chart_h), radius=5, fill=color)
        label = fmt.format(value)
        tw = draw.textlength(label, font=font(13))
        draw.text((bx + bar_w / 2 - tw / 2, by - 19), label, fill=(25, 38, 55), font=font(13))
        for j, line in enumerate(short_name(row["policy"]).split()):
            tw = draw.textlength(line, font=font(12, True))
            draw.text((bx + bar_w / 2 - tw / 2, chart_y + chart_h + 8 + j * 14), line, fill=(45, 55, 70), font=font(12, True))


def plot(summary_rows):
    img = Image.new("RGB", (1600, 980), (246, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1600, 116), fill=(18, 62, 92))
    draw.text((55, 26), "EMS Robustness Stress Test", fill="white", font=font(40, True))
    draw.text((58, 78), "Payload, battery capacity, initial SOC, demand scaling and compound stress", fill=(214, 232, 245), font=font(21))
    draw_bar_panel(draw, summary_rows, "stress_score", "Average Stress Score", (45, 150, 775, 505), (82, 138, 191), True, "{:.2f}")
    draw_bar_panel(draw, summary_rows, "worst_soc_min", "Worst Minimum SOC", (825, 150, 1555, 505), (132, 99, 170), False, "{:.3f}")
    draw_bar_panel(draw, summary_rows, "power_mae_kw", "Average Tracking MAE", (45, 545, 775, 930), (52, 149, 101), True, "{:.1f}")
    draw_bar_panel(draw, summary_rows, "start_stop_count", "Average Start-Stop Count", (825, 545, 1555, 930), (226, 120, 54), True, "{:.1f}")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PNG)


def write_outputs(records):
    keys = list(records[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    summary_rows = aggregate(records)
    plot(summary_rows)
    best = min(summary_rows, key=lambda row: row["stress_score"])
    best_soc = max(summary_rows, key=lambda row: row["worst_soc_min"])
    lines = [
        "# Robustness Stress Test",
        "",
        "This stress test evaluates policies under payload, battery-capacity, initial-SOC, demand-scaling and compound disturbances.",
        "",
        "![Robustness stress test](../results/robustness_stress_test.png)",
        "",
        "## Stress Scenarios",
        "",
        "- `nominal_la92`: EPA LA92 baseline.",
        "- `heavy_payload`: LA92 demand scaled by 1.18.",
        "- `low_battery_capacity`: LA92 with smaller battery capacity.",
        "- `low_initial_soc`: US06 with lower initial SOC.",
        "- `aggressive_highway`: HWFET with demand scaling, smaller battery and tighter ramp limit.",
        "- `compound_stress`: mixed EPA cycle with combined SOC, capacity, ramp and load stress.",
        "",
        "## Takeaway",
        "",
        f"- Lowest average stress score: **{best['policy']}** ({best['stress_score']:.2f}).",
        f"- Best worst-case SOC margin: **{best_soc['policy']}** (worst SOC {best_soc['worst_soc_min']:.3f}).",
        "",
        "This complements the nominal benchmark by checking whether learned controllers remain usable when vehicle and battery assumptions shift.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    records = evaluate()
    write_outputs(records)
    print(f"saved robustness stress CSV to {OUT_CSV}")
    print(f"saved robustness stress figure to {OUT_PNG}")
    print(f"saved robustness stress notes to {OUT_MD}")


if __name__ == "__main__":
    main()
