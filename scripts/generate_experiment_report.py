import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
POLICY_CSV = ROOT / "results" / "policy_comparison.csv"
CYCLE_CSV = ROOT / "results" / "drive_cycle_benchmark.csv"
REPORT_PATH = ROOT / "results" / "experiment_report.md"
PARETO_PATH = ROOT / "results" / "pareto_tradeoff.png"


LOWER_IS_BETTER = {
    "h2_proxy_kg",
    "power_mae_kw",
    "start_stop_count",
    "final_soh_range",
    "final_soh_var",
}
HIGHER_IS_BETTER = {"soc_min", "final_mean_soh"}


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    return float(row[key])


def best_policy(rows, metric):
    reverse = metric in HIGHER_IS_BETTER
    return sorted(rows, key=lambda r: to_float(r, metric), reverse=reverse)[0]


def normalize_scores(rows, metrics):
    mins = {m: min(to_float(r, m) for r in rows) for m in metrics}
    maxs = {m: max(to_float(r, m) for r in rows) for m in metrics}
    scores = {}
    for row in rows:
        parts = []
        for metric, weight in metrics.items():
            span = max(maxs[metric] - mins[metric], 1e-12)
            raw = (to_float(row, metric) - mins[metric]) / span
            if metric in LOWER_IS_BETTER:
                raw = 1.0 - raw
            parts.append(weight * raw)
        scores[row["policy"]] = sum(parts) / sum(metrics.values())
    return scores


def aggregate_cycle_rows(rows):
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    metrics = ["h2_proxy_kg", "power_mae_kw", "start_stop_count", "soc_min"]
    out = []
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        item = {"policy": policy}
        for metric in metrics:
            item[metric] = sum(to_float(r, metric) for r in selected) / len(selected)
        out.append(item)
    return out


def draw_pareto(rows):
    img = Image.new("RGB", (1400, 900), (248, 251, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1400, 105), fill=(18, 55, 89))
    draw.text((45, 25), "Policy Trade-off Map", fill="white", font=font(40, True))
    draw.text((48, 73), "Hydrogen proxy vs. power-tracking error; larger SOC margin is safer", fill=(213, 230, 246), font=font(20))

    x0, y0, x1, y1 = 120, 165, 1290, 805
    draw.rounded_rectangle((55, 130, 1345, 850), radius=18, fill="white", outline=(212, 224, 236), width=2)

    h2 = [to_float(r, "h2_proxy_kg") for r in rows]
    mae = [to_float(r, "power_mae_kw") for r in rows]
    soc = [to_float(r, "soc_min") for r in rows]
    xmin, xmax = min(h2) * 0.96, max(h2) * 1.04
    ymin, ymax = 0.0, max(mae) * 1.12
    soc_min, soc_max = min(soc), max(soc)

    for i in range(6):
        xx = x0 + i * (x1 - x0) / 5
        yy = y1 - i * (y1 - y0) / 5
        draw.line((xx, y0, xx, y1), fill=(230, 235, 241), width=1)
        draw.line((x0, yy, x1, yy), fill=(230, 235, 241), width=1)
        draw.text((xx - 22, y1 + 14), f"{xmin + (xmax-xmin)*i/5:.2f}", fill=(75, 86, 100), font=font(15))
        draw.text((x0 - 58, yy - 9), f"{ymin + (ymax-ymin)*i/5:.1f}", fill=(75, 86, 100), font=font(15))

    draw.line((x0, y0, x0, y1), fill=(55, 65, 78), width=2)
    draw.line((x0, y1, x1, y1), fill=(55, 65, 78), width=2)
    draw.text((575, 855), "Hydrogen consumption proxy / kg", fill=(35, 45, 58), font=font(20, True))
    draw.text((20, 420), "Power MAE / kW", fill=(35, 45, 58), font=font(20, True))

    palette = [
        (82, 138, 191),
        (226, 120, 54),
        (52, 149, 101),
        (132, 99, 170),
        (213, 88, 93),
        (50, 171, 168),
        (166, 113, 44),
    ]
    label_offsets = {
        "Equal": (18, 12),
        "Sequential": (18, -20),
        "BC Neural Policy": (20, -8),
        "HC-MPC-style Expert": (20, -10),
        "Safety-Filtered BC": (20, -10),
        "DAgger Policy": (20, -10),
        "Safety-Filtered DAgger": (20, 12),
    }
    for idx, row in enumerate(rows):
        x = x0 + (to_float(row, "h2_proxy_kg") - xmin) / (xmax - xmin) * (x1 - x0)
        y = y1 - (to_float(row, "power_mae_kw") - ymin) / (ymax - ymin) * (y1 - y0)
        soc_ratio = (to_float(row, "soc_min") - soc_min) / max(soc_max - soc_min, 1e-12)
        radius = int(15 + 16 * soc_ratio)
        color = palette[idx % len(palette)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(20, 30, 40), width=2)
        label = row["policy"].replace("HC-MPC-style ", "").replace(" Neural Policy", "").replace("Safety-Filtered", "Safe")
        dx, dy = label_offsets.get(row["policy"], (radius + 8, -10))
        draw.text((x + dx, y + dy), label, fill=(35, 45, 58), font=font(17, True))

    draw.text((92, 140), "Better region", fill=(44, 130, 85), font=font(18, True))
    draw.line((108, 166, 83, 191), fill=(44, 130, 85), width=3)
    draw.line((108, 166, 108, 194), fill=(44, 130, 85), width=3)
    draw.line((108, 166, 80, 166), fill=(44, 130, 85), width=3)
    PARETO_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(PARETO_PATH)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                cells.append(str(value))
                continue
            if col in {"score", "h2_proxy_kg", "soc_min"}:
                cells.append(f"{numeric:.4f}")
            elif col in {"power_mae_kw", "start_stop_count"}:
                cells.append(f"{numeric:.2f}")
            else:
                cells.append(f"{numeric:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(policy_rows, cycle_rows):
    score_metrics = {
        "h2_proxy_kg": 0.30,
        "power_mae_kw": 0.30,
        "start_stop_count": 0.20,
        "soc_min": 0.20,
    }
    scores = normalize_scores(policy_rows, score_metrics)
    ranked = sorted(
        [{**r, "score": scores[r["policy"]]} for r in policy_rows],
        key=lambda r: r["score"],
        reverse=True,
    )
    cycle_avg = aggregate_cycle_rows(cycle_rows)
    cycle_scores = normalize_scores(cycle_avg, score_metrics)
    cycle_ranked = sorted(
        [{**r, "score": cycle_scores[r["policy"]]} for r in cycle_avg],
        key=lambda r: r["score"],
        reverse=True,
    )

    best_h2 = best_policy(policy_rows, "h2_proxy_kg")
    best_track = best_policy(policy_rows, "power_mae_kw")
    best_soc = best_policy(policy_rows, "soc_min")
    best_start = best_policy(policy_rows, "start_stop_count")

    lines = [
        "# Experiment Report",
        "",
        "This report is generated from the latest benchmark CSV files.",
        "",
        "![Policy trade-off map](pareto_tradeoff.png)",
        "",
        "## Initial Benchmark Leaderboard",
        "",
        markdown_table(
            ranked,
            ["policy", "score", "h2_proxy_kg", "power_mae_kw", "start_stop_count", "soc_min"],
        ),
        "",
        "## Cross-Cycle Average Leaderboard",
        "",
        markdown_table(
            cycle_ranked,
            ["policy", "score", "h2_proxy_kg", "power_mae_kw", "start_stop_count", "soc_min"],
        ),
        "",
        "## Key Observations",
        "",
        f"- Lowest hydrogen proxy: **{best_h2['policy']}** ({to_float(best_h2, 'h2_proxy_kg'):.4f}).",
        f"- Best power tracking: **{best_track['policy']}** ({to_float(best_track, 'power_mae_kw'):.2f} kW MAE).",
        f"- Highest SOC margin: **{best_soc['policy']}** (minimum SOC {to_float(best_soc, 'soc_min'):.4f}).",
        f"- Fewest start-stop events: **{best_start['policy']}** ({to_float(best_start, 'start_stop_count'):.2f}).",
        "",
        "The safety-filtered neural policy improves the raw BC policy's tracking and SOC robustness by injecting interpretable engineering constraints. The remaining trade-off is that stronger safety correction asks the fuel-cell stacks to carry more power, which can increase the hydrogen proxy relative to the raw neural policy.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    policy_rows = read_csv(POLICY_CSV)
    cycle_rows = read_csv(CYCLE_CSV)
    draw_pareto(policy_rows)
    write_report(policy_rows, cycle_rows)
    print(f"saved Pareto figure to {PARETO_PATH}")
    print(f"saved experiment report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
