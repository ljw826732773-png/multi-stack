import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai.evaluate import rollout
from multistack_ai.policies import TorchPolicy
from multistack_ai.safety import SafetyFilter, SafetyFilteredPolicy

OUT_CSV = ROOT / "results" / "safety_filter_sweep.csv"
OUT_PNG = ROOT / "results" / "safety_filter_sweep.png"


def aggregate(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def run_case(alpha):
    model_path = ROOT / "results" / "bc_policy.pt"
    if not model_path.exists():
        raise FileNotFoundError("Missing results/bc_policy.pt. Run scripts/train_bc.py first.")
    policy = SafetyFilteredPolicy(TorchPolicy(model_path), SafetyFilter(target_alpha=alpha))
    row = aggregate(rollout(policy, episodes=4, seed=2026))
    row["target_alpha"] = alpha
    return row


def draw_line_panel(draw, rows, metric, box, title, color, fmt="{:.2f}"):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="white", outline=(214, 224, 235), width=2)
    draw.text((x0 + 24, y0 + 18), title, fill=(28, 42, 58), font=font(24, True))

    chart_x = x0 + 72
    chart_y = y0 + 70
    chart_w = x1 - chart_x - 35
    chart_h = y1 - chart_y - 70
    xs = [float(r["target_alpha"]) for r in rows]
    ys = [float(r[metric]) for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max((ymax - ymin) * 0.15, 1e-6)
    ymin -= pad
    ymax += pad

    for i in range(5):
        yy = chart_y + chart_h - i * chart_h / 4
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=(226, 232, 238), width=1)
        draw.text((chart_x - 62, yy - 8), fmt.format(ymin + (ymax - ymin) * i / 4), fill=(92, 102, 115), font=font(13))
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=(75, 84, 96), width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(75, 84, 96), width=2)

    points = []
    for x, y in zip(xs, ys):
        px = chart_x + (x - xmin) / max(xmax - xmin, 1e-12) * chart_w
        py = chart_y + chart_h - (y - ymin) / max(ymax - ymin, 1e-12) * chart_h
        points.append((px, py))
    for a, b in zip(points, points[1:]):
        draw.line((*a, *b), fill=color, width=4)
    for (px, py), x, y in zip(points, xs, ys):
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline=(20, 30, 40), width=2)
        draw.text((px - 16, chart_y + chart_h + 12), f"{x:.2f}", fill=(55, 65, 78), font=font(13))


def plot(rows):
    img = Image.new("RGB", (1500, 950), (246, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1500, 118), fill=(18, 62, 92))
    draw.text((55, 26), "Safety Filter Parameter Sweep", fill="white", font=font(40, True))
    draw.text((58, 78), "Effect of target smoothing coefficient on BC safety layer", fill=(214, 232, 245), font=font(21))
    draw_line_panel(draw, rows, "h2_proxy_kg", (45, 155, 735, 515), "Hydrogen Proxy", (66, 135, 196), "{:.3f}")
    draw_line_panel(draw, rows, "power_mae_kw", (765, 155, 1455, 515), "Power Tracking MAE", (99, 160, 92), "{:.1f}")
    draw_line_panel(draw, rows, "start_stop_count", (45, 555, 735, 910), "Start-Stop Count", (219, 117, 58), "{:.0f}")
    draw_line_panel(draw, rows, "soc_min", (765, 555, 1455, 910), "Minimum SOC", (132, 99, 170), "{:.3f}")
    img.save(OUT_PNG)


def main():
    alphas = [0.06, 0.12, 0.18, 0.24, 0.30, 0.40]
    rows = [run_case(alpha) for alpha in alphas]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = ["target_alpha"] + [k for k in rows[0].keys() if k != "target_alpha"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    for row in rows:
        print(row)
    print(f"saved safety sweep to {OUT_CSV}")
    print(f"saved safety sweep figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
