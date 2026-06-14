import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import HealthAwareExpert
from multistack_ai.evaluate import rollout

OUT_CSV = ROOT / "results" / "sensitivity_study.csv"
OUT_PNG = ROOT / "results" / "sensitivity_study.png"


def aggregate(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def run_case(case_name, **kwargs):
    policy = HealthAwareExpert(**kwargs)
    summary = aggregate(rollout(policy, episodes=4, seed=2042))
    summary.update({"case": case_name, **kwargs})
    return summary


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_metric(draw, rows, metric, box, title, color):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=(255, 255, 255), outline=(214, 224, 235), width=2)
    draw.text((x0 + 28, y0 + 20), title, fill=(28, 42, 58), font=font(26, True))
    chart_x = x0 + 78
    chart_y = y0 + 75
    chart_w = x1 - chart_x - 28
    chart_h = y1 - chart_y - 90
    vals = [float(r[metric]) for r in rows]
    max_v = max(vals) * 1.15 if max(vals) > 0 else 1.0
    min_v = min(vals)
    for i in range(5):
        yy = chart_y + chart_h - i * chart_h / 4
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=(226, 232, 238), width=1)
        draw.text((chart_x - 62, yy - 9), f"{max_v*i/4:.2f}", fill=(92, 102, 115), font=font(15))
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=(75, 84, 96), width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(75, 84, 96), width=2)
    gap = 18
    bw = (chart_w - gap * (len(rows) + 1)) / len(rows)
    for i, (row, val) in enumerate(zip(rows, vals)):
        bx = chart_x + gap + i * (bw + gap)
        bh = val / max_v * chart_h
        by = chart_y + chart_h - bh
        fill = color if val != min_v else (34, 139, 92)
        draw.rounded_rectangle((bx, by, bx + bw, chart_y + chart_h), radius=6, fill=fill)
        draw.text((bx - 8, by - 23), f"{val:.2f}", fill=(35, 45, 58), font=font(15))
        label = row["case"].replace("_", "\n")
        for j, line in enumerate(label.split("\n")):
            tw = draw.textlength(line, font=font(14))
            draw.text((bx + bw / 2 - tw / 2, chart_y + chart_h + 10 + j * 17), line, fill=(55, 65, 78), font=font(14))


def plot(rows):
    img = Image.new("RGB", (1500, 950), (246, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1500, 118), fill=(20, 64, 96))
    draw.text((55, 26), "HC-MPC-style Expert Sensitivity Study", fill="white", font=font(40, True))
    draw.text((58, 78), "alpha, SOC feedback gain and health allocation exponent sweep", fill=(214, 232, 245), font=font(21))
    draw_metric(draw, rows, "h2_proxy_kg", (45, 155, 735, 515), "Hydrogen Proxy", (66, 135, 196))
    draw_metric(draw, rows, "start_stop_count", (765, 155, 1455, 515), "Start-Stop Count", (219, 117, 58))
    draw_metric(draw, rows, "power_mae_kw", (45, 555, 735, 910), "Power Tracking MAE", (111, 154, 90))
    draw_metric(draw, rows, "soc_min", (765, 555, 1455, 910), "Minimum SOC", (132, 99, 170))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PNG)


def main():
    cases = [
        ("base", dict(alpha=0.06, soc_gain=180.0, health_power=2.2)),
        ("slow_filter", dict(alpha=0.03, soc_gain=180.0, health_power=2.2)),
        ("fast_filter", dict(alpha=0.12, soc_gain=180.0, health_power=2.2)),
        ("weak_soc", dict(alpha=0.06, soc_gain=100.0, health_power=2.2)),
        ("strong_soc", dict(alpha=0.06, soc_gain=260.0, health_power=2.2)),
        ("mild_health", dict(alpha=0.06, soc_gain=180.0, health_power=1.2)),
        ("strong_health", dict(alpha=0.06, soc_gain=180.0, health_power=3.4)),
    ]
    rows = [run_case(name, **params) for name, params in cases]
    keys = ["case", "alpha", "soc_gain", "health_power"] + [k for k in rows[0].keys() if k not in {"case", "alpha", "soc_gain", "health_power"}]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    for row in rows:
        print(row)
    print(f"saved sensitivity csv to {OUT_CSV}")
    print(f"saved sensitivity figure to {OUT_PNG}")


if __name__ == "__main__":
    main()