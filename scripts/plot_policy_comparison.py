import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "policy_comparison.csv"
OUT_PATH = ROOT / "results" / "policy_comparison.png"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_rows():
    with CSV_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw_bar_group(draw, rows, metric, title, x, y, w, h, color, lower_is_better=True):
    title_font = font(26, True)
    axis_font = font(18)
    label_font = font(16)
    draw.text((x, y), title, fill=(25, 38, 55), font=title_font)
    chart_y = y + 52
    chart_h = h - 95
    chart_x = x + 58
    chart_w = w - 75
    values = [float(r[metric]) for r in rows]
    max_v = max(values) * 1.12 if max(values) > 0 else 1.0
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=(70, 80, 90), width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(70, 80, 90), width=2)

    for i in range(5):
        yy = chart_y + chart_h - i * chart_h / 4
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=(222, 228, 235), width=1)
        draw.text((chart_x - 52, yy - 10), f"{max_v*i/4:.2f}", fill=(90, 95, 105), font=label_font)

    gap = 25
    bar_w = (chart_w - gap * (len(rows) + 1)) / len(rows)
    for i, (row, value) in enumerate(zip(rows, values)):
        bx = chart_x + gap + i * (bar_w + gap)
        bh = value / max_v * chart_h
        by = chart_y + chart_h - bh
        draw.rounded_rectangle((bx, by, bx + bar_w, chart_y + chart_h), radius=7, fill=color)
        draw.text((bx + bar_w / 2 - 28, by - 25), f"{value:.2f}", fill=(25, 38, 55), font=label_font)
        if row["policy"] == "HC-MPC-style Expert":
            name = "HC-MPC\nExpert"
        elif row["policy"] == "BC Neural Policy":
            name = "BC Neural\nPolicy"
        else:
            name = row["policy"]
        lines = name.split("\n")
        for j, line in enumerate(lines):
            tw = draw.textlength(line, font=axis_font)
            draw.text((bx + bar_w / 2 - tw / 2, chart_y + chart_h + 8 + 20*j), line, fill=(55, 65, 78), font=axis_font)

    hint = "lower is better" if lower_is_better else "higher is better"
    draw.text((x + w - 155, y + 8), hint, fill=(110, 120, 130), font=label_font)


def main():
    rows = read_rows()
    img = Image.new("RGB", (1600, 980), (248, 251, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1600, 120), fill=(18, 55, 89))
    draw.text((55, 28), "Multi-Stack Fuel Cell EMS: Initial AI Benchmark", fill="white", font=font(42, True))
    draw.text((58, 82), "Rule strategies vs. HC-MPC-style expert vs. behavior cloning neural policy", fill=(210, 230, 248), font=font(22))

    card_fill = (255, 255, 255)
    for box in [(45, 155, 765, 530), (835, 155, 1555, 530), (45, 585, 765, 940), (835, 585, 1555, 940)]:
        draw.rounded_rectangle(box, radius=20, fill=card_fill, outline=(210, 222, 235), width=2)

    draw_bar_group(draw, rows, "h2_proxy_kg", "Hydrogen Consumption Proxy", 85, 185, 650, 315, (42, 123, 186), True)
    draw_bar_group(draw, rows, "start_stop_count", "Start-Stop Count", 875, 185, 650, 315, (224, 115, 48), True)
    draw_bar_group(draw, rows, "power_mae_kw", "Power Tracking MAE", 85, 615, 650, 300, (99, 160, 92), True)
    draw_bar_group(draw, rows, "soc_min", "Minimum SOC", 875, 615, 650, 300, (125, 93, 170), False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)
    print(f"saved benchmark figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
