import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "policy_comparison.csv"
OUT_PATH = ROOT / "results" / "policy_comparison.png"
PALETTE = [(82, 138, 191), (226, 120, 54), (52, 149, 101), (132, 99, 170), (213, 88, 93)]


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


def short_name(name: str) -> str:
    mapping = {
        "HC-MPC-style Expert": "HC-MPC\nExpert",
        "BC Neural Policy": "BC Neural\nPolicy",
        "Safety-Filtered BC": "Safety\nFiltered BC",
        "DAgger Policy": "DAgger\nPolicy",
        "Safety-Filtered DAgger": "Safety\nDAgger",
        "GRU Sequence BC": "GRU\nSequence BC",
        "Safety-Filtered GRU": "Safety\nGRU",
    }
    return mapping.get(name, name)

def draw_bar_group(draw, rows, metric, title, x, y, w, h, color, lower_is_better=True):
    title_font = font(24, True)
    axis_font = font(15)
    label_font = font(14)
    draw.text((x, y), title, fill=(25, 38, 55), font=title_font)
    chart_y = y + 48
    chart_h = h - 104
    chart_x = x + 58
    chart_w = w - 75
    values = [float(r[metric]) for r in rows]
    max_v = max(values) * 1.12 if max(values) > 0 else 1.0
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=(70, 80, 90), width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(70, 80, 90), width=2)

    for i in range(5):
        yy = chart_y + chart_h - i * chart_h / 4
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=(222, 228, 235), width=1)
        draw.text((chart_x - 52, yy - 9), f"{max_v*i/4:.2f}", fill=(90, 95, 105), font=label_font)

    gap = 18
    bar_w = (chart_w - gap * (len(rows) + 1)) / len(rows)
    for i, (row, value) in enumerate(zip(rows, values)):
        bx = chart_x + gap + i * (bar_w + gap)
        bh = value / max_v * chart_h
        by = chart_y + chart_h - bh
        draw.rounded_rectangle((bx, by, bx + bar_w, chart_y + chart_h), radius=6, fill=color)
        draw.text((bx + bar_w / 2 - 23, by - 22), f"{value:.2f}", fill=(25, 38, 55), font=label_font)
        for j, line in enumerate(short_name(row["policy"]).splitlines()):
            tw = draw.textlength(line, font=axis_font)
            draw.text((bx + bar_w / 2 - tw / 2, chart_y + chart_h + 8 + 17*j), line, fill=(55, 65, 78), font=axis_font)

    hint = "lower is better" if lower_is_better else "higher is better"
    draw.text((x + w - 145, y + 7), hint, fill=(110, 120, 130), font=label_font)


def main():
    rows = read_rows()
    img = Image.new("RGB", (1800, 1040), (248, 251, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1800, 120), fill=(18, 55, 89))
    draw.text((55, 28), "Multi-Stack Fuel Cell EMS: AI Benchmark", fill="white", font=font(42, True))
    draw.text((58, 82), "Rule strategies, HC-MPC-style expert, behavior cloning and safety-filtered neural policy", fill=(210, 230, 248), font=font(22))

    for box in [(45, 155, 865, 545), (935, 155, 1755, 545), (45, 600, 865, 995), (935, 600, 1755, 995)]:
        draw.rounded_rectangle(box, radius=20, fill=(255, 255, 255), outline=(210, 222, 235), width=2)

    draw_bar_group(draw, rows, "h2_proxy_kg", "Hydrogen Consumption Proxy", 85, 185, 750, 330, (42, 123, 186), True)
    draw_bar_group(draw, rows, "start_stop_count", "Start-Stop Count", 975, 185, 750, 330, (224, 115, 48), True)
    draw_bar_group(draw, rows, "power_mae_kw", "Power Tracking MAE", 85, 630, 750, 330, (99, 160, 92), True)
    draw_bar_group(draw, rows, "soc_min", "Minimum SOC", 975, 630, 750, 330, (125, 93, 170), False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)
    print(f"saved benchmark figure to {OUT_PATH}")


if __name__ == "__main__":
    main()


