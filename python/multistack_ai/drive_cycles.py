from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EPA_CYCLE_DIR = ROOT / "data" / "epa_cycles"
MPH_TO_KPH = 1.609344

EPA_CYCLES = {
    "epa_la92": {
        "file": "la92col.txt",
        "label": "EPA LA92 HD",
        "description": "EPA Class 3 Heavy-Duty LA92 chassis dynamometer schedule.",
    },
    "epa_us06": {
        "file": "us06col.txt",
        "label": "EPA US06",
        "description": "EPA high-acceleration supplemental FTP schedule.",
    },
    "epa_udds": {
        "file": "uddscol.txt",
        "label": "EPA UDDS",
        "description": "EPA stop-and-go urban dynamometer driving schedule.",
    },
    "epa_hwfet": {
        "file": "hwycol.txt",
        "label": "EPA HWFET",
        "description": "EPA highway fuel-economy driving schedule.",
    },
}

ALIASES = {
    "la92": "epa_la92",
    "class3_hd": "epa_la92",
    "heavy_duty": "epa_la92",
    "us06": "epa_us06",
    "udds": "epa_udds",
    "hwfet": "epa_hwfet",
}


def _interp_profile(points: list[tuple[float, float]], dt: float = 1.0) -> np.ndarray:
    times = np.array([p[0] for p in points], dtype=float)
    speeds = np.array([p[1] for p in points], dtype=float)
    grid = np.arange(0.0, times[-1] + dt, dt)
    return np.interp(grid, times, speeds)


def _parse_float_pair(line: str) -> tuple[float, float] | None:
    clean = line.replace('"', " ").replace(",", " ")
    parts = clean.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def epa_speed_profile(name: str, dt: float = 1.0) -> np.ndarray:
    """Load an EPA one-hertz speed trace and return speed in km/h."""

    key = ALIASES.get(name.lower(), name.lower())
    if key not in EPA_CYCLES:
        raise ValueError(f"unknown EPA cycle: {name}")
    path = EPA_CYCLE_DIR / EPA_CYCLES[key]["file"]
    if not path.exists():
        raise FileNotFoundError(f"missing EPA drive-cycle file: {path}")

    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="ignore")

    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        pair = _parse_float_pair(line)
        if pair is not None:
            rows.append(pair)
    if len(rows) < 2:
        raise ValueError(f"EPA drive-cycle file has no usable time-speed data: {path}")

    times = np.array([row[0] for row in rows], dtype=float)
    speeds_mph = np.array([row[1] for row in rows], dtype=float)
    order = np.argsort(times)
    times = times[order]
    speeds_mph = speeds_mph[order]
    if dt == 1.0 and np.allclose(np.diff(times), 1.0):
        return speeds_mph * MPH_TO_KPH

    grid = np.arange(times[0], times[-1] + dt, dt)
    return np.interp(grid, times, speeds_mph) * MPH_TO_KPH


def urban_speed_profile(dt: float = 1.0) -> np.ndarray:
    """Representative stop-and-go city cycle in km/h.

    This is not a regulatory UDDS trace. It is a compact synthetic profile
    designed for reproducible EMS experiments when external cycle files are
    unavailable.
    """

    points = [
        (0, 0), (25, 28), (50, 0), (80, 35), (120, 45), (150, 0),
        (180, 25), (220, 58), (270, 20), (300, 0), (340, 42), (390, 0),
        (430, 55), (480, 30), (530, 0), (560, 38), (620, 46), (680, 0),
        (720, 50), (790, 34), (850, 0), (900, 44), (980, 18), (1030, 0),
        (1090, 48), (1160, 36), (1200, 0),
    ]
    return _interp_profile(points, dt)


def highway_speed_profile(dt: float = 1.0) -> np.ndarray:
    """Representative highway cycle in km/h."""

    points = [
        (0, 0), (60, 70), (130, 82), (220, 78), (300, 92), (420, 95),
        (520, 88), (650, 86), (760, 96), (860, 84), (980, 90), (1100, 72),
        (1200, 0),
    ]
    return _interp_profile(points, dt)


def mixed_speed_profile(dt: float = 1.0) -> np.ndarray:
    """Urban-to-highway mixed profile in km/h."""

    urban = urban_speed_profile(dt)[:600]
    highway = highway_speed_profile(dt)[:600]
    return np.concatenate([urban, highway])


def speed_to_power(speed_kph: np.ndarray, mass_kg: float = 12000.0, dt: float = 1.0) -> np.ndarray:
    """Convert a speed trace to a simple traction-power demand profile.

    Positive values represent traction demand and negative values represent
    regenerative braking availability. The constants are intentionally compact
    and transparent rather than vehicle-calibration-grade.
    """

    v = np.asarray(speed_kph, dtype=float) / 3.6
    acc = np.gradient(v, dt)
    rho = 1.225
    cd_a = 5.8
    c_rr = 0.0075
    g = 9.81
    aero = 0.5 * rho * cd_a * v**3
    rolling = mass_kg * g * c_rr * v
    inertial = mass_kg * acc * v
    power_kw = (aero + rolling + inertial) / 1000.0
    traction = np.where(power_kw >= 0.0, power_kw / 0.90, power_kw * 0.55)
    return np.clip(traction, -120.0, 280.0)


def make_cycle_demand(name: str, dt: float = 1.0) -> np.ndarray:
    key = name.lower()
    if key in EPA_CYCLES or key in ALIASES:
        speed = epa_speed_profile(key, dt)
    elif key == "epa_mixed":
        speed = np.concatenate(
            [
                epa_speed_profile("epa_la92", dt)[:900],
                epa_speed_profile("epa_hwfet", dt),
                epa_speed_profile("epa_us06", dt),
            ]
        )
    elif key in {"urban", "city"}:
        speed = urban_speed_profile(dt)
    elif key in {"highway", "hwfet"}:
        speed = highway_speed_profile(dt)
    elif key in {"mixed", "combined"}:
        speed = mixed_speed_profile(dt)
    else:
        raise ValueError(f"unknown cycle: {name}")
    return speed_to_power(speed, dt=dt)


def available_cycles() -> list[str]:
    return ["epa_la92", "epa_us06", "epa_udds", "epa_hwfet", "epa_mixed"]


def cycle_label(name: str) -> str:
    key = ALIASES.get(name.lower(), name.lower())
    if key == "epa_mixed":
        return "EPA Mixed"
    if key in EPA_CYCLES:
        return EPA_CYCLES[key]["label"]
    return name.replace("_", " ").title()
