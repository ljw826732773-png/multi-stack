from pathlib import Path

from scripts.generate_experiment_report import draw_pareto, read_csv, write_report


def test_report_generation_functions(tmp_path, monkeypatch):
    rows = [
        {
            "policy": "A",
            "h2_proxy_kg": "1.0",
            "power_mae_kw": "2.0",
            "start_stop_count": "3.0",
            "soc_min": "0.60",
            "final_soh_range": "0.10",
            "final_soh_var": "0.01",
            "final_mean_soh": "0.90",
        },
        {
            "policy": "B",
            "h2_proxy_kg": "0.8",
            "power_mae_kw": "4.0",
            "start_stop_count": "1.0",
            "soc_min": "0.55",
            "final_soh_range": "0.20",
            "final_soh_var": "0.02",
            "final_mean_soh": "0.85",
        },
    ]
    cycle_rows = [
        {"cycle": "urban", **rows[0]},
        {"cycle": "urban", **rows[1]},
        {"cycle": "highway", **rows[0]},
        {"cycle": "highway", **rows[1]},
    ]
    pareto_path = tmp_path / "pareto.png"
    report_path = tmp_path / "report.md"

    import scripts.generate_experiment_report as report

    monkeypatch.setattr(report, "PARETO_PATH", pareto_path)
    monkeypatch.setattr(report, "REPORT_PATH", report_path)

    draw_pareto(rows)
    write_report(rows, cycle_rows)

    assert pareto_path.exists()
    assert report_path.exists()
    assert "Experiment Report" in report_path.read_text(encoding="utf-8")


def test_read_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("policy,h2_proxy_kg\nA,1.0\n", encoding="utf-8")

    rows = read_csv(Path(path))

    assert rows == [{"policy": "A", "h2_proxy_kg": "1.0"}]
