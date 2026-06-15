# EPA Drive-Cycle Data

This folder stores official one-hertz EPA chassis dynamometer driving schedules used by the Python benchmark pipeline.

| File | Project key | Purpose |
|---|---|---|
| `la92col.txt` | `epa_la92` | Main Class 3 Heavy-Duty dynamic cycle for the 12-ton vehicle study. |
| `us06col.txt` | `epa_us06` | Aggressive high-acceleration supplemental FTP comparison cycle. |
| `uddscol.txt` | `epa_udds` | Stop-and-go urban comparison cycle. |
| `hwycol.txt` | `epa_hwfet` | Highway fuel-economy comparison cycle. |

Source page: <https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules>

The EPA files provide target speed in mph. The project loader converts speed to km/h and then computes a simplified 12-ton traction-power demand using the shared `speed_to_power` model in `python/multistack_ai/drive_cycles.py`.
