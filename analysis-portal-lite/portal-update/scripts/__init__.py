"""
Script Registry
================
Maps display names to analysis functions.
Each function: run(input_dir: str, output_dir: str) -> dict
"""

from scripts.ecsa_analysis import run as ecsa_run
from scripts.eis_analysis import run as eis_run
from scripts.h2_crossover_analysis import run as crossover_run
from scripts.polcurve_analysis import run as polcurve_run
from scripts.electrolyzer_polcurve import run as elx_polcurve_run
from scripts.fuelcell_analysis import run as fuelcell_run

SCRIPT_REGISTRY = {
    "Fuel Cell ECSA": ecsa_run,
    "EIS Analysis": eis_run,
    "H2 Crossover": crossover_run,
    "FC Polarization Curve": polcurve_run,
    "Electrolyzer Pol Curve": elx_polcurve_run,
    "Fuel Cell Full Analysis": fuelcell_run,
}
