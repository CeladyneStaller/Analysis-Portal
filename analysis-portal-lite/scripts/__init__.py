"""
Script Registry
================
Map display names to analysis functions.

Each function signature:
    def run(input_dir: str, output_dir: str) -> dict

Add your scripts:
    from scripts.my_script import run as my_run
    SCRIPT_REGISTRY["My Analysis"] = my_run
"""

from scripts.example import run as example_run
from scripts.ecsa_analysis import run as ecsa_run
from scripts.eis_analysis import run as eis_run
from scripts.electrolyzer_polcurve import run as electrolyzer_run
from scripts.fuelcell_analysis import run as fuelcell_run
from scripts.h2_crossover_analysis import run as h2x_run
from scripts.polcurve_analysis import run as polcurve_run

SCRIPT_REGISTRY = {
    "Fuel Cell ECSA": ecsa_run,
    "Fuel Cell EIS": eis_run,
    "Electrolyzer Analysis (Full Run)": electrolyzer_run,
    "Fuel Cell Analysis (Full Run)": fuelcell_run,
    "Fuel Cell H2X": h2x_run,
    "Fuel Cell Polarization Curves": polcurve_run,
}
