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

SCRIPT_REGISTRY = {
    "Fuel Cell ECSA": ecsa_analysis,
    "Fuel Cell EIS": eis_analysis,
    "Electrolyzer Analysis (Full Run)": electrolyzer_polcurve,
    "Fuel Cell Analysis (Full Run)": fuelcell_analysis,
    "Fuel Cell H2X": h2_crossover_analysis,
    "Fuel Cell Polarization Curves": polcurve_analysis,
}
