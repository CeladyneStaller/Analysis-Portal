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
    "Example Analysis": example_run,
    # "Polarization Curves": polarization_run,
    # "Durability Report":   durability_run,
}
