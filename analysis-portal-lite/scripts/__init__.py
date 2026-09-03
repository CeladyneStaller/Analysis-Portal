"""
Script Registry + Parameter Definitions
========================================
Maps display names to run() functions and defines the
user-configurable parameters each script accepts.

The "sample_name" field is prepended to all output filenames by the
job runner (main.py), so individual scripts don't need to handle it.
"""

import importlib

# Analysis modules are imported tolerantly. This package is imported by every
# endpoint that lists or runs a script, so a single missing or broken module
# here takes the whole portal down — dropdowns included — rather than just
# removing one entry. Two deployments of this portal carry different script
# sets, so absence is normal rather than exceptional.
MISSING_SCRIPTS = {}


def _optional(module, attr="run"):
    """The module's run(), or None if it is not present or fails to import."""
    try:
        return getattr(importlib.import_module(f"scripts.{module}"), attr)
    except Exception as exc:                       # noqa: BLE001
        MISSING_SCRIPTS[module] = f"{type(exc).__name__}: {exc}"
        return None


ecsa_run = _optional("ecsa_analysis")
eis_run = _optional("eis_analysis")
crossover_run = _optional("h2_crossover_analysis")
polcurve_run = _optional("polcurve_analysis")
elx_polcurve_run = _optional("electrolyzer_polcurve")
elx_durability_run = _optional("electrolyzer_durability")
fuelcell_run = _optional("fuelcell_analysis")
ocv_run = _optional("ocv_analysis")
activation_run = _optional("activation_analysis")
cleaning_run = _optional("electrode_cleaning_analysis")
polcurve_down_run = _optional("polcurve_analysis_down")
polcurve_hfrcompare_run = _optional("polcurve_analysis_hfr_compare")
plot_comparison_run = _optional("compare_polcurves")

SCRIPT_REGISTRY = {
    "Fuel Cell ECSA": ecsa_run,
    "EIS Analysis": eis_run,
    "H2 Crossover": crossover_run,
    "FC Polarization Curve": polcurve_run,
    "FC Polarization Curve (Downswing)": polcurve_down_run,
    "FC Polarization Curve (HFR Compare)": polcurve_hfrcompare_run,
    "OCV Analysis": ocv_run,
    "FC Activation": activation_run,
    "FC Electrode Cleaning": cleaning_run,
    "Electrolyzer Pol Curve": elx_polcurve_run,
    "Electrolyzer Durability": elx_durability_run,
    "Fuel Cell Full Analysis": fuelcell_run,
    # Internal-only — invoked via /api/compare, hidden from script dropdown
    "Plot Comparison": plot_comparison_run,
}

# ─── Short labels for filename prefixing ─────────────────────────
SCRIPT_SHORT = {
    "Fuel Cell ECSA": "ECSA",
    "EIS Analysis": "EIS",
    "H2 Crossover": "H2Xover",
    "FC Polarization Curve": "PolCurve",
    "FC Polarization Curve (Downswing)": "PolCurveDown",
    "FC Polarization Curve (HFR Compare)": "PolCurveHFRcmp",
    "OCV Analysis": "OCV",
    "FC Activation": "Activation",
    "FC Electrode Cleaning": "Cleaning",
    "Electrolyzer Pol Curve": "ElxPolCurve",
    "Electrolyzer Durability": "ElxDurability",
    "Fuel Cell Full Analysis": "FCAnalysis",
    "Plot Comparison": "Comparison",
}

# ─── Common sample_name field (inserted first for every script) ──
_SAMPLE_FIELD = {"key": "sample_name", "label": "Sample Name", "type": "text", "default": ""}

_IMAGE_FORMAT_FIELD = {"key": "image_format", "label": "Image Format", "type": "select",
    "default": "png",
    "options": [{"value": "png", "label": "PNG"},
                {"value": "svg", "label": "SVG"},
                {"value": "pdf", "label": "PDF"},
                {"value": "tiff", "label": "TIFF"},
                {"value": "none", "label": "No Images"}]}

SCRIPT_PARAMS = {
    "Fuel Cell ECSA": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "stand", "label": "Test Stand", "type": "select", "default": "",
         # The number identifies the stand; it does not change how files are
         # parsed. Blank keeps the previous behaviour of inferring the family
         # from the file extensions, so a run left on the default is not
         # labelled with a stand it may not have come from.
         "options": [{"value": "", "label": "Auto-detect"},
                     {"value": "Scribner 1", "label": "Scribner 1"},
                     {"value": "Scribner 2", "label": "Scribner 2"},
                     {"value": "FCTS 1", "label": "FCTS 1"},
                     {"value": "FCTS 2", "label": "FCTS 2"},
                     {"value": "FCTS 3", "label": "FCTS 3"},
                     {"value": "FCTS 4", "label": "FCTS 4"}]},
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "scan_rate", "label": "Scan Rate (V/s)", "type": "number",
         "default": 0.050, "step": 0.001, "min": 0.001},
        {"key": "loading", "label": "Cathode Pt Loading (mg/cm²)", "type": "number",
         "default": 0.20, "step": 0.01, "min": 0},
        {"key": "v_low", "label": "H_UPD Lower Bound (V vs RHE)", "type": "number",
         "default": 0.08, "step": 0.01, "min": 0},
        {"key": "v_high", "label": "H_UPD Upper Bound (V vs RHE)", "type": "number",
         "default": 0.40, "step": 0.01, "min": 0},
        {"key": "cycle", "label": "Cycle to Analyze", "type": "select", "default": "2",
         "options": [{"value": "2", "label": "Second"},
                     {"value": "last", "label": "Last"},
                     {"value": "first", "label": "First"},
                     {"value": "3", "label": "Third"},
                     {"value": "average", "label": "Average all"}]},
    ],
    "EIS Analysis": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "model_name", "label": "Equivalent Circuit Model", "type": "select",
         "default": "R-RC",
         "options": [{"value": "R-RC", "label": "R-RC (simple)"},
                     {"value": "R-RC-RC", "label": "R-RC-RC (two arcs)"},
                     {"value": "Randles-W", "label": "Randles + Warburg"}]},
    ],
    "H2 Crossover": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "avg_V_min", "label": "Averaging Window Low (V)", "type": "number",
         "default": 0.35, "step": 0.01, "min": 0},
        {"key": "avg_V_max", "label": "Averaging Window High (V)", "type": "number",
         "default": 0.50, "step": 0.01, "min": 0},
        {"key": "membrane_thickness", "label": "Membrane Thickness (µm, 0 = skip)",
         "type": "number", "default": 0, "step": 1, "min": 0},
    ],
    "FC Polarization Curve": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "tafel_j_min", "label": "Tafel Region j_min (A/cm²)", "type": "number",
         "default": 0.01, "step": 0.001, "min": 0},
        {"key": "tafel_j_max", "label": "Tafel Region j_max (A/cm²)", "type": "number",
         "default": 0.10, "step": 0.001, "min": 0},
    ],
    "FC Polarization Curve (Downswing)": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "tafel_j_min", "label": "Tafel Region j_min (A/cm²)", "type": "number",
         "default": 0.01, "step": 0.001, "min": 0},
        {"key": "tafel_j_max", "label": "Tafel Region j_max (A/cm²)", "type": "number",
         "default": 0.10, "step": 0.001, "min": 0},
    ],
    "FC Polarization Curve (HFR Compare)": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "tafel_j_min", "label": "Tafel Region j_min (A/cm²)", "type": "number",
         "default": 0.01, "step": 0.001, "min": 0},
        {"key": "tafel_j_max", "label": "Tafel Region j_max (A/cm²)", "type": "number",
         "default": 0.10, "step": 0.001, "min": 0},
    ],
    "OCV Analysis": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "interval_s", "label": "Resampling Interval (seconds)", "type": "number",
         "default": 60.0, "step": 1, "min": 1},
    ],
    "FC Activation": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "interval_s", "label": "Resampling Interval (seconds)", "type": "number",
         "default": 60.0, "step": 1, "min": 1},
    ],
    "FC Electrode Cleaning": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "scan_rate", "label": "Scan Rate (V/s)", "type": "number",
         "default": 0.5, "step": 0.05, "min": 0.001},
        {"key": "v_hupd_low", "label": "H_UPD low (V)", "type": "number",
         "default": 0.05, "step": 0.01, "min": 0},
        {"key": "v_hupd_high", "label": "H_UPD high (V)", "type": "number",
         "default": 0.40, "step": 0.01, "min": 0},
        {"key": "v_dl_low", "label": "DL low (V)", "type": "number",
         "default": 0.40, "step": 0.01, "min": 0},
        {"key": "v_dl_high", "label": "DL high (V)", "type": "number",
         "default": 0.50, "step": 0.01, "min": 0},
    ],
    "Electrolyzer Pol Curve": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "cell_id", "label": "Cell ID (for folder scan)", "type": "text",
         "default": "a1"},
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "T_C", "label": "Temperature (°C)", "type": "number",
         "default": 80.0, "step": 1, "min": 0},
        {"key": "p_cathode_barg", "label": "Cathode Pressure (barg)", "type": "number",
         "default": 0.0, "step": 0.1, "min": 0},
        {"key": "p_anode_barg", "label": "Anode Pressure (barg)", "type": "number",
         "default": 0.0, "step": 0.1, "min": 0},
        {"key": "eis_ref_voltage", "label": "EIS Reference Voltage (V, blank=skip)",
         "type": "number", "default": "", "step": 0.01, "min": 0},
    ],
    "Electrolyzer Durability": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 25.0, "step": 0.1, "min": 0.1},
        {"key": "eis_ref_voltage", "label": "EIS Reference Voltage (V)",
         "type": "number", "default": 1.25, "step": 0.01, "min": 0},
        {"key": "data_interval_min", "label": "Durability Data Interval (minutes, blank=all)",
         "type": "number", "default": "", "step": 1, "min": 1},
    ],
    "Fuel Cell Full Analysis": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "stand", "label": "Test Stand", "type": "select", "default": "",
         # The number identifies the stand; it does not change how files are
         # parsed. Blank keeps the previous behaviour of inferring the family
         # from the file extensions, so a run left on the default is not
         # labelled with a stand it may not have come from.
         "options": [{"value": "", "label": "Auto-detect"},
                     {"value": "Scribner 1", "label": "Scribner 1"},
                     {"value": "Scribner 2", "label": "Scribner 2"},
                     {"value": "FCTS 1", "label": "FCTS 1"},
                     {"value": "FCTS 2", "label": "FCTS 2"},
                     {"value": "FCTS 3", "label": "FCTS 3"},
                     {"value": "FCTS 4", "label": "FCTS 4"}]},
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "loading", "label": "Cathode Pt Loading (mg/cm²)", "type": "number",
         "default": 0.20, "step": 0.01, "min": 0},
        {"key": "interval_s", "label": "OCV Resampling Interval (seconds)", "type": "number",
         "default": 60.0, "step": 1, "min": 1},
    ],
}


# Entries whose module was absent are removed rather than left pointing at
# None, so callers never have to test for it. SCRIPT_SHORT and SCRIPT_PARAMS
# are trimmed to match, keeping the three dicts consistent.
SCRIPT_REGISTRY = {k: v for k, v in SCRIPT_REGISTRY.items() if v is not None}
SCRIPT_SHORT = {k: v for k, v in SCRIPT_SHORT.items() if k in SCRIPT_REGISTRY}
SCRIPT_PARAMS = {k: v for k, v in SCRIPT_PARAMS.items() if k in SCRIPT_REGISTRY}

if MISSING_SCRIPTS:
    for _m, _why in sorted(MISSING_SCRIPTS.items()):
        print(f"[scripts] not registered: {_m} ({_why})", flush=True)