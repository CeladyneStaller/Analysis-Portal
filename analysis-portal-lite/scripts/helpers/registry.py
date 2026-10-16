"""
Analysis and metric registry
============================

One table that says what each analysis is and what its numbers mean: which
summary fields feed an index value, the unit it is stored in, the unit it is
shown in, and which direction is better.

The record layer builds the index from it — `record.KEY_VALUES` and
`record.KEY_VALUE_UNITS` are derived here rather than written out a second
time. /api/view/runs sends the same table to the views (as `registry`), to
replace the front-end tables that held this and had drifted apart (Trends
labelled HFR mΩ·cm² while Recall said Ω·cm²; sort direction was keyed by bare
metric name). Recall reads it; Comparisons and Trends move over next.

Direction belongs to (analysis, metric), not to the metric name alone:
V @ 1 A/cm² is higher-is-better for a fuel cell and lower-is-better for an
electrolyzer, so the same name appears under both with opposite signs.

Scope: every value that reaches the index, plus the electrolyzer values that
stay in the detail record. Fuel-cell values seen only in run detail (Peak P,
R_ohm, ...) are not listed: run detail names them with the page's alias table
and takes their units from the readouts printed on the plots.

Pure data and lookups. Nothing here imports from the rest of the package, so
anything can import it.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple


def _metric(name: str, sources: List[str], unit: str, *,
            better: int = 0, show: Optional[str] = None, scale: float = 1,
            index: bool = True) -> Dict[str, Any]:
    """One metric.

    name     canonical name, as the index and the views carry it
    sources  summary fields or readout keys that supply it, tried in order
    unit     unit of the stored number
    show     unit it is displayed in; `scale` converts stored to shown
    better   +1 higher is better, -1 lower is better, 0 not ranked
    index    promoted to the index; False keeps it in the detail record only
    """
    return {'name': name, 'sources': list(sources), 'unit': unit,
            'show': show or unit, 'scale': scale, 'better': better,
            'index': index}


# Families. A run belongs to the family of its analyses; `scripts` names the
# scripts whose runs belong to a family even when their analyses say
# otherwise — the electrolyzer runs of Jul 27 before 16:15 were filed under
# fuel-cell analyses, and the rebuild leaves them that way.
FAMILIES: Dict[str, Dict[str, Any]] = {
    'fc': {'label': 'Fuel cell'},
    'elx': {'label': 'Electrolyzer',
            'scripts': ['Electrolyzer Pol Curve', 'Electrolyzer Durability']},
}

# Analyses, each in one family.
#
# step_label  what a step means for this analysis, where it is not a protocol
#             step: an electrolyzer step is the cell id ('a1')
# stepless    one unit per run with no step at all
# main_plots  plot types that make up the analysis at a glance, in display
#             order. They are kept ahead of everything else when a record has
#             to shed plots to fit the transport limit.
# other_plots the analysis's remaining plot types, with the name of the group
#             Recall folds each into below the main plots. A type not listed
#             folds under 'other'.
# bookkeeping summary fields that are not results — inputs, file counts, or a
#             value already shown under another name — which run detail
#             leaves out of the values beneath the plots.
# summary_conditions
#             summary fields that are run conditions, by the condition key
#             they are: run detail shows them with the conditions, filling in
#             what the plots do not declare (older runs' plots did not).
ANALYSES: Dict[str, Dict[str, Any]] = {
    'polcurve':   {'label': 'Polarization curve', 'family': 'fc', 'letter': 'P'},
    'eis':        {'label': 'EIS', 'family': 'fc', 'letter': 'E'},
    'crossover':  {'label': 'H₂ crossover', 'family': 'fc', 'letter': 'X'},
    'ecsa':       {'label': 'ECSA', 'family': 'fc', 'letter': 'A'},
    'ocv':        {'label': 'OCV', 'family': 'fc', 'letter': 'O'},
    'activation': {'label': 'Activation', 'family': 'fc', 'letter': 'B'},
    'cleaning':   {'label': 'Electrode cleaning', 'family': 'fc', 'letter': 'C'},
    'CLR':        {'label': 'Catalyst layer resistance', 'family': 'fc'},
    # FC Durability: one whole-test unit per sample, holding the test's
    # summary values. Its plots are shown by run detail as the whole test,
    # so none is singled out as a main plot.
    'fc_durability': {'label': 'FC durability', 'family': 'fc', 'letter': 'F',
                      'stepless': True},
    'elx_polcurve': {
        'label': 'Electrolyzer pol curve', 'family': 'elx', 'letter': 'P',
        'step_label': 'Cell',
        'main_plots': ['elx_polcurve', 'elx_j_vs_cycle', 'elx_v_vs_cycle',
                       'elx_j_and_losses_vs_cycle', 'elx_v_and_losses_vs_cycle',
                       'elx_model_fit'],
        'other_plots': {'elx_ir_correction': 'iR correction'},
    },
    'elx_eis': {
        'label': 'Electrolyzer EIS', 'family': 'elx', 'letter': 'Z',
        'step_label': 'Cell',
        'main_plots': ['elx_nyquist', 'elx_eis_for_ir'],
        'other_plots': {
            'elx_eis_fit': 'per-spectrum fits',
            'elx_clr_analysis': 'catalyst-layer resistance (CLR)',
            'elx_clr_model_fit': 'catalyst-layer resistance (CLR)',
            'elx_eis_losses_vs_cycle': 'EIS losses vs cycle',
        },
        # The same R0 as HFR, in mΩ·cm².
        'bookkeeping': ['R0_asr_mohm_cm2'],
    },
    'durability': {
        'label': 'Durability', 'family': 'elx', 'letter': 'D', 'stepless': True,
        'main_plots': ['durability_voltage', 'durability_degradation',
                       'durability_hfr', 'durability_polcurve_evolution'],
        'other_plots': {'durability_asr_evolution': 'ASR evolution',
                        'durability_loss_evolution': 'loss evolution'},
        'bookkeeping': ['geo_area', 'n_eis', 'n_polcurves'],
        'summary_conditions': {'j_hold': 'j_hold_Acm2'},
    },
}

# Area resistances are stored in Ω·cm² under the shared 'HFR' name and shown
# in mΩ·cm² everywhere. The durability endpoints are separate names that the
# script reports in mΩ·cm² natively, so they are stored that way.
_HFR_SHOWN = dict(show='mΩ·cm²', scale=1000)

METRICS: Dict[str, List[Dict[str, Any]]] = {
    'polcurve': [
        _metric('OCV', ['OCV', 'OCV (V)'], 'V', better=1),
        _metric('V @ 1 A/cm²',
                ['V @ 1 A/cm²', 'V@1A (V)', 'V_at_1Acm2', 'V at 1 A/cm2'],
                'V', better=1),
        # Current density at fixed operating voltages. Sparse by design — a key
        # is absent when the curve did not reach that voltage, never estimated.
        _metric('j @ 0.7 V', ['j_at_0.7V', 'j @ 0.7 V'], 'A/cm²', better=1),
        _metric('j @ 0.65 V', ['j_at_0.65V', 'j @ 0.65 V'], 'A/cm²', better=1),
        _metric('j @ 0.6 V', ['j_at_0.6V', 'j @ 0.6 V'], 'A/cm²', better=1),
        _metric('j @ 0.5 V', ['j_at_0.5V', 'j @ 0.5 V'], 'A/cm²', better=1),
        _metric('j @ 0.4 V', ['j_at_0.4V', 'j @ 0.4 V'], 'A/cm²', better=1),
    ],
    'eis': [
        _metric('HFR', ['HFR', 'Mean HFR', 'HFR_mean', 'Mean HFR (mOhm·cm2)'],
                'Ω·cm²', better=-1, **_HFR_SHOWN),
        # The fitted circuit's resistances sit beside HFR in run detail, so
        # they are shown in the same unit.
        _metric('R_ohm', ['R_ohm'], 'Ω·cm²', index=False, **_HFR_SHOWN),
        _metric('R_ct', ['R_ct'], 'Ω·cm²', index=False, **_HFR_SHOWN),
        _metric('R_mt', ['R_mt'], 'Ω·cm²', index=False, **_HFR_SHOWN),
    ],
    'crossover': [
        _metric('|j_xover|', ['|j_xover|', 'j_xover', 'j_xover_mA_cm2',
                              'j_xover (mA/cm2)'], 'mA/cm²', better=-1),
    ],
    'ecsa': [
        _metric('Average ECSA', ['Average ECSA', 'average_ECSA_m2_per_g',
                                 'Avg ECSA (m2/g)'], 'm²/g', better=1),
    ],
    # One step-less unit per FC Durability test. Hours to death exists only
    # for a sample that died: absent means alive, censored at Hours tested.
    # A test that cycles RH also counts its lifetime in RH cycles, the same
    # way. How long a test has run is neither good nor bad — a long test that
    # died is not the best sample — so those two are not ranked.
    'fc_durability': [
        _metric('Hours tested', ['Hours tested'], 'h'),
        _metric('Hours to death', ['Hours to death'], 'h', better=1),
        _metric('OCV initial', ['OCV initial'], 'V', better=1),
        _metric('OCV decay rate', ['OCV decay rate'], 'μV/h', better=-1),
        _metric('OCV loss', ['OCV loss'], '%', better=-1),
        _metric('j_xover initial', ['j_xover initial'], 'mA/cm²', better=-1),
        _metric('j_xover final', ['j_xover final'], 'mA/cm²', better=-1),
        # A membrane short: a higher resistance is a lesser short.
        _metric('R_short final', ['R_short final'], 'Ω·cm²', better=1),
        _metric('RH cycles tested', ['RH cycles tested'], 'cycles'),
        _metric('RH cycles to death', ['RH cycles to death'], 'cycles',
                better=1),
    ],

    # Electrolyzer polarization curve. Voltages are from the last complete
    # cycle, the one the model fit is given. 'V @ 1 A/cm²' reuses the
    # fuel-cell name for the same physical quantity; the analysis keeps the
    # two apart.
    'elx_polcurve': [
        _metric('V @ 1 A/cm²', ['V @ 1 A/cm²', 'V_at_1Acm2', 'V at 1 A/cm2'],
                'V', better=-1),
        _metric('V @ 2 A/cm²', ['V @ 2 A/cm²', 'V_at_2Acm2', 'V at 2 A/cm2'],
                'V', better=-1),
        # Change in cell voltage at 2 A/cm² from the first cycle to the last:
        # cycling stability as one number. Needs two sweeps of the same
        # control mode that both reach 2 A/cm².
        _metric('ΔV @ 2 A/cm²', ['ΔV @ 2 A/cm²', 'dV_at_2Acm2_mV'], 'mV',
                better=-1),
        # Fit-derived, so kept out of the index until the model is fixed.
        # ASR_total is pinned to EIS when EIS files are present and free
        # otherwise, so two runs' values need not mean the same thing; the
        # model applies RT/αnF to log₁₀, which puts the Tafel slope off by
        # 2.303; E_rev is computed from the form, not measured.
        _metric('ASR_total', ['ASR_total', 'ASR_total (mOhm.cm2)', 'ASR total'],
                'mΩ·cm²', index=False),
        _metric('Tafel slope (anode)', ['Tafel slope (anode)',
                                        'tafel_anode_mVdec', 'Anode Tafel slope'],
                'mV/dec', index=False),
        _metric('E_rev', ['E_rev', 'E_rev (V)'], 'V', index=False),
        _metric('Fit RMSE', ['Fit RMSE', 'rmse_mV', 'RMSE'], 'mV', index=False),
    ],
    # R0 from one fixed spectrum, normally the reference spectrum of the last
    # cycle. The script reports it in mΩ·cm² and converts it to the shared
    # 'HFR' name's Ω·cm² in its summary.
    'elx_eis': [
        _metric('HFR', ['HFR', 'HFR (Ohm.cm2)', 'R0_ohm_cm2'], 'Ω·cm²',
                better=-1, **_HFR_SHOWN),
        # Which spectrum the HFR was read from, by its DC voltage.
        _metric('HFR spectrum DC voltage', ['HFR_dc_V'], 'V', index=False),
    ],
    'durability': [
        _metric('Degradation rate', ['Degradation rate', 'rate_uV_hr',
                                     'Degradation rate (uV/hr)'],
                'μV/h', better=-1),
        _metric('V_initial', ['V_initial', 'V initial'], 'V', better=-1),
        _metric('V_final', ['V_final', 'V final'], 'V', better=-1),
        _metric('HFR_initial', ['HFR_initial', 'HFR initial'], 'mΩ·cm²',
                better=-1),
        _metric('HFR_final', ['HFR_final', 'HFR final'], 'mΩ·cm²', better=-1),
        _metric('Duration', ['Duration', 'duration_hr', 'Duration (hr)'], 'h'),
        # The degradation rate as a share of the starting voltage, and how
        # well the straight line it comes from fits.
        _metric('Relative rate', ['rate_Vpct_khr'], '%/kh', index=False),
        _metric('Fit R²', ['fit_R2'], '', index=False),
    ],
}


# ─────────────────────────────────────────────────────────────────────
#  Lookups
# ─────────────────────────────────────────────────────────────────────

def family(analysis: Optional[str]) -> Optional[str]:
    """'fc', 'elx', or None for an analysis the registry does not know."""
    return (ANALYSES.get(analysis or '') or {}).get('family')


def metric(analysis: str, name: str) -> Optional[Dict[str, Any]]:
    """The registry entry for one (analysis, metric), or None."""
    for m in METRICS.get(analysis, []):
        if m['name'] == name:
            return m
    return None


_MAIN_PLOTS = frozenset(pt for a in ANALYSES.values()
                        for pt in a.get('main_plots', []))


def is_main_plot(plot_type: Optional[str]) -> bool:
    """Whether a plot type is one of its analysis's main plots."""
    return (plot_type or '') in _MAIN_PLOTS


def main_plots(analysis: Optional[str]) -> List[str]:
    """An analysis's main plot types in display order, [] if it lists none."""
    return list((ANALYSES.get(analysis or '') or {}).get('main_plots') or [])


def key_value_sources() -> Dict[str, List[Tuple[str, List[str]]]]:
    """analysis -> [(canonical name, source fields)] for the indexed metrics.

    The shape `record.KEY_VALUES` has always had, so the record layer reads it
    unchanged.
    """
    return {a: [(m['name'], list(m['sources'])) for m in ms if m['index']]
            for a, ms in METRICS.items()
            if any(m['index'] for m in ms)}


def key_value_units() -> Dict[str, str]:
    """Canonical name -> stored unit, for every indexed metric.

    Keyed by name alone, which is only sound while a name means one stored
    unit wherever it appears. That is checked here rather than assumed: a
    name reused under a different unit raises at import.
    """
    out: Dict[str, str] = {}
    for a, ms in METRICS.items():
        for m in ms:
            if not m['index']:
                continue
            seen = out.setdefault(m['name'], m['unit'])
            if seen != m['unit']:
                raise ValueError(
                    f"registry: {m['name']!r} is stored in {seen} elsewhere but "
                    f"in {m['unit']} under {a!r}")
    return out


def as_json() -> Dict[str, Any]:
    """The whole registry, for sending to the views.

    A copy, so a caller that annotates what it is about to send cannot change
    the table every later lookup reads.
    """
    return copy.deepcopy({'families': FAMILIES, 'analyses': ANALYSES,
                          'metrics': METRICS})


def aliases(analysis_ids, bucket_of) -> Dict[str, str]:
    """Stored analysis ids the registry does not list -> the analysis they are.

    Electrolyzer runs stored between Aug 4 and the record fix filed each plot
    type as a bucket of its own (elx_model_fit, elx_nyquist, ...), and their
    index entries name those buckets until the rebuild re-files them. Mapping
    an id through `bucket_of` — the record layer's `plot_bucket`, the rule
    that re-files them — says which analysis each belongs to, so the views can
    letter and group such an entry correctly in the meantime. Passed in rather
    than imported because this module imports nothing from the package.
    """
    out: Dict[str, str] = {}
    for a in analysis_ids:
        if not a or a in ANALYSES:
            continue
        target = bucket_of(a)
        if target in ANALYSES and target != a:
            out[a] = target
    return out
