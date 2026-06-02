# Core modules

The `core/` directory contains the released runtime-defense modules used by RISA.

- `source_aware_action_router.py`
  Action-side authorization and routing checks.
- `observation_quarantine.py`
  Tool-output cleaning and contamination handling.
- `final_answer_safeguard.py`
  Final-response release control.
- `executor.py`
  Main guarded executor that connects the runtime checks.
- `guard_prompts.py`
  Slow-guard prompt templates.
- `json_utils.py`
  Robust parsing helpers for guard outputs.
- `risk_indicators.py`
  Risk-signal and truncation helpers.
- `action_signal_heuristics.py`
  Lightweight action-level heuristics.
