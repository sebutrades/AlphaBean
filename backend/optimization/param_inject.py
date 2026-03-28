"""
backend/optimization/param_inject.py — Runtime Parameter Injection

This is how the optimizer changes strategy parameters without rewriting
every detector function. Each detector reads its tunable params from
this module instead of hardcoded values.

HOW IT WORKS:
  1. Default params are set here (matching current hardcoded values)
  2. The optimizer calls set_params("Juicer Long", {"adx_threshold": 20})
  3. The detector calls get_param("Juicer Long", "adx_threshold", default=25)
  4. It gets 20 (the override) instead of 25 (the default)
  5. After optimization, call clear_overrides() to reset

INTEGRATION:
  In each detector function, replace hardcoded values with get_param() calls.
  
  Example in _detect_juicer_long:
    BEFORE: if adx is None or adx < 25:
    AFTER:  if adx is None or adx < get_param("Juicer Long", "adx_threshold", 25):

  This is a ONE-LINE change per parameter. The detector works exactly
  the same with defaults, but the optimizer can now tune it.

USAGE:
  from backend.optimization.param_inject import get_param, set_params, clear_overrides
  
  # Normal operation (uses defaults):
  threshold = get_param("Juicer Long", "adx_threshold", 25)  # Returns 25
  
  # During optimization:
  set_params("Juicer Long", {"adx_threshold": 20})
  threshold = get_param("Juicer Long", "adx_threshold", 25)  # Returns 20
  
  clear_overrides()  # Reset all
"""

# Global override storage
_OVERRIDES: dict[str, dict[str, any]] = {}


def get_param(strategy_name: str, param_name: str, default):
    """Get a strategy parameter, checking overrides first.
    
    This is the ONLY function detectors need to call.
    Fast (dict lookup) — no performance impact.
    """
    strategy_overrides = _OVERRIDES.get(strategy_name)
    if strategy_overrides and param_name in strategy_overrides:
        return strategy_overrides[param_name]
    return default


def set_params(strategy_name: str, params: dict):
    """Set parameter overrides for a strategy (used by optimizer)."""
    _OVERRIDES[strategy_name] = params


def set_param(strategy_name: str, param_name: str, value):
    """Set a single parameter override."""
    if strategy_name not in _OVERRIDES:
        _OVERRIDES[strategy_name] = {}
    _OVERRIDES[strategy_name][param_name] = value


def clear_overrides(strategy_name: str = None):
    """Clear overrides. If strategy_name given, clear only that strategy."""
    global _OVERRIDES
    if strategy_name:
        _OVERRIDES.pop(strategy_name, None)
    else:
        _OVERRIDES = {}


def get_all_overrides() -> dict:
    """Return all current overrides (for debugging/logging)."""
    return dict(_OVERRIDES)


def has_overrides(strategy_name: str) -> bool:
    """Check if a strategy has any active overrides."""
    return strategy_name in _OVERRIDES and len(_OVERRIDES[strategy_name]) > 0