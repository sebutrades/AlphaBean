"""
run_backtest_optimized.py — Backtest ONLY the 8 optimized strategies.

Loads optimized params, patches the detector map to only include
the strategies we have results for, then runs the full backtest.

Usage:
  python run_backtest_optimized.py --from-cache --days 160 --daily
"""
import json
from pathlib import Path

# Load optimized params BEFORE importing classifier
from backend.optimization.param_inject import set_params

params_file = Path("cache/optimized_params.json")
if params_file.exists():
    data = json.loads(params_file.read_text())
    for name, params in data.items():
        set_params(name, params)
    print(f"  Loaded optimized params for {len(data)} strategies")
else:
    print("  ✗ No optimized params found. Run optimization first.")
    exit(1)

# Now import classifier and patch the detector map
from backend.patterns.classifier import _DETECTOR_MAP

OPTIMIZED_ONLY = list(data.keys())
print(f"  Strategies: {', '.join(OPTIMIZED_ONLY)}")

# Remove everything except the optimized strategies
all_names = list(_DETECTOR_MAP.keys())
for name in all_names:
    if name not in OPTIMIZED_ONLY:
        del _DETECTOR_MAP[name]

print(f"  Active detectors: {len(_DETECTOR_MAP)}")

# Run the normal backtest
from run_backtest import main
main()