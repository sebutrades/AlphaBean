# test_patterns.py — put this in your edgefinder/ root folder
from backend.patterns.edgefinder_patterns import (
    RubberBandScalp,
    HitchHikerScalp,
    OpeningRangeBreak,
    SecondChanceScalp,
    BackSideScalp,
    FashionablyLateScalp,
    SpencerScalp,
    GapGiveAndGo,
    TidalWaveBouncyBall,
    BreakingNewsStrategy,
    DoubleBottom,
    DoubleTop,
    BullFlag,
    BearFlag,
    HeadAndShoulders,
    get_all_detectors,
)

# Just check everything imports correctly
detectors = get_all_detectors()
print(f"✓ Successfully loaded {len(detectors)} pattern detectors:\n")
for d in detectors:
    print(f"  • {d.name}  ({d.default_bias.value})")
