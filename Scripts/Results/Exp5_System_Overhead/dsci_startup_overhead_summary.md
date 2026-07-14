# DSCI Startup-Mode Overhead Summary

This summary reuses existing SolutionCache training events only; no PPO run is re-executed.

## Main finding
- Cold-start PPO training costs 376.03 s on average.
- Medium warm-start reduces mean training time to 98.12 s (3.83x faster than cold mean).
- Near warm-start reduces mean training time to 19.67 s (19.12x faster than cold mean).
- Cache reuse has 0 s PPO retraining overhead because it returns the cached policy directly.

## Canonical sequence
- cold -> medium -> near: 397.05s -> 98.12s -> 15.16s.
- End-to-end startup speedup in this sequence is 26.20x.

## Measured modes
- near: runs 3, mean 19.67 s, min 15.16 s, max 23.92 s, speedup vs cold 19.12x.
- medium: runs 1, mean 98.12 s, min 98.12 s, max 98.12 s, speedup vs cold 3.83x.
- cold_warm: runs 2, mean 599.88 s, min 579.90 s, max 619.86 s, speedup vs cold 0.63x.
- cold: runs 4, mean 376.03 s, min 64.13 s, max 573.26 s, speedup vs cold 1.00x.
