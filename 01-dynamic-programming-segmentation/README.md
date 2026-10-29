# Optimal signal segmentation

A dynamic-programming implementation for partitioning signals into piecewise-constant segments. The objective combines within-segment reconstruction error with a penalty on the number of segments.

## Implementation

- Prefix sums and sums of squares accelerate segment-cost computation.
- Dynamic programming searches over segment boundaries with a configurable maximum length.
- Single-channel and multichannel implementations share the same objective structure.
- [`time_analysis.py`](time_analysis.py) generates synthetic cases and studies runtime versus input length and segment constraints.

**Code:** [`segment.py`](segment.py), [`time_analysis.py`](time_analysis.py)

## Running

```bash
python segment.py --filepath signal.tsv --penalty 0.15 --max_len 100
```

A compatible tab-separated signal is required. For multichannel use, the function `segment_multi_channel(...)` provides a direct Python entry point. The current CLI defines `--is_multi_channel` with `argparse` `type=bool`, which should be taken into account when using that flag.

**Skills:** dynamic programming, numeric optimization, prefix-sum acceleration, experimental runtime analysis.
