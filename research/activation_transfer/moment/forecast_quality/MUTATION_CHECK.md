# Mutation checks

Five independent one-line regressions were planted in an isolated source copy. Each produced one assertion failure in its named test and exited 1. Restoring the original sources produced five passes. The production suite separately returned five passes.

| Tested condition | Planted behavior | Failure |
|---|---|---|
| Smallest length with two complete disjoint futures | Reject zero extra slack, losing the exactly 512-value case | Expected origins 480 and 496 |
| Seasonal median with arbitrary phase | Advance the phase by one before prediction | Wrong sequence through wraparound |
| Error scaling at two different origins | Pool raw errors and divide by pooled scale | Expected MASE1 1.5 |
| Weather context gap imputation | Search all data up to forecast origin when filling each earlier gap | Right-side observation changes the expected 18, 14 values |
| Forecast pass cache provenance | Accept an equal-shaped pass after its signature changed | Stale predictions and wrong cache-hit status |
