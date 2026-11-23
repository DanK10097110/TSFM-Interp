# Tests for the permutation and paired contrast

The two unmodified tests passed in 5.10 seconds. Using isolated source copies, we then applied two implementation errors. Each failed the intended assertion.

| Change | Observable failure |
|---|---|
| Preserve the ordinary increasing order in the purported reversal. | The six interior edit values disagree with the independently constructed reverse order. One failed, one deselected in 4.51 s. |
| Reverse the subtraction for the second target while computing the two-target mean. | The directional pooled advantage disagrees with independently calculated values for two paired draws. One failed, one deselected in 4.40 s. |

[Test](test_ramp_interior_order.py) · [Mutation runner](mutations.py)
