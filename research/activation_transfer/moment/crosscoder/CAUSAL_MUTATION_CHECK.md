# Intervention scorer regression checks

All four unmodified tests passed in 4.54 seconds. Actual isolated copies caught each planted error below.

| Planted error | Result |
|---|---|
| Exclude false-positive writes at closed target gates from the primary score. | One failed, three deselected in 5.00 s. |
| Ignore independently sampled date weights. | One failed, three deselected in 4.37 s. |
| Compare unmasked ordinary joint to masked shuffled and call it a controlled pairing contrast. | One failed, three deselected in 4.15 s. |
| Restrict full-graph verification to genuinely active target gates. | One failed, three deselected in 4.80 s. |

[Tests](test_causal_interventions.py) · [Isolated regression runner](causal_mutations.py)
