# Scientific regression checks

Four corresponding unmodified tests passed in 6.40 seconds. Every planted regression was made in an isolated source copy and failed the intended assertion.

| Change | Result |
|---|---|
| Ignore a gap between adjacent native receiver tokens when grouping physical intervals. | One failed, three deselected in 5.78 s. |
| Allow the matched coordinate to enter its own distractor average. | One failed, three deselected in 6.90 s. |
| Apply normalized decoder weights without restoring the model-specific hidden activation scale. | One failed, three deselected in 5.81 s. |
| Select four source features with replacement, duplicating interventions within a history. | One failed, three deselected in 6.18 s. |

[Tests](test_cross_family_directions.py) · [Isolated mutation runner](mutations.py)
