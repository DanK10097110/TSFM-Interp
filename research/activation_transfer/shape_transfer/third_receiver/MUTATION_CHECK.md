# Independent planted control errors

The clean suite passed both control tests. Each row copied the actual source and planted one change before running the named test. [Runner](mutation_check.py)

| Test | Plant | Observed |
|---|---|---|
| `mean_donor_uses` | Average the donor derivatives on evaluation histories. | `1 failed, 1 deselected in 4.00s` |
| `mean_donor_uses` | Supply the true item derivative to the mean-donor control. | `1 failed, 1 deselected in 4.01s` |
| `wrong_lead_control` | Leave recipient leads in their correct order for the wrong-lead control. | `1 failed, 1 deselected in 3.97s` |
