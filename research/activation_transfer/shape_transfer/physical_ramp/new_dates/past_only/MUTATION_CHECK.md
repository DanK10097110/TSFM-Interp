# Calendar cutoff verification

The strict date test compares the full eligibility matrix for original and target dates, including a same-day observation, repeated stations, and a target before all available history. It passed in 4.48 seconds. We then changed the earlier-than comparison to earlier-than-or-equal in an isolated copy. The test failed in 5.00 seconds, identifying the same-day leak. [Test](test_ramp_past_only.py) · [Mutation runner](mutation.py)
