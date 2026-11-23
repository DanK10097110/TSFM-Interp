# Crossed uncertainty check

The test explicitly duplicates stations and dates in an unequal-weight toy panel. It recomputes the response mean over sampled stations within each date and compares the centered and full sum of squares with the vectorized bootstrap. The unmodified implementation produced **1 passed in 5.56 s**. In an isolated temporary module checkout, replacing the date weights in the first term of the energy estimator with all ones produced **1 failed in 4.68 s**. [Implementation](score.py) · [Test](test_ramp_station_date.py) · [Actual mutation runner](mutation.py)
