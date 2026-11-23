# Station inference test mutations

Three tests passed on the original source. Each planted implementation error below failed its expected assertion in an isolated copy.

| Test | Plant | Result |
|---|---|---|
| station_eligibility | Treat multiple origins of one station as different eligible stations. | 1 failed, 2 deselected in 6.04s |
| station_derangement | Compare the next season at the other station. | 1 failed, 2 deselected in 6.43s |
| station_equal | Drop the inverse number-of-firings weight for equal active stations. | 1 failed, 2 deselected in 6.39s |
