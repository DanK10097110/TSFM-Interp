# Detector verification

All three tests passed in 6.01 s. Three isolated planted failures verified that the tests exercise the registered rules.

| Regression planted | Designated test | Result |
|---|---|---|
| Return any positive regression output even if below the validation-selected threshold. | Closed receiving threshold | 1 failed, 2 deselected in 6.07 s |
| Center the input with the validation as well as training data. | Training-only scaling | 1 failed, 2 deselected in 5.92 s |
| Suppress false positives on histories where only the receiver fires. | All-history detection | 1 failed, 2 deselected in 6.10 s |

[Test cases](test_receiver_detector.py) · [Isolated mutation runner](mutations.py)
