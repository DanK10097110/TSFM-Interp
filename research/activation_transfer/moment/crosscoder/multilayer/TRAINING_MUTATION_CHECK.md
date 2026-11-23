# Four-view training checks

Both original tests passed in 5.10 seconds. In isolated temporary copies, two planted errors failed the intended test.

| Change | Result |
|---|---|
| Leave small-final on the original history while shuffling large-final. | One failed, one deselected in 4.49 s. |
| Feed large-middle to the nominally small-middle-only encoder. | One failed, one deselected in 4.70 s. |

[Tests](test_multilayer_training.py) · [Mutation runner](mutations.py)
