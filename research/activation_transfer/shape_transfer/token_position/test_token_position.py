import numpy as np
import pytest

from research.activation_transfer.shape_transfer.token_position.run import observed_positions


def test_window_resolves_observations_with_a_terminal_and_a_leading_special_token():
    spans = np.array([[128,144],[144,160],[160,176],[176,192]])
    # Raw positions 0 and 5 are non-data tokens.
    np.testing.assert_array_equal(observed_positions(spans,np.arange(1,5),2),[3,4])
    with pytest.raises(ValueError,match="contiguous"):
        observed_positions(spans[[0,1,3]],np.arange(1,4),2)
