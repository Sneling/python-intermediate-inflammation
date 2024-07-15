"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import os
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max_integers():
    """Test that the max function works on an array of positive integers."""
    from inflammation.models import daily_max

    test_input = np.array([[1, 4],
                           [3, 2],
                           [0, 1]])
    test_result = np.array([3, 4])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_negatives():
    """Test that the max function works with negative integers."""

    from inflammation.models import daily_max

    test_input = np.array([[-10, 4],
                           [3, -3],
                           [0, 1]])
    test_result = np.array([3, 4])

    npt.assert_array_equal(daily_max(test_input), test_result)    


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


def test_load_from_json(tmpdir):
    from inflammation.models import load_json
    example_path = os.path.join(tmpdir, 'example.json')
    with open(example_path, 'w') as temp_json_file:
        temp_json_file.write('[{"observations":[1, 2, 3]},{"observations":[4, 5, 6]}]')
    result = load_json(example_path)
    npt.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])
