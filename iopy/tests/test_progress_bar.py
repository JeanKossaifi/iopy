from .. import progress_bar
from numpy.testing import assert_equal

def test_format_time():
    """ Test for format_time

    Check that the conversion is done correctly
    """
    # just seconds
    time_s = 40
    true_res = '40s'
    res = progress_bar.time_to_string(time_s)
    assert_equal(res, true_res)

    # minutes and seconds
    time_s = 70
    true_res = '1mn, 10s'
    res = progress_bar.time_to_string(time_s)
    assert_equal(res, true_res)

    # hours, minutes and seconds
    time_s = 3700
    true_res = '1h, 1mn, 40s'
    res = progress_bar.time_to_string(time_s)
 
    # days, hours, minutes and seconds
    time_s = 90110
    true_res = '1d, 1h, 1mn, 50s'
    res = progress_bar.time_to_string(time_s)
    assert_equal(res, true_res)
