import numpy as np
from numpy.testing import assert_equal, assert_raises
from ..misc import _numpy_to_latex

def test__numpy_to_latex():
    """Test for _numpy_to_latex"""
    a = np.array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])
    true_res = '\\left[ \\begin{matrix}\n   0 & 1 & 2 & 3\\\\ \n   4 & 5 & 6 & 7\\\\ \n   8 & 9 & 10 & 11\\\\ \n \\end{matrix} \\right]\n'
    res = _numpy_to_latex(a)
    assert_equal(true_res, res)
    
    # Check for errors
    a = np.arange(3*4*2).reshape((3, 4, 2))
    assert_raises(ValueError, _numpy_to_latex, a)
    
    
test__numpy_to_latex()
