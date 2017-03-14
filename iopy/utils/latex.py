import numpy as np

def _numpy_to_latex(mat, indent=0, print_result=True):
    """Converts a numpy 2D array into a latex matrix

        An utility function that given a numpy array produces latex code.
        You can then print the result and copy paste into your latex file.
    
    Parameters
    ----------
    mat : 2D-array

    Returns
    -------
    latex_mat : str
        print it and copy paste into your latex document
    """
    mat = np.atleast_2d(mat)
    
    if mat.ndim != 2:
        raise ValueError('Only an array of 2 dim can be converted into a latex matrix ({} dims given).'.format(mat.ndim))
    
    indent = ' '*indent
    prefix = indent + "\\left[ \\begin{matrix}\n"
    core = ''
    for line in mat:
        core += indent + '   ' +  ' & '.join(str(i) for i in line) + '\\\\ \n'
    suffix = indent + " \\end{matrix} \\right]\n"
    
    result = prefix + core + suffix

    if print_result:
        print(result)

    return result
