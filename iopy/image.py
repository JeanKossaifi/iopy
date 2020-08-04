import numpy as np
from PIL import Image
from urllib import request
from io import BytesIO


def read_image(name, flatten=False, to=None, dtype=None):
    """ Reads the image stored in the given absolute filename

    Parameters
    ----------

    name : string
        absolute path to the image file or file object to be read

    flatten : bool, default in False
        if True, the image is flattened

    to: {'gray', 'rgb', None}, default is None
        if gray the image is converted to gray-scale
        if rgb the image is converted to gray-scale

    dtype: type, default is np.float64
        type of the image returned (uint8 for a normal image)

    Returns
    -------
    array of dtype np.float and shape (size_img_y, size_img_x) is flatten is
    False, and (size_img_y*size_img_x) otherwise
    """
    if to is None:
        image = np.asarray(Image.open(name))
    elif to == 'rgb':
        image = gray_to_rgb(np.asarray(Image.open(name)))
    elif to == 'gray':
        image = rgb_to_gray(np.asarray(Image.open(name).convert('F')))

    if dtype is not None:
        image = image.astype(dtype)

    if flatten:
        return image.ravel()
    else:
        return image


def image_from_url(url, flatten=False, to_gray=False, dtype=None):
    """ Reads an image from an url and returns a numpy array

    Parameters
    ----------

    url : string
        url of the image

    flatten : bool, default in False
        if True, the image is flattened

    to_gray : bool, default is False
        if True the image is converted to gray-scale

    dtype: type, default is np.float64
        type of the image returned (uint8 for a normal image)

    Returns
    -------
    array of dtype np.float and shape (size_img_y, size_img_x) is flatten is
    False, and (size_img_y*size_img_x) otherwise

    Notes
    -----
    Works by opening a stream of Bytes from the url and feeding it to
    read_image.
    """
    data = BytesIO(request.urlopen(url).read())
    image = rgb_to_gray(np.asarray(Image.open(data).convert('F'))) \
        if to_gray else np.asarray(Image.open(data))
    if dtype is not None:
        image = image.astype(dtype)
    if flatten:
        return image.ravel()
    return image


def rgb_to_gray(rgb_image):
    """ Simple hack to convert an RGB image to a gray image
    gray = rgb_image.dot([0.299, 0.587, 0.144])

    Parameters
    ----------
    rgb_image: np array of shape (h, w, 3)
                image

    Returns
    -------
    gray : np array of shape rgb_image.shape[:-1]
        gray version of the image

    Notes
    -----
    We use the ITU-R Recommendation BT.601-7 _[ITU-R]

    .. [ITU-R] ITU-R Recommendation BT.601-7 
       https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
    """
    if rgb_image.ndim != 3:
        print('Image has only two dimensions, must be already gray.')
        return rgb_image

    if rgb_image.shape[-1] != 3:
        print('Last dimension has shape {}: image is not RGB'.format(rgb_image.shape[-1]))
        return rgb_image

    return rgb_image.dot([0.299, 0.587, 0.144])


def gray_to_rgb(gray_image):
    if gray_image.ndim == 3 and gray_image.shape[-1] == 3:
        # Image is already RGB
        return gray_image
    if gray_image.ndim != 2:
        print('gray image should have two dimensions')
        return gray_image
    image = np.empty((gray_image.shape[0], gray_image.shape[1], 3))
    image[...] = gray_image[..., None]
    return image


def save_image(filename, image, extension=None):
    """ Saves an array as an image

    Parameters
    ----------
    filename: str or file object
        Output file name or file object
    image: array of shape MxN or MxNx3 or MxNx4
        array containing the image
    extension: str
        Image format. If omitted, the format to use is determined from the
        file name extension. If a file object was used instead of a file name,
        this parameter should always be used.
    """
    image_out = Image.fromarray(image.astype(np.uint8))
    image_out.save(filename, format=extension)
