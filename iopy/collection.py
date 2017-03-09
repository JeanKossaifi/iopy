""" Module to handle images and shapes as lazy lists.
"""

# Author: Jean KOSSAIFI <jean.kossaifi+code@gmail.com>

from .base import _check_extension
from .image import read_image, image_from_url
from .video import FFMpegVideoReader
from .video import matplotlib_image_list_to_video
from .video import ffmpeg_image_list_to_video
from .image import rgb_to_gray

from pathlib import Path
import numpy as np


class Collection():
    """ List like object of any type of elements

    Parameters
    ----------
    get_elem_fun: function
        fun: param_list[i] -> element
        function that given param_list[i] returs the i-th element of the collection

    param_list  : list of elements
        has as many elements as many elements in the collectin
        its element i is fed to get_elem_fun to get the i-th element of the collection

    get_by_key_fun: function
        fun: str -> element
        function that allows to access an element using a string identifier (dict-like)

    verbose : bool, default is True

    cache_elements : bool, default is False
        if True, the elements are saved in memory when first read and the cached version 
        is used when next accessed.

    **get_elem_params: dictionary of arguments, default is None
        if specified, each of the arguments is passed to get_elem_fun, at each call

    """
    def __init__(self, get_elem_fun, param_list, get_by_key_fun=None,
                 verbose=True, cache_elements=False,
                 **get_elem_params):
        self.get_elem_fun = get_elem_fun
        self.param_list = param_list
        self.size = len(param_list)
        self.get_elem_params = get_elem_params
        self.get_by_key_fun = get_by_key_fun
        self.verbose = verbose
        self.cache_elements = cache_elements
        if self.cache_elements:
            self.cached = dict()

        if verbose and self.size == 0:
            print('Warning: empty collection')

    def __len__(self):
        return self.size

    def __iter__(self):
        for index in range(self.size):
            yield self[index]

    def __getitem__(self, index):
        # See PEP 357
        if hasattr(index, '__index__'):
            index = index.__index__()

        if type(index) not in [int, slice, str]:
            # We also accept a list of int for fancy indexing...
            if not all(isinstance(element, int) for element in index):
                raise TypeError('slicing must be with an int, str or slice'
                                'object or a list of int')

        # If cache is enabled try to get from cache otherwise add elem to cache
        if self.cache_elements and type(index) is int:
            try:
                if type(index) is int:
                    return self.cached[index]
            except KeyError:
                new_elem = self.get_elem_fun(self.param_list[index],
                                             **self.get_elem_params)
                self.cached[index] = new_elem
                return new_elem

        elif type(index) is int:
            return self.get_elem_fun(self.param_list[index],
                                     **self.get_elem_params)

        elif type(index) is slice:
            return self.__class__(self.get_elem_fun, self.param_list[index],
                                  cache_elements=self.cache_elements,
                                  **self.get_elem_params)

        elif type(index) is list:
            return self.__class__(self.get_elem_fun,
                                  [self.param_list[i] for i in index],
                                  cache_elements=self.cache_elements,
                                  **self.get_elem_params)

        elif type(index) is str:
            if self.get_by_key_fun is not None:
                return self.get_by_key_fun(index)
            else:
                raise TypeError('This collection does not support accessing using a string.'
                                'Slicing must be with an int, str or slice object or a list of int.')

    def __setitem__(self, index, value):
        # See PEP 357
        if hasattr(index, '__index__'):
            index = index.__index__()

        if type(index) not in [int, slice]:
            if not all(isinstance(element, int) for element in index):
                raise TypeError('slicing must be with an int, str or slice object')

        if type(value) is type(self):
            if type(index) is list:
                for i, e in enumerate(index):
                    self.param_list[e] = value.param_list[i]
            else:
                self.param_list[index] = value.param_list
        elif self.cache_elements:
            if type(index) is list:
                for i, e in enumerate(index):
                    self.cached[e] = value[i]
            elif type(index) is int:
                self.cached[index] = value
        else:
            raise TypeError('Caching elements is disabled.'
                            'Therefore you can only do collection[index] = value if value is also a Collection.')

    def __str__(self):
        return '{}, collection of {} elements, caching {}abled.'.format(
                        self.__class__.__name__,
                        len(self),
                        'en' if self.cache_elements else 'dis')

    def __repr__(self):
        return '{}, collection of {} elements, cachine {}abled..'.format(
                        self.__class__.__name__,
                        len(self),
                        'en' if self.cache_elements else 'dis')


class ImageCollection(Collection):
    """ List like object of any type of elements

    Parameters
    ----------
    get_elem_fun: function
        fun: param_list[i] -> element
        function that given param_list[i] returs the i-th element of the collection
    param_list  : list of elements
        has as many elements as many elements in the collectin
        its element i is fed to get_elem_fun to get the i-th element of the collection

    """

    @classmethod
    def from_folder(cls, image_folder, image_ext='.png', image_to_gray=False,
                    image_to=None,
                    image_dtype=np.float64, max_n_elements=None,
                    recursive=False, verbose=0, cache_elements=False):
        """ Builds a list-like collection of images.

        Parameters
        ----------
        image_folder: string
            path to the folder containing the target images

        image_ext: string or list of strings, default is '.png'
            extension of the images files
            if list, list of extensions or the images to load

        image_to_gray : bool, default is True
            if True the image is converted to gray-scale

        image_dtype: type, default is np.float64
            type of the image returned (uint8 for a normal image)

        max_n_elements: int, default is None
            if specified, indicates the maximum number of elements to load

        recursive: bool, default is False
            if True, images/shapes are searched for recursively in the
            provided folder

        verbose: int, default is 0
            level of verbosity
        """
        images_path = Path(image_folder)
        # Enable recursive search
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'

        if type(image_ext) is list:
            # Concatenate the image path for all the image extensions provided
            image_ext = [_check_extension(ext) for ext in image_ext]
            image_path_list = sum([list(images_path.glob(pattern + ext)) for ext in image_ext], [])
            image_path_list = sorted(set(image_path_list))
        else:
            image_ext = _check_extension(image_ext)
            image_path_list = sorted(images_path.glob(pattern + image_ext))

        image_list = list()
        for i, path in enumerate(image_path_list):
            base_name = path.stem
            if max_n_elements is not None and i >= max_n_elements:
                break
            image_list.append(path.as_posix())

        if verbose:
            print('Read collection of {} images.'.format(len(image_list)))

        return cls.from_filename_list(image_filename_list=image_list,
                                      image_to_gray=image_to_gray,
                                      image_dtype=image_dtype,
                                      verbose=verbose,
                                      cache_elements=cache_elements)

    @classmethod
    def from_filename_list(cls, image_filename_list, image_to_gray=True,
                           image_dtype=np.float64, verbose=0, cache_elements=False):
        """
        Parameters
        ----------
        image_filename_list: list of strings
            if element should be the absolute path to a image file
            if element i is None, then element i of the collection will be None
        """
        def get_elem_fun(filename, image_to_gray, image_dtype):
            if filename is None:
                return None
            else:
                return read_image(filename,
                                  to_gray=image_to_gray,
                                  dtype=image_dtype)

        return cls(get_elem_fun=get_elem_fun,
                   param_list=image_filename_list,
                   image_to_gray=image_to_gray,
                   image_dtype=image_dtype,
                   cache_elements=cache_elements)

    @classmethod
    def from_video_filename(cls, video_filename, image_to_gray=False,
                            image_dtype=np.float64, verbose=0, cache_elements=False):
        """ Reads images from a video
        """
        reader = FFMpegVideoReader(video_filename)

        def get_elem_fun(index, reader, image_to_gray, image_dtype):
            if image_to_gray:
                frame = rgb_to_gray(reader[index])
            else:
                frame = reader[index]

            if image_dtype is not None:
                return frame.astype(image_dtype)
            else:
                return frame


        param_list = list(range(len(reader)))

        return cls(get_elem_fun=get_elem_fun,
                   param_list=param_list,
                   reader=reader,
                   image_to_gray=image_to_gray,
                   image_dtype=image_dtype,
                   cache_elements=cache_elements)

    @classmethod
    def from_url_list(cls, url_list, image_to_gray=False,
                           image_dtype=None, verbose=0,
                           cache_elements=False):
        """
        Parameters
        ----------
        url_list: list of urls
            if element i is None, then element i of the collection will be None
        """
        def get_elem_fun(url, image_to_gray, image_dtype):
            if url is None:
                return None
            else:
                return image_from_url(url,
                                      to_gray=image_to_gray,
                                      dtype=image_dtype)

        return cls(get_elem_fun=get_elem_fun,
                   param_list=url_list,
                   image_to_gray=image_to_gray,
                   image_dtype=image_dtype,
                   cache_elements=cache_elements)

    def to_matrix(self, apply_fun=None, flatten_images=True):
        """ Returns a matrix with all the images

        Parameters
        ----------

        flatten_images : bool, default is True
            if True, the images are flattened and stacked in a matrix

        apply_fun : bool, default is None
            if not None, the given function is applied to each image

        Returns
        -------
        matrix of shape (n_images, image.shape)
        """
        for i, image in enumerate(self.__iter__()):
            if apply_fun is not None:
                image = apply_fun(image)
            if flatten_images:
                image = image.ravel()
            if not i:
                image_matrix = np.empty((self.size,) + image.shape)
            image_matrix[i] = image
        return image_matrix

    def to_video(self, video_filename, backend='ffmpeg',
                 dpi=92, fps=24, title='Created with iopy',
                 frame_numbers=False,
                 verbose=True):
        """ Saves the collection into a video

        Parameters
        ----------
        video_filename: string
            absolute path to the target video name
            (e.g. : '/temp/video.mp4')

        backend : {'matplotlib', 'ffmpeg'}
                ffmpeg is faster and draws the points directly in the numpy
                array before streaming the arrays to ffmpeg.
        
        fps: int, default is 24
            number of frames (images) per second

        dpi: int, default is 92
            matplotlib only

        title: string
            matplotlib only

        frame_numbers: bool, default is False
            matplotlib only
            if True, displays the number of each frame.

        verbose: bool
        """
        if len(self) == 0:
            if verbose:
                print('Collection is empty, nothing to save...')
            return
               
        if backend == 'matplotlib':
            matplotlib_image_list_to_video(video_filename, self, dpi=dpi,
                                           fps=fps, frame_numbers=frame_numbers,
                                           verbose=verbose)
        else: #  use ffmpeg by default, it is so much faster
            ffmpeg_image_list_to_video(video_filename, self,
                                       fps=fps, verbose=verbose)


