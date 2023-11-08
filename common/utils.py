import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm


####################################################################################################
#                                              Array utils                                         #
####################################################################################################


def get_slice(arr, rows, cols):
    """
    Extract a slice of values from a 2D NumPy array.

    Parameters:
        arr (numpy.ndarray): The input 2D array from which to extract the slice.
        rows (list): A list of row indexes for the slice.
        cols (list): A list of column indexes for the slice.

    Returns:
        numpy.ndarray: A 1D array containing the extracted slice values.

    Raises:
        ValueError: If the lengths of 'rows' and 'cols' are different.
    """
    if len(rows) != len(cols):
        raise ValueError('Row and column indexes arrays have different sizes')

    arr_cols, arr_rows = arr.shape
    sliced_arr = np.zeros_like(rows)

    for index, (row, col) in enumerate(zip(rows, cols)):
        c_l, c_r = int(col), int(col) + 1
        r_t, r_b = int(row), int(row) + 1
        eta_col, eta_row = (col - c_l), (row - r_t)

        if c_r >= arr_cols:
            c_r = c_r - 1

        if r_b >= arr_rows:
            r_b = r_b - 1

        top = (1 - eta_row) * arr[r_t, c_l] + eta_row * arr[r_t, c_r]
        bottom = (1 - eta_row) * arr[r_b, c_l] + eta_row * arr[r_b, c_r]
        total = (1 - eta_col) * top + eta_col * bottom
        sliced_arr[index] = total

    return sliced_arr


def normalize(arr, zero_padding=False, offset_coef=0.001):
    """
    Normalize a NumPy array to the [0, 1] range.

    Parameters:
        arr (numpy.ndarray): The input array to be normalized.
        zero_padding (bool, optional): If True, normalize without zero-padding.
        offset_coef (float, optional): Coefficient to calculate the offset when
        zero_padding is False.

    Returns:
        numpy.ndarray: The normalized array.

    Notes:
        If zero_padding is True, the normalization is performed in the range [0, 1] without
        zero-padding.
        If zero_padding is False, an offset is added to ensure positive values and
        normalize to [0, 1].

    Examples:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> normalize(arr)
        array([0., 0.25, 0.5 , 0.75, 1.])
        >>> normalize(arr, zero_padding=True)
        array([0., 0.25, 0.5 , 0.75, 1.])
    """
    min_val = np.min(arr)
    max_val = np.max(arr)

    if zero_padding:
        new_arr = (arr - min_val) / (max_val - min_val)
    else:
        offset = offset_coef * abs(min_val)
        new_arr = (arr - min_val + offset) / (max_val - min_val + offset)

    return new_arr


def normalize_img(arr):
    """
    Normalize an image array to the [0, 255] range.

    Parameters:
        arr (numpy.ndarray): The input image array to be normalized.

    Returns:
        numpy.ndarray: The normalized image array.

    Notes:
        If the input array values are already in the range [0, 255], no normalization is performed.
        Otherwise, the input array values are scaled to fit within the [0, 255] range.

    Examples:
        >>> image = np.array([[100, 200], [50, 150]])
        >>> normalize_img(image)
        array([[100, 200],
               [ 50, 150]])
        >>> image = np.array([[50, 150], [25, 75]])
        >>> normalize_img(image)
        array([[  0., 255.],
               [  0., 127.5]])
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    if arr_min >= 0 and arr_max <= 255:
        return arr
    else:
        arr_norm = 255 * (arr - arr_min) / (arr_max - arr_min)
        return arr_norm


def threshold_arr_1d(arr, th_val, use_abs=False):
    """
    Apply thresholding to a 1D NumPy array.

    Parameters:
        arr (numpy.ndarray): The input 1D array to be thresholded.
        th_val (float): The threshold value. Values below this threshold will be replaced.
        use_abs (bool, optional): If True, use the absolute value of th_val for thresholding.

    Returns:
        numpy.ndarray: The thresholded 1D array.

    Notes:
        If 'use_abs' is True, the absolute value of 'th_val' is used for thresholding.
        Values in 'arr' that are less than 'th_val' are replaced with 'th_val'.

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> threshold_arr_1d(data, 3)
        array([3, 3, 3, 4, 5])
        >>> threshold_arr_1d(data, -2, use_abs=True)
        array([2, 2, 3, 4, 5])
    """
    th_val = abs(th_val) if use_abs else th_val
    for i in range(len(arr)):
        if arr[i] < th_val:
            arr[i] = th_val
    return arr


def threshold_arr_2d(arr, th_val, use_abs=False):
    """
    Apply thresholding to a 2D NumPy array row-wise.

    Parameters:
        arr (numpy.ndarray): The input 2D array to be thresholded row-wise.
        th_val (float): The threshold value. Values below this threshold will be replaced.
        use_abs (bool, optional): If True, use the absolute value of th_val for thresholding.

    Returns:
        numpy.ndarray: The thresholded 2D array.

    Notes:
        If 'use_abs' is True, the absolute value of 'th_val' is used for thresholding.
        Thresholding is applied row-wise to each row in 'arr'.

    Examples:
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> threshold_arr_2d(data, 4)
        array([[4, 4, 4],
               [4, 5, 6],
               [7, 8, 9]])
        >>> threshold_arr_2d(data, -2, use_abs=True)
        array([[2, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    """
    for i in range(arr.shape[0]):
        arr[i, :] = threshold_arr_1d(arr[i, :], th_val, use_abs)
    return arr


def clip_graph(x_arr, y_arr, x_min=0, x_max=100, y_min=0, y_max=100):
    """
    Clip data points within specified x and y ranges.

    Parameters:
        x_arr (numpy.ndarray): The x-coordinates of the data points.
        y_arr (numpy.ndarray): The y-coordinates of the data points.
        x_min (float, optional): The minimum x-value for clipping.
        x_max (float, optional): The maximum x-value for clipping.
        y_min (float, optional): The minimum y-value for clipping.
        y_max (float, optional): The maximum y-value for clipping.

    Returns:
        tuple: A tuple containing the clipped x and y arrays.

    Notes:
        Data points outside the specified ranges [x_min, x_max] and [y_min, y_max] are removed.

    Examples:
        >>> x = np.array([0, 50, 100, 150, 200])
        >>> y = np.array([0, 25, 50, 75, 100])
        >>> clipped_x, clipped_y = clip_graph(x, y, x_min=50, x_max=150, y_min=25, y_max=75)
        >>> clipped_x
        array([100])
        >>> clipped_y
        array([50])
    """
    x_mask = (x_arr >= x_min) & (x_arr <= x_max)
    y_mask = (y_arr >= y_min) & (y_arr <= y_max)

    # Apply clipping to both x and y arrays
    x_arr = x_arr[x_mask & y_mask]
    y_arr = y_arr[x_mask & y_mask]

    return x_arr, y_arr


####################################################################################################
#                                            FFT utils                                             #
####################################################################################################


def find_ft_1d(arr):
    """
    Calculate the 1D Fourier transform of an array and shift the frequencies to the center.

    Parameters:
        arr (numpy.ndarray): The input 1D array for Fourier transform.

    Returns:
        numpy.ndarray: The 1D Fourier transform with shifted frequencies.

    Notes:
        The function computes the Fourier transform of the input array and shifts the frequencies
        to center them around zero frequency.

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> ft = find_ft_1d(data)
        >>> ft
        array([ 7.5       +0.j        , -2.11803399+1.53884177j,
               -2.11803399-1.53884177j, -2.11803399+0.36327126j,
               -2.11803399-0.36327126j])
    """
    ft = np.fft.fft(arr)
    return np.fft.fftshift(ft)


def find_ift_1d(arr):
    """
    Calculate the 1D inverse Fourier transform of an array with shifted frequencies.

    Parameters:
        arr (numpy.ndarray): The input 1D array with shifted frequencies.

    Returns:
        numpy.ndarray: The 1D inverse Fourier transform as a real-valued array.

    Notes:
        The function takes an array with shifted frequencies, performs an inverse Fourier transform,
        and returns the real part of the result.

    Examples:
        >>> data = np.array([ 7.5       +0.j        , -2.11803399+1.53884177j,
        ...                   -2.11803399-1.53884177j, -2.11803399+0.36327126j,
        ...                   -2.11803399-0.36327126j])
        >>> ift = find_ift_1d(data)
        >>> ift
        array([1., 2., 3., 4., 5.])
    """
    ift = np.fft.ifftshift(arr)
    return np.fft.ifft(ift).real


def adjust_spectrum_1d(img):
    """
    Adjust a 1D spectrum array to maintain symmetry for Fourier transform processing.

    This function checks if the length of the input array (spectrum) is even. If it is,
    the absolute value of the first element is appended to the array to ensure symmetry.
    This is particularly useful in the context of Fourier transforms where symmetric
    spectra are often required.

    Parameters:
    img (numpy.ndarray): The 1D spectrum array obtained from a Fourier transform.

    Returns:
    numpy.ndarray: The adjusted spectrum array, ensuring symmetry if the original length is even.

    Example:
    >>> spectrum = np.array([-2, -1, 0, 1])
    >>> adjusted_spectrum = adjust_spectrum_1d(spectrum)
    >>> print(adjusted_spectrum)  # Output for even length: [-2 -1  0  1  2]
    >>> odd_spectrum = np.array([-2, -1, 0, 1, 2])
    >>> adjusted_odd_spectrum = adjust_spectrum_1d(odd_spectrum)
    >>> print(adjusted_odd_spectrum)  # Output for odd length: [-2 -1  0  1  2]
    """
    if len(img) % 2:
        return img
    else:
        return np.append(img, abs(img[0]))


def find_ft_2d(arr):
    """
    Calculate the 2D Fourier transform of a 2D array and shift the frequencies to the center.

    Parameters:
        arr (numpy.ndarray): The input 2D array for Fourier transform.

    Returns:
        numpy.ndarray: The 2D Fourier transform with shifted frequencies.

    Notes:
        The function computes the 2D Fourier transform of the input array and shifts the frequencies
        to center them around zero frequency.

    Examples:
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> ft = find_ft_2d(data)
        >>> ft
        array([[ 0. +0.j,  0. +0.j,  0. +0.j],
               [ 0.5-1.5j,  0. +0.j, -0.5+1.5j],
               [ 0. +0.j,  0. +0.j,  0. +0.j]])
    """
    ft = np.fft.fft2(arr)
    return np.fft.fftshift(ft)


def find_ift_2d(arr):
    """
    Calculate the 2D inverse Fourier transform of a 2D array with shifted frequencies.

    Parameters:
        arr (numpy.ndarray): The input 2D array with shifted frequencies.

    Returns:
        numpy.ndarray: The 2D inverse Fourier transform as a real-valued array.

    Notes:
        The function takes a 2D array with shifted frequencies, performs an inverse Fourier
        transform, and returns the real part of the result.

    Examples:
        >>> data = np.array([[ 0. +0.j,  0. +0.j,  0. +0.j],
        ...                  [ 0.5-1.5j,  0. +0.j, -0.5+1.5j],
        ...                  [ 0. +0.j,  0. +0.j,  0. +0.j]])
        >>> ift = find_ift_2d(data)
        >>> ift
        array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 9.]])
    """
    ift = np.fft.ifftshift(arr)
    return np.fft.ifft2(ift).real


def freq_numbers_1d(size):
    """
    Generate a 1D array of frequency numbers centered around zero.

    Parameters:
        size (int): The number of elements in the output array.

    Returns:
        numpy.ndarray: A 1D array of frequency numbers.

    Notes:
        The function generates a 1D array of frequency numbers ranging from
        -(size // 2) to size // 2 (inclusive), centered around zero.

    Examples:
        >>> freq_numbers_1d(5)
        array([-2, -1,  0,  1,  2])
        >>> freq_numbers_1d(6)
        array([-2, -1,  0,  1,  2,  3])
    """
    if size % 2:
        return np.arange(-(size // 2), size // 2 + 1, 1)
    else:
        return np.arange(-(size // 2), size // 2, 1)


def freq_arr_1d(size, step=1):
    """
    Generate a 1D array of frequencies scaled by a specified step size.

    Parameters:
        size (int): The number of elements in the output array.
        step (float, optional): The scaling factor for frequencies (default is 1).

    Returns:
        numpy.ndarray: A 1D array of scaled frequencies.

    Notes:
        The function generates a 1D array of frequencies ranging from
        -(size // 2) to size // 2 (inclusive), centered around zero. The frequencies
        are then scaled by the specified 'step' value.

    Examples:
        >>> freq_arr_1d(5)
        array([-0.4, -0.2,  0.,  0.2,  0.4])
        >>> freq_arr_1d(6, step=0.5)
        array([-0.33333333, -0.16666667,  0.,  0.16666667,  0.33333333, 0.5])
    """
    freq = freq_numbers_1d(size)
    return freq / step / size


def freq_numbers_2d(shape):
    """
    Generate 2D arrays of frequency numbers centered around zero.

    Parameters:
        shape (tuple): A tuple specifying the shape of the 2D arrays (y_size, x_size).

    Returns:
        tuple: A tuple containing 2D arrays of frequency numbers for both x and y axes.

    Notes:
        The function generates 2D arrays of frequency numbers for both x and y axes, centered
        around zero. The 'shape' tuple specifies the dimensions of the output arrays.

    Examples:
        >>> freq_numbers_2d((3, 4))
        (array([ 0,  0,  0,  0]),
         array([-1,  0,  1]))
        >>> freq_numbers_2d((5, 5))
        (array([-2, -1,  0,  1,  2]),
         array([-2, -1,  0,  1,  2]))
    """
    y_size, x_size = shape
    y_freq_numbers = freq_numbers_1d(y_size)
    x_freq_numbers = freq_numbers_1d(x_size)
    return x_freq_numbers, y_freq_numbers


def freq_arr_2d(shape, x_step=1, y_step=1):
    """
    Generate 2D arrays of scaled frequencies for both x and y axes.

    Parameters:
        shape (tuple): A tuple specifying the shape of the 2D arrays (y_size, x_size).
        x_step (float, optional): The scaling factor for the x-axis frequencies (default is 1).
        y_step (float, optional): The scaling factor for the y-axis frequencies (default is 1).

    Returns:
        tuple: A tuple containing 2D arrays of scaled frequencies for both x and y axes.

    Notes:
        The function generates 2D arrays of scaled frequencies for both x and y axes, based on the
        specified 'shape', 'x_step', and 'y_step' values.

    Examples:
        >>> freq_arr_2d((3, 4))
        (array([ 0.,  0.,  0.,  0.]),
         array([-1.,  0.,  1.]))
        >>> freq_arr_2d((5, 5), x_step=0.5, y_step=0.25)
        (array([-2., -1.,  0.,  1.,  2.]),
         array([-2., -1.,  0.,  1.,  2.]))
    """
    y_size, x_size = shape
    x_freq_numbers, y_freq_numbers = freq_numbers_2d(shape)
    x_freq = x_freq_numbers / x_step / x_size
    y_freq = y_freq_numbers / y_step / y_size
    return x_freq, y_freq


def freq_mesh_2d(shape):
    """
    Generate 2D mesh grids of frequencies for both x and y axes.

    Parameters:
        shape (tuple): A tuple specifying the shape of the 2D arrays (y_size, x_size).

    Returns:
        tuple: A tuple containing 2D mesh grids of frequencies for both x and y axes.

    Notes:
        The function generates 2D mesh grids of frequencies for both x and y axes, based on the
        specified 'shape'.

    Examples:
        >>> freq_mesh_2d((3, 4))
        (array([[-1,  0,  1,  2],
                [-1,  0,  1,  2],
                [-1,  0,  1,  2]]), array([0, 0, 0, 0]))
        >>> freq_mesh_2d((5, 5))
        (array([[-2, -1,  0,  1,  2],
                [-2, -1,  0,  1,  2],
                [-2, -1,  0,  1,  2],
                [-2, -1,  0,  1,  2],
                [-2, -1,  0,  1,  2]]), array([-2, -1,  0,  1,  2]))
    """
    x_freq_numbers, y_freq_numbers = freq_numbers_2d(shape)
    x_mesh, y_mesh = np.meshgrid(x_freq_numbers, y_freq_numbers)
    return x_mesh, y_mesh


####################################################################################################
#                                      Frequency domain filters                                    #
####################################################################################################


def freq_pink_filter_1d(x_freq, factor=0.5, no_mean=False):
    """
    Apply a 1D pink noise filter (1 / f) to a frequency domain signal.

    Parameters:
        x_freq (numpy.ndarray): The input 1D frequency signal.
        factor (float, optional): Exponent factor for the pink noise filter (default is 0.5).
        no_mean (bool, optional): Whether to remove the mean component (default is False).

    Returns:
        numpy.ndarray: The filtered frequency signal.

    Notes:
        The function applies a pink noise filter to the input frequency signal in the
        frequency domain.
        The 'factor' parameter controls the shape of the filter, and 'no_mean' can be used to remove
        the mean component from the filtered signal.

    Examples:
        >>> freq_signal = np.array([1, 2, 3, 4, 5])
        >>> filtered_signal = freq_pink_filter_1d(freq_signal)
        >>> filtered_signal
        array([1., 1., 0.70710678, 0.57735027, 0.5, 0.4472136])
    """
    x_freq = np.abs(x_freq)
    f = 1 / np.where(x_freq <= 1, 1, x_freq)
    f = f ** factor
    if no_mean:
        f_mask = x_freq < 1
        f[f_mask] = 0
    return f


def freq_pink_filter_1d_alt(x_freq, factor=0.5, no_mean=False):
    """
    Apply an alternative 1D pink noise filter (1 / (1 + f)) to a frequency domain signal.

    Parameters:
        x_freq (numpy.ndarray): The input 1D frequency signal.
        factor (float, optional): Exponent factor for the pink noise filter (default is 0.5).
        no_mean (bool, optional): Whether to remove the mean component (default is False).

    Returns:
        numpy.ndarray: The filtered frequency signal.

    Notes:
        The function applies a pink noise filter to the input frequency signal in the
        frequency domain.
        The 'factor' parameter controls the shape of the filter, and 'no_mean' can be used to remove
        the mean component from the filtered signal.

    Examples:
        >>> freq_signal = np.array([1, 2, 3, 4, 5])
        >>> filtered_signal = freq_pink_filter_1d_old(freq_signal)
        >>> filtered_signal
        array([1., 0.70710678, 0.57735027, 0.5, 0.4472136, 0.40824829])
    """
    x_freq = np.abs(x_freq)
    f = 1 / (1 + x_freq)
    f = f ** factor
    if no_mean:
        f_mask = x_freq < 1
        f[f_mask] = 0
    return f


def freq_filter_2d(x_freq, y_freq):
    """
    Generate a 2D frequency filter based on 1D frequency arrays.

    Parameters:
        x_freq (numpy.ndarray): 1D array of frequencies for the x-axis.
        y_freq (numpy.ndarray): 1D array of frequencies for the y-axis.

    Returns:
        numpy.ndarray: A 2D frequency filter based on the input frequencies.

    Notes:
        The function generates a 2D frequency filter based on the provided 1D frequency arrays for
        both the x and y axes. The resulting filter is a 2D array with values representing
        the magnitude of frequencies at different points.

    Examples:
        >>> x_freq = np.array([-2, -1, 0, 1, 2])
        >>> y_freq = np.array([-2, -1, 0, 1, 2])
        >>> filter = freq_filter_2d(x_freq, y_freq)
        >>> filter
        array([[2.82842712, 2.23606798, 2., 2.23606798, 2.82842712],
               [2.23606798, 1.41421356, 1., 1.41421356, 2.23606798],
               [2.        , 1.        , 0., 1.        , 2.        ],
               [2.23606798, 1.41421356, 1., 1.41421356, 2.23606798],
               [2.82842712, 2.23606798, 2., 2.23606798, 2.82842712]])
    """
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y)
    return f


def freq_pink_filter_2d(x_freq, y_freq, factor=1, x_stretch=1, y_stretch=1, no_mean=False):
    """
    Apply a 2D pink noise filter to a frequency domain signal.

    Parameters:
        x_freq (numpy.ndarray): 1D array of frequencies for the x-axis.
        y_freq (numpy.ndarray): 1D array of frequencies for the y-axis.
        factor (float, optional): Exponent factor for the pink noise filter (default is 1).
        x_stretch (float, optional): Scaling factor for the x-axis frequencies (default is 1).
        y_stretch (float, optional): Scaling factor for the y-axis frequencies (default is 1).
        no_mean (bool, optional): Whether to remove the mean component (default is False).

    Returns:
        numpy.ndarray: The filtered 2D frequency signal.

    Notes:
        The function applies a 2D pink noise filter to the input frequency signal in the
        frequency domain.
        The 'factor' parameter controls the shape of the filter, 'x_stretch' and 'y_stretch'
        can be used
        to scale the frequency axes, and 'no_mean' can be used to remove the mean component from the
        filtered signal.

    Examples:
        >>> x_freq = np.array([-2, -1, 0, 1, 2])
        >>> y_freq = np.array([-2, -1, 0, 1, 2])
        >>> filtered_signal = freq_pink_filter_2d(x_freq, y_freq, factor=0.5, x_stretch=2,
                                                  y_stretch=1, no_mean=True)
        >>> filtered_signal
        array([[0.6687403 , 0.69647057, 0.70710678, 0.69647057, 0.6687403 ],
               [0.84089642, 0.94574161, 1.        , 0.94574161, 0.84089642],
               [1.        , 0.        , 0.        , 0.        , 1.        ],
               [0.84089642, 0.94574161, 1.        , 0.94574161, 0.84089642],
               [0.6687403 , 0.69647057, 0.70710678, 0.69647057, 0.6687403 ]])
    """
    x_freq, y_freq = np.abs(x_freq), np.abs(y_freq)
    fr = freq_filter_2d(x_freq / x_stretch, y_freq / y_stretch)
    f = 1 / np.where(fr <= 1, 1, fr)
    f = f ** factor
    if no_mean:
        f_mask = np.abs(fr) < 1
        f[f_mask] = 0
    return f


def freq_pink_filter_2d_alt(x_freq, y_freq, factor=1, x_stretch=1, y_stretch=1, no_mean=False):
    """
    Apply an alternative 2D pink noise filter to a frequency domain signal.

    Parameters:
        x_freq (numpy.ndarray): 1D array of frequencies for the x-axis.
        y_freq (numpy.ndarray): 1D array of frequencies for the y-axis.
        factor (float, optional): Exponent factor for the pink noise filter (default is 1).
        x_stretch (float, optional): Scaling factor for the x-axis frequencies (default is 1).
        y_stretch (float, optional): Scaling factor for the y-axis frequencies (default is 1).
        no_mean (bool, optional): Whether to remove the mean component (default is False).

    Returns:
        numpy.ndarray: The filtered 2D frequency signal.

    Notes:
        The function applies an alternative 2D pink noise filter to the input frequency signal
        in the frequency domain.
        The 'factor' parameter controls the shape of the filter, 'x_stretch' and 'y_stretch'
        can be used
        to scale the frequency axes, and 'no_mean' can be used to remove the mean component from
        the filtered signal.

    Examples:
        >>> x_freq = np.array([-2, -1, 0, 1, 2])
        >>> y_freq = np.array([-2, -1, 0, 1, 2])
        >>> filtered_signal = freq_pink_filter_2d_alt(x_freq, y_freq, factor=0.5,
                                                      x_stretch=2, y_stretch=1)
        >>> filtered_signal
        array([[0.55589297, 0.57151696, 0.57735027, 0.57151696, 0.55589297],
               [0.64359425, 0.6871215 , 0.70710678, 0.6871215 , 0.64359425],
               [0.70710678, 0.81649658, 1.        , 0.81649658, 0.70710678],
               [0.64359425, 0.6871215 , 0.70710678, 0.6871215 , 0.64359425],
               [0.55589297, 0.57151696, 0.57735027, 0.57151696, 0.55589297]])
    """
    f = freq_filter_2d(x_freq / x_stretch, y_freq / y_stretch)
    f = 1 / (1 + f)
    f = f ** factor
    f = np.where(f == 1, 0, f) if no_mean else f
    return f


def freq_sharp_round_filter_2d(x_freq, y_freq, radius, low_pass_filter=True):
    """
    Apply a 2D sharp-round filter to a frequency domain signal.

    Parameters:
        x_freq (numpy.ndarray): 1D array of frequencies for the x-axis.
        y_freq (numpy.ndarray): 1D array of frequencies for the y-axis.
        radius (float): The radius of the circular filter.
        low_pass_filter (bool, optional): Whether to apply a low-pass or high-pass
                                          filter (default is True).

    Returns:
        numpy.ndarray: The filtered 2D frequency signal.

    Notes:
        The function applies a 2D sharp-round filter to the input frequency signal in the
        frequency domain.
        The 'radius' parameter determines the size of the circular filter, and 'low_pass_filter'
        can be used to choose between a low-pass or high-pass filter.

    Examples:
        >>> x_freq = np.array([-2, -1, 0, 1, 2])
        >>> y_freq = np.array([-2, -1, 0, 1, 2])
        >>> filtered_signal = freq_sharp_round_filter_2d(x_freq, y_freq, radius=1.5,
                                                         low_pass_filter=True)
        >>> filtered_signal
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])
    """
    check = np.less if low_pass_filter else np.greater
    f = freq_filter_2d(x_freq, y_freq)
    f = np.where(check(f, radius), 1, 0)
    return f


def freq_sharp_square_filter_2d(x_freq, y_freq, width, angle=0, low_pass_filter=True):
    """
    Apply a 2D sharp-square filter to a frequency domain signal.

    Parameters:
        x_freq (numpy.ndarray): 1D array of frequencies for the x-axis.
        y_freq (numpy.ndarray): 1D array of frequencies for the y-axis.
        width (float): The width of the square filter.
        angle (float, optional): The rotation angle of the square filter in degrees (default is 0).
        low_pass_filter (bool, optional): Whether to apply a low-pass or high-pass
                                          filter (default is True).

    Returns:
        numpy.ndarray: The filtered 2D frequency signal.

    Notes:
        The function applies a 2D sharp-square filter to the input frequency signal in the
        frequency domain.
        The 'width' parameter determines the size of the square filter, 'angle' can be used to
        rotate the filter, and 'low_pass_filter' can be used to choose between a low-pass or
        high-pass filter.

    Examples:
        >>> x_freq = np.array([-2, -1, 0, 1, 2])
        >>> y_freq = np.array([-2, -1, 0, 1, 2])
        >>> filtered_signal = freq_sharp_square_filter_2d(x_freq, y_freq, width=1.5, angle=30,
                                                          low_pass_filter=True)
        >>> filtered_signal
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])
    """
    check = np.less if low_pass_filter else np.greater
    angle_radians = np.radians(angle)

    x, y = np.meshgrid(x_freq, y_freq)
    rotated_x = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    rotated_x = np.where(check(np.abs(rotated_x), width), 1, 0)

    rotated_y = x * np.sin(angle_radians) + y * np.cos(angle_radians)
    rotated_y = np.where(check(np.abs(rotated_y), width), 1, 0)

    f = rotated_x * rotated_y
    return f


####################################################################################################
#                                       Spatial domain utils                                       #
####################################################################################################


def spatial_smooth_filter(x_size, y_size, depth, horiz=True):
    """
    Generate a 2D spatial smoothing filter.

    Parameters:
        x_size (int): The width of the filter.
        y_size (int): The height of the filter.
        depth (int): The depth of the filter.
        horiz (bool, optional): Whether to create a horizontal or vertical filter (default is True).

    Returns:
        numpy.ndarray: A 2D spatial smoothing filter.

    Notes:
        The function generates a 2D spatial smoothing filter with the specified dimensions.
        The 'depth' parameter controls the number of levels in the filter, and 'horizontal'
        determines whether the filter is horizontal (if True) or vertical (if False).

    Examples:
        >>> x_size = 5
        >>> y_size = 3
        >>> depth = 4
        >>> horizontal_filter = spatial_smooth_filter(x_size, y_size, depth, horiz=True)
        >>> horizontal_filter
        array([[1.0, 0.79012346, 0.20987654, 0.],
               [1.0, 0.79012346, 0.20987654, 0.],
               [1.0, 0.79012346, 0.20987654, 0.]])
    """
    values = np.linspace(0, 1, depth)
    values = 1 - fifth_order_interp(values)

    if horiz:
        kernel = np.tile(values, (y_size, 1))
    else:
        kernel = values[:, np.newaxis] * np.ones((1, x_size))

    return kernel


def make_img_transition_x(img, depth, is_dx_pos=True, outer_smooth=False):
    """
    Create an image with a smooth transition in the x-direction.

    Parameters:
        img (numpy.ndarray): The input image.
        depth (int): The depth of the x-direction transition region.
        is_dx_pos (bool, optional): Whether the transition is in the positive
                                    x-direction (default is True).
        outer_smooth (bool, optional): Whether to apply outer smoothing to the
                                       transition (default is False).

    Returns:
        numpy.ndarray: The image with the x-direction transition.

    Notes:
        The function creates an image with a transition in the x-direction. The 'depth'
        parameter controls the width of the transition, 'is_dx_pos' determines whether the
        transition is in the positive x-direction, and 'outer_smooth' can be used to apply
        outer smoothing to the transition region.

    Examples:
        >>> input_img = np.random.normal(0, 1, (3, 3))
        >>> depth = 2
        >>> is_dx_pos = True
        >>> outer_smooth = False
        >>> output_img = make_img_transition_x(input_img, depth, is_dx_pos, outer_smooth)
    """
    y_size, x_size = img.shape
    add_img = gen_cloud(x_size + depth, y_size)
    transition_kernel = spatial_smooth_filter(x_size, y_size, depth)

    img_copy = np.copy(img)
    if is_dx_pos:
        img_copy[:, -depth:] = (img[:, -depth:] * transition_kernel +
                                add_img[:, :depth] * (1 - transition_kernel))
        img_copy = np.concatenate((img_copy, add_img[:, depth:]), axis=1)
    else:
        transition_kernel = np.fliplr(transition_kernel)
        img_copy[:, :depth] = (img[:, :depth] * transition_kernel +
                               add_img[:, -depth:] * (1 - transition_kernel))
        img_copy = np.concatenate((add_img[:, 0:-depth], img_copy), axis=1)

    if outer_smooth:
        add_img = gen_cloud(2 * depth, y_size)
        transition_kernel = spatial_smooth_filter(x_size, y_size, depth)
        transition_kernel = np.fliplr(transition_kernel)
        img_copy[:, :depth] = (img_copy[:, :depth] * transition_kernel +
                               add_img[:, :depth] * (1 - transition_kernel))
        transition_kernel = np.fliplr(transition_kernel)
        img_copy[:, -depth:] = (img_copy[:, -depth:] * transition_kernel +
                                add_img[:, -depth:] * (1 - transition_kernel))
    return img_copy


def make_img_transition_y(img, depth, is_dy_pos=True, outer_smooth=False):
    """
    Create an image with a transition in the y-direction.

    Parameters:
        img (numpy.ndarray): The input image.
        depth (int): The depth of the y-direction transition.
        is_dy_pos (bool, optional): Whether the transition is in the positive
                                    y-direction (default is True).
        outer_smooth (bool, optional): Whether to apply outer smoothing to the
                                       transition (default is False).

    Returns:
        numpy.ndarray: The image with the y-direction transition.

    Notes:
        The function creates an image with a transition in the y-direction. The 'depth'
        parameter controls the height of the transition, 'is_dy_pos' determines whether the
        transition is in the positive y-direction, and 'outer_smooth' can be used to apply outer
        smoothing to the transition region.

    Examples:
        >>> input_img = np.random.normal(0, 1, (3, 3))
        >>> depth = 2
        >>> is_dy_pos = True
        >>> outer_smooth = False
        >>> output_img = make_img_transition_y(input_img, depth, is_dy_pos, outer_smooth)
    """
    y_size, x_size = img.shape
    add_img = gen_cloud(x_size, y_size + depth)
    transition_kernel = spatial_smooth_filter(x_size, y_size, depth, horiz=False)

    img_copy = np.copy(img)
    if is_dy_pos:
        img_copy[-depth:, :] = img[-depth:, :] * transition_kernel + \
                               add_img[:depth, :] * (1 - transition_kernel)
        img_copy = np.concatenate((img_copy, add_img[depth:, :]), axis=0)
    else:
        transition_kernel = np.flipud(transition_kernel)
        img_copy[:depth, :] = (img[:depth, :] * transition_kernel + 
                               add_img[-depth:, :] * (1 - transition_kernel))
        img_copy = np.concatenate((add_img[:-depth, :], img_copy), axis=0)

    if outer_smooth:
        add_img = gen_cloud(x_size, 2 * depth)
        transition_kernel = spatial_smooth_filter(x_size, y_size, depth, horiz=False)
        transition_kernel = np.flipud(transition_kernel)
        img_copy[:depth, :] = (img_copy[:depth, :] * transition_kernel + 
                               add_img[:depth, :] * (1 - transition_kernel))
        transition_kernel = np.flipud(transition_kernel)
        img_copy[-depth:, :] = (img_copy[-depth:, :] * transition_kernel + 
                                add_img[-depth:, :] * (1 - transition_kernel))
    return img_copy


def make_img_transition_xy(img, depth, is_dx_pos=True, is_dy_pos=True, outer_smooth=False):
    """
    Create an image with transitions in both the x and y directions.

    Parameters:
        img (numpy.ndarray): The input image.
        depth (int): The depth of the transitions.
        is_dx_pos (bool, optional): Whether the x-direction transition is in the positive
                                    x-direction (default is True).
        is_dy_pos (bool, optional): Whether the y-direction transition is in the positive
                                    y-direction (default is True).
        outer_smooth (bool, optional): Whether to apply outer smoothing to the
                                       transitions (default is False).

    Returns:
        numpy.ndarray: The image with transitions in both the x and y directions.

    Notes:
        The function creates an image with transitions in both the x and y directions.
        The 'depth' parameter controls the width and height of the transitions, 'is_dx_pos' and
        'is_dy_pos' determine the direction of the x and y transitions, and 'outer_smooth'
        can be used to apply outer smoothing to the transition regions.

    Examples:
        >>> input_img = np.random.normal(0, 1, (3, 3))
        >>> depth = 2
        >>> is_dx_pos = True
        >>> is_dy_pos = True
        >>> outer_smooth = False
        >>> output_img = make_img_transition_xy(input_img, depth, is_dx_pos, is_dy_pos,outer_smooth)
    """
    new_img = make_img_transition_x(img, depth, is_dx_pos=is_dx_pos, outer_smooth=outer_smooth)
    new_img = make_img_transition_y(new_img, depth, is_dy_pos=is_dy_pos, outer_smooth=outer_smooth)
    return new_img


def shift_img_x(img, dx, is_dx_pos=True):
    """
    Shift an image horizontally in the x-direction.

    Parameters:
        img (numpy.ndarray): The input image.
        dx (int): The amount of horizontal shift.
        is_dx_pos (bool, optional): Whether the shift is in the positive
                                    x-direction (default is True).

    Returns:
        numpy.ndarray: The image shifted in the x-direction.

    Notes:
        The function shifts an input image horizontally in the x-direction by the specified
        amount 'dx'. The 'is_dx_pos' parameter determines whether the shift is in the positive
        x-direction (right) or negative x-direction (left).

    Examples:
        >>> input_img = np.random.normal(0, 1, (3, 5))
        >>> dx = 2
        >>> is_dx_pos = True
        >>> shifted_img = shift_img_x(input_img, dx, is_dx_pos)
    """
    _, width = img.shape

    if is_dx_pos:
        c1 = img[:, -dx:]
        c2 = img[:, :-dx]
    else:
        c1 = img[:, dx:]
        c2 = img[:, :dx]

    return np.concatenate((c1, c2), axis=1)


def shift_img_y(img, dy, is_dy_pos=True):
    """
    Shift an image vertically in the y-direction.

    Parameters:
        img (numpy.ndarray): The input image.
        dy (int): The amount of vertical shift.
        is_dy_pos (bool, optional): Whether the shift is in the positive
                                    y-direction (default is True).

    Returns:
        numpy.ndarray: The image shifted in the y-direction.

    Notes:
        The function shifts an input image vertically in the y-direction by the specified
        amount 'dy'. The 'is_dy_pos' parameter determines whether the shift is in the
        positive y-direction (down) or negative y-direction (up).

    Examples:
        >>> input_img = np.random.normal(0, 1, (3, 5))
        >>> dy = 2
        >>> is_dy_pos = True
        >>> shifted_img = shift_img_y(input_img, dy, is_dy_pos)
    """
    height, _ = img.shape

    if is_dy_pos:
        c1 = img[-dy:, :]
        c2 = img[:-dy, :]
    else:
        c1 = img[dy:, :]
        c2 = img[:dy, :]

    return np.concatenate((c1, c2), axis=0)


def shift_img_xy(img, dx, dy, is_dx_pos=True, is_dy_pos=True):
    """
    Shift an image both horizontally and vertically.

    Parameters:
        img (numpy.ndarray): The input image.
        dx (int): The amount of horizontal shift.
        dy (int): The amount of vertical shift.
        is_dx_pos (bool, optional): Whether the x-direction shift is in the positive
                                    x-direction (default is True).
        is_dy_pos (bool, optional): Whether the y-direction shift is in the positive
                                    y-direction (default is True).

    Returns:
        numpy.ndarray: The image shifted both horizontally and vertically.

    Notes:
        The function shifts an input image both horizontally and vertically by the specified
        amounts 'dx' and 'dy'. The 'is_dx_pos' and 'is_dy_pos' parameters determine the
        directions of the shifts.

    Examples:
        >>> input_img = np.random.normal(0, 1, (3, 5))
        >>> dx = 2
        >>> dy = 1
        >>> is_dx_pos = True
        >>> is_dy_pos = True
        >>> shifted_img = shift_img_xy(input_img, dx, dy, is_dx_pos, is_dy_pos)
    """
    shifted_img = shift_img_x(img, dx, is_dx_pos=is_dx_pos)
    shifted_img = shift_img_y(shifted_img, dy, is_dy_pos=is_dy_pos)
    return shifted_img


####################################################################################################
#                                           Other                               			       #
####################################################################################################


def normalize_psd(original_magn, modified_magn):
    """
    Calculate the normalization coefficient for power spectral densities (PSD).

    This function computes the normalization factor needed to adjust the PSD of 
    a modified signal (represented by its magnitudes) to match the PSD of the original signal.

    Parameters:
    original_magn (array_like): Magnitudes of the original signal.
    modified_magn (array_like): Magnitudes of the modified signal.

    Returns:
    float: The square root of the ratio of the sum of squares of original magnitudes
           to the sum of squares of modified magnitudes.
    """
    original_psd_sum = np.sum(original_magn ** 2)
    modified_psd_sum = np.sum(modified_magn ** 2)
    psd_coef = original_psd_sum / modified_psd_sum
    return np.sqrt(psd_coef)


def show_images(*images, vrange=None, figsize=(10, 10), cmap='gray',
                aspect='equal', graphs_per_row=2):
    """
    Display multiple images in a grid layout using Matplotlib.

    Parameters:
        images (array_like): Variable number of images to display.
        vrange (tuple, optional): Range of values for normalizing luminance data (vmin, vmax).
        figsize (tuple, optional): Size of the figure (width, height in inch). Default is (10, 10).
        cmap (str, optional): Colormap used for displaying images. Default is 'gray'.
        aspect (str, optional): Aspect ratio of the images. Default is 'equal'. Can be 'auto',
                                a number, or 'equal'.
        graphs_per_row (int, optional): Number of images to display per row. Default is 2.

    Returns:
        tuple: A tuple containing the figure and axes array created by plt.subplots.

    Example:
        >>> img1 = np.random.rand(100, 100)
        >>> img2 = np.random.rand(100, 100)
        >>> f, axes = show_images(img1, img2, graphs_per_row=2)
        >>> plt.show()  # This will display two images side by side.
    """
    total_images = len(images)
    row_num = total_images // graphs_per_row + bool(total_images % graphs_per_row)
    col_num = total_images // row_num + bool(total_images % row_num)

    f, axes = plt.subplots(row_num, col_num, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for index, ax in enumerate(axes):
        if index < total_images:
            display_params = {'cmap': cmap, 'aspect': aspect}
            if vrange:
                display_params.update({'vmin': vrange[0], 'vmax': vrange[1]})
            im = ax.imshow(images[index], **display_params)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            ax.axis('off')

    plt.tight_layout()
    return f, axes


def show_surfaces(*surfaces, axes=None, vrange=None, colorscale='Thermal', showscale=True):
    """
    Plot multiple 3D surfaces using Plotly, with options for custom axes, color range, and
    color scale.

    Parameters:
        surfaces (array_like): Variable number of surfaces to plot. Each surface can be a
                               tuple (x, y, z) or just z.
        axes (tuple, optional): Tuple of x and y axes if only z values are provided in
                                surfaces. Default is None.
        vrange (tuple, optional): Color range for the surfaces as (min, max). If not provided,
                                  it's auto-calculated. Default is None.
        colorscale (str, optional): Color scale name for the surfaces. Default is 'Thermal'.
        showscale (bool, optional): Whether to show the color scale. Default is True.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure containing the surface plots.

    Example:
        >>> x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
        >>> y = x.copy().T
        >>> z = np.cos(x ** 2 + y ** 2)
        >>> fig = show_surfaces((x, y, z), vrange=(-1, 1), colorscale='Viridis')
        >>> fig.show()  # Displays a 3D surface plot with the Viridis color scale.
    """
    if vrange:
        cmin, cmax = vrange
    else:
        z_data = surfaces if axes else [surf[-1] for surf in surfaces]
        cmin, cmax = np.min(z_data), np.max(z_data)

    fig = go.Figure()
    for surf in surfaces:
        x, y, z = (*axes, surf) if axes else surf
        fig.add_trace(go.Surface(x=x, y=y, z=z, cmin=cmin, cmax=cmax,
                                 colorscale=colorscale, showscale=showscale))
        showscale = False  # Turn off showscale for subsequent surfaces
    return fig


def plot_graphs(*graphs, ax=None, figsize=None, grid=False, xlabel='', ylabel='', title=''):
    """
    Plot multiple graphs on a single Matplotlib axes.

    This function allows plotting multiple graphs, either as individual data series or as 
    (x, y) pairs. It provides options for customizing the figure size, grid, labels, and title.

    Parameters:
        graphs (array_like): Variable number of graphs to plot. Each graph can be a data series
                             or a tuple/list of (x, y).
        ax (matplotlib.axes.Axes, optional): Axes object on which to plot the graphs. If None,
                                             a new figure and axes are created.
        figsize (tuple, optional): Size of the figure (width, height in inches). If None,
                                   default size is used.
        grid (bool, optional): Whether to display a grid. Default is False.
        xlabel (str, optional): Label for the x-axis. Default is an empty string.
        ylabel (str, optional): Label for the y-axis. Default is an empty string.
        title (str, optional): Title of the plot. Default is an empty string.

    Returns:
        matplotlib.axes.Axes: The axes object with the plotted graphs.

    Example:
        >>> x = range(10)
        >>> y1 = [i**2 for i in x]
        >>> y2 = [i**3 for i in x]
        >>> ax = plot_graphs((x, y1), (x, y2), xlabel='X-axis', ylabel='Y-axis',
                             title='Example Plot')
        >>> plt.show()  # Displays the plot with two graphs.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for graph in graphs:
        ax.plot(*graph) if isinstance(graph, (list, tuple)) else ax.plot(graph)

    ax.grid(grid)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def open_img(filename, no_mean=True, grayscale=True):
    """
    Open an image file, with options to convert it to grayscale and subtract its mean.

    This function loads an image from a specified file, optionally converts it to grayscale,
    and subtracts the mean pixel value across the image if specified.

    Parameters:
        filename (str): Path to the image file.
        no_mean (bool, optional): If True, subtracts the mean pixel value from the
                                  image. Default is True.
        grayscale (bool, optional): If True, converts the image to grayscale. Default is True.

    Returns:
        numpy.ndarray: The processed image as a NumPy array.

    Example:
        >>> img_array = open_img('path/to/image.jpg', no_mean=False, grayscale=True)
        >>> print(img_array.shape)  # Prints the shape of the grayscale image array.
    """
    img = Image.open(filename)
    img_array = np.array(img, dtype=float)

    if grayscale:
        img_array = img.convert('L')

    if no_mean:
        img_array -= np.mean(img_array)

    return img_array


def rmse(arr1, arr2, weight_arr=None, normalize=False):
    """
    Calculate the Root Mean Squared Error (RMSE) between two arrays.

    This function computes the RMSE between two input arrays, with an option to apply 
    weights to the calculation. If normalization is enabled, the RMSE is normalized 
    by the mean of the weights.

    Parameters:
        arr1 (numpy.ndarray): First input array.
        arr2 (numpy.ndarray): Second input array, must be the same shape as arr1.
        weight_arr (numpy.ndarray, optional): Weights for each element in arr1 and arr2.
                                              Default is equal weights.
        normalize (bool, optional): If True, normalizes the RMSE by the mean of the weights.
                                    Default is False.

    Returns:
        float: The computed RMSE value.

    Raises:
        ValueError: If the shapes of arr1, arr2, and weight_arr (if provided) do not match.

    Example:
        >>> arr1 = np.array([1, 2, 3])
        >>> arr2 = np.array([1, 3, 5])
        >>> print(rmse(arr1, arr2))  # Basic RMSE calculation without weights.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Shape mismatch between input arrays.")

    if weight_arr is None:
        weight_arr = np.ones_like(arr1)
    elif arr1.shape != weight_arr.shape:
        raise ValueError("Shape mismatch between input arrays and weight array.")

    diff = weight_arr * (arr1 - arr2) ** 2
    mse = np.sum(diff) / arr1.size
    mse = mse / np.mean(weight_arr) if normalize else mse
    return np.sqrt(mse)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1,
                       length=100, fill='█', print_end='\r'):
    """
    Print a dynamic progress bar in the console.

    This function generates a text-based progress bar, useful for tracking the progress
    of iterative processes. It supports customization of its appearance and format.

    Parameters:
        iteration (int): Current iteration (should be <= total).
        total (int): Total iterations.
        prefix (str, optional): Prefix string. Default is an empty string.
        suffix (str, optional): Suffix string. Default is an empty string.
        decimals (int, optional): Positive number of decimals in percent complete. Default is 1.
        length (int, optional): Character length of the bar. Default is 100.
        fill (str, optional): Bar fill character. Default is a solid block ('█').
        print_end (str, optional): End character (e.g., '\r', '\n'). Default is '\r'.

    Example:
        >>> import time
        >>> total = 10
        >>> for i in range(total):
        >>>     time.sleep(0.1)  # Simulate some work.
        >>>     print_progress_bar(i + 1, total, prefix='Progress:', suffix='Complete', length=50)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def find_index_by_val(arr, target_val, find_last=True):
    """
    Find the index of a target value in an array, with an option to find the first or
    last occurrence.

    Parameters:
        arr (list): The array to search through.
        target_val (any): The value to search for in the array.
        find_last (bool, optional): If True, returns the index of the last occurrence of the
                                    target value.
                                    If False, returns the index of the first occurrence.
                                    Default is True.

    Returns:
        int: The index of the target value in the array. Returns -1 if the value is not found.

    Example:
        >>> arr = [1, 2, 3, 4, 2, 5]
        >>> print(find_index_by_val(arr, 2))  # Finds the last index (4) of 2
        >>> print(find_index_by_val(arr, 2, find_last=False))  # Finds the first index (1) of 2
    """
    idx = -1
    for index, val in enumerate(arr):
        if val == target_val:
            idx = index
            if not find_last:
                break
    return idx


def lin_regression(x, y):
    """
    Calculate the coefficients of a linear regression line between two datasets.

    This function computes the slope (a) and intercept (b) of the linear regression line 
    that best fits the datasets x (independent variable) and y (dependent variable) based on
    the least squares method.

    Parameters:
        x (numpy.ndarray): Array representing the independent variable (e.g., restored image).
        y (numpy.ndarray): Array representing the dependent variable (e.g., original image).

    Returns:
        tuple: A tuple (a, b) where 'a' is the slope and 'b' is the intercept of
        the regression line.

    Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>> slope, intercept = lin_regression(x, y)
        >>> print(slope, intercept)  #Expected output: (2.0, 0.0) for a perfect linear relationship.
    """
    num = np.mean(x * y) - np.mean(x) * np.mean(y)
    denum = np.mean(x ** 2) - np.mean(x) ** 2
    a = num / denum
    b = np.mean(y) - a * np.mean(x)
    return a, b


def gaussian(x_vals, y_vals, std):
    """
    Calculate the Gaussian function for given x and y values with a specified standard deviation.

    The function calculates the Gaussian value based on the sum of x and y values. 
    This is not a typical 2D Gaussian distribution, which would consider x and y independently.

    Parameters:
        x_vals (numpy.ndarray): An array of x values.
        y_vals (numpy.ndarray): An array of y values. It should be the same shape as x_vals.
        std (float): The standard deviation for the Gaussian function.

    Returns:
    numpy.ndarray: The calculated Gaussian values.

    Example:
        >>> x = np.array([0, 1, 2])
        >>> y = np.array([0, 1, 2])
        >>> gaussian_values = gaussian(x, y, 1)
        >>> print(gaussian_values)
    """
    arg = x_vals + y_vals
    exp = np.exp(-(arg ** 2) / (2 * std ** 2))
    return exp


####################################################################################################
#                                        Lattice noise utils                                       #
####################################################################################################


def graph_mesh(x_min, x_max, x_step):
    """
    Generate a 1D NumPy array (mesh) with values from x_min to x_max (inclusive) at
    specified intervals.

    This function is useful for creating coordinate grids, especially for graphing purposes, 
    where you need a range of values at a certain step size.

    Parameters:
        x_min (float or int): The starting value of the range.
        x_max (float or int): The ending value of the range. The function includes this value in
                              the output if possible.
        x_step (float or int): The step size between each value in the range.

    Returns:
        numpy.ndarray: A NumPy array containing values from x_min to x_max at intervals of x_step.

    Example:
        >>> mesh = graph_mesh(0, 10, 2)
        >>> print(mesh)  # Output: [ 0  2  4  6  8 10]
    """
    mesh = np.arange(x_min, x_max + x_step, x_step)
    return mesh


def qubic_interp(x):
    """
    Perform cubic Hermite interpolation on a given input.

    This function applies a cubic interpolation formula to the input value 'x'.
    It's a common spline interpolation method used in smoothing and generating curves.

    Parameters:
        x (float or numpy.ndarray): Input value or array of values for interpolation.

    Returns:
        float or numpy.ndarray: The result of cubic Hermite interpolation on 'x'.

    Example:
        >>> print(qubic_interp(0.5))  # Interpolate at the midpoint
        >>> print(qubic_interp(np.array([0, 0.5, 1])))  # Interpolate at several points
    """
    return x * x * (3 - 2 * x)


def fifth_order_interp(x):
    """
    Perform fifth-order polynomial interpolation on a given input.

    This function applies a fifth-order polynomial formula to the input value 'x'.
    It's a more complex spline interpolation method than cubic interpolation, providing 
    a smoother transition commonly used in graphics and numerical simulations.

    Parameters:
        x (float or numpy.ndarray): Input value or array of values for interpolation.

    Returns:
        float or numpy.ndarray: The result of fifth-order polynomial interpolation on 'x'.

    Example:
        >>> print(fifth_order_interp(0.5))  # Interpolate at the midpoint
        >>> print(fifth_order_interp(np.array([0, 0.5, 1])))  # Interpolate at several points
    """
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3


def gen_value_noise_1d(outp_noise_size, iterm_mesh_size, octaves_num, persistence, func=None):
    """
    Generate 1D value noise over multiple octaves.

    This function creates 1D value noise, typically used in procedural generation. It generates
    noise for a specified number of octaves, each with decreasing amplitude influenced by the 
    persistence value. A custom interpolation function can be applied to each octave.

    Parameters:
        outp_noise_size (int): The size of the output noise array.
        iterm_mesh_size (int): The size of the mesh (grid) for the first octave.
        octaves_num (int): The number of octaves to generate.
        persistence (float): The factor by which the amplitude of each octave decreases.
        func (callable, optional): Custom function for interpolating noise values. Default is None.

    Returns:
        tuple: A tuple containing two lists - one for the harmonics (noise values) of each octave
               and another for the random values used to generate these harmonics.

    Example:
        >>> harmonics, random_values = gen_value_noise_1d(1024, 256, 4, 0.5)
        >>> print(len(harmonics))  # Number of octaves
        >>> print(len(random_values))  # Number of random value sets
    """
    random_values = []
    harmonics = []
    amplitude = persistence

    for _ in range(octaves_num):
        rv = amplitude * np.random.rand(outp_noise_size // iterm_mesh_size + 1)
        octave = gen_single_octave_1d(rv, outp_noise_size, iterm_mesh_size, func=func)

        random_values.append(rv)
        harmonics.append(octave)
        iterm_mesh_size //= 2
        amplitude *= persistence

    return harmonics, random_values


def gen_single_octave_1d(random_values, size, grid_size, func=None):
    """
    Generate a single octave of 1D noise.

    This function creates a single octave of 1D noise using linear interpolation between 
    random values. An optional custom interpolation function can be provided for different 
    smoothing effects.

    Parameters:
        random_values (numpy.ndarray): An array of random values used for noise generation.
        size (int): The size of the output noise map.
        grid_size (int): The size of the grid cells used for interpolation.
        func (callable, optional): Custom function for interpolating noise values. Default is None.

    Returns:
        numpy.ndarray: A NumPy array containing the generated noise map.

    Example:
        >>> random_values = np.random.rand(11)
        >>> noise_map = gen_single_octave_1d(random_values, 100, 10)
        >>> print(noise_map.shape)  # Output: (101,)
    """
    noise_map = np.zeros(size + 1)

    for x in range(size):
        cell_x = x // grid_size
        local_x = (x % grid_size) / grid_size
        left = random_values[cell_x]
        right = random_values[cell_x + 1]
        smooth_x = func(local_x) if func else local_x
        interp = left * (1 - smooth_x) + right * smooth_x
        noise_map[x] += interp

    noise_map[-1] = random_values[-1]
    return noise_map


def gen_value_noise_2d(outp_noise_shape, iterm_mesh_shape, octave_nums, persistence, func=None):
    """
    Generate 2D value noise over multiple octaves.

    This function creates 2D value noise for a specified number of octaves, each with 
    decreasing amplitude influenced by the persistence value. A custom interpolation 
    function can be applied to each octave for different smoothing effects.

    Parameters:
        outp_noise_shape (tuple): The shape (height, width) of the output noise map.
        iterm_mesh_shape (tuple): The shape (grid height, grid width) for the first octave.
        octave_nums (int): The number of octaves to generate.
        persistence (float): The factor by which the amplitude of each octave decreases.
        func (callable, optional): Custom function for interpolating noise values. Default is None.

    Returns:
        list: A list containing the harmonics (noise values) of each octave.

    Example:
        >>> outp_shape = (256, 256)
        >>> mesh_shape = (32, 32)
        >>> harmonics = gen_value_noise_2d(outp_shape, mesh_shape, 4, 0.5)
        >>> print(len(harmonics))  # Output: 4
    """
    harmonics = []
    random_values = []
    h, w = outp_noise_shape
    gr_h, gr_w = iterm_mesh_shape
    amplitude = persistence

    for _ in range(octave_nums):
        rv = amplitude * np.random.rand(h // gr_h + 1, w // gr_w + 1)
        octave = gen_single_octave_2d(rv, outp_noise_shape, (gr_h, gr_w), func=func)

        random_values.append(rv)
        harmonics.append(octave)
        gr_h, gr_w = gr_h // 2, gr_w // 2
        amplitude *= persistence

    # noise_map = np.sum(harmonics, axis=0)
    # noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    return harmonics


def gen_single_octave_2d(random_values, outp_noise_shape, iterm_mesh_shape, func=None):
    """
    Generate a single octave of 2D noise.

    This function creates a single octave of 2D noise using bilinear interpolation 
    between points in a grid of random values. An optional custom interpolation function 
    can be provided for different smoothing effects.

    Parameters:
        random_values (numpy.ndarray): A 2D array of random values used for noise generation.
        outp_noise_shape (tuple): The shape (height, width) of the output noise map.
        iterm_mesh_shape (tuple): The shape (grid height, grid width) for interpolation.
        func (callable, optional): Custom function for interpolating noise values. Default is None.

    Returns:
        numpy.ndarray: A NumPy array containing the generated 2D noise map.

    Example:
        >>> random_values = np.random.rand(11, 11)
        >>> noise_map = gen_single_octave_2d(random_values, (100, 100), (10, 10))
        >>> print(noise_map.shape)  # Output: (101, 101)
    """
    h, w = outp_noise_shape
    gr_h, gr_w = iterm_mesh_shape
    noise_map = np.zeros((h + 1, w + 1))

    for y in range(h):
        for x in range(w):
            cell_x, cell_y = x // gr_w, y // gr_h
            local_x, local_y = (x % gr_w) / gr_w, (y % gr_h) / gr_h

            top_left = random_values[cell_y, cell_x]
            top_right = random_values[cell_y, cell_x + 1]
            bottom_left = random_values[cell_y + 1, cell_x]
            bottom_right = random_values[cell_y + 1, cell_x + 1]

            smooth_x = func(local_x) if func else local_x
            smooth_y = func(local_y) if func else local_y

            interp_top = top_left * (1 - smooth_x) + top_right * smooth_x
            interp_bottom = bottom_left * (1 - smooth_x) + bottom_right * smooth_x
            noise_map[y, x] = interp_top * (1 - smooth_y) + interp_bottom * smooth_y
            noise_map[-1, x] = bottom_left * (1 - smooth_x) + bottom_right * smooth_x
        noise_map[y, -1] = top_right * (1 - smooth_y) + bottom_right * smooth_y
    noise_map[-1, -1] = random_values[-1, -1]
    return noise_map


####################################################################################################
#                                       Other (Deprecation candidates)                      	   #
####################################################################################################


def fit_clement(x_freq, y_freq, alpha1, alpha2, eta=0.1, angle=3):
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y)
    fp = abs(angle * x + y)
    f = np.sqrt((1 - eta) * f ** 2 + eta * fp ** 2)
    f = (1 + abs(f)) ** (-alpha1) + 0.02 * (1 + abs(f)) ** (-alpha2)
    return f


def fit_clement_new(x_freq, y_freq, alpha1, eta=0.1, angle=1):
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y)
    fp = abs(y - angle * x)
    f = np.sqrt((1 - eta) * f ** 2 + eta * fp ** 2)
    f = 1 / ((1 + abs(f)) ** alpha1)
    return f


def surrogates(x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"):
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)
    return xs


def gen_cloud(x_size, y_size, factor=2.4):
    xx = np.linspace(-x_size / 2, x_size / 2, x_size)
    yy = np.linspace(-y_size / 2, y_size / 2, y_size)
    whitenoise = np.random.normal(0, 1, (y_size, x_size))
    cloud_freq = find_ft_2d(whitenoise)
    kernel = freq_pink_filter_2d(xx, yy, factor=factor)
    cloud_freq_filtered = cloud_freq * kernel
    cloud_spatial = find_ift_2d(cloud_freq_filtered).real
    return normalize_img(cloud_spatial)


def lin_phase(start, end, size):
    pos_freq = np.linspace(start, end, size // 2)
    neg_freq = -pos_freq[::-1]
    if size % 2:
        neg_freq = np.append(neg_freq, [0])
        freq = np.append(neg_freq, pos_freq)
    else:
        freq = np.append(neg_freq, pos_freq)
    return freq
