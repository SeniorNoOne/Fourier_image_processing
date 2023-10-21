import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import random

from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm


###################################################################################################################
#                                              Array utils                                                        #
###################################################################################################################


def get_slice(arr, rows, cols):
    '''
    Extract a slice of values from a 2D NumPy array.

    Parameters:
        arr (numpy.ndarray): The input 2D array from which to extract the slice.
        rows (list): A list of row indexes for the slice.
        cols (list): A list of column indexes for the slice.

    Returns:
        numpy.ndarray: A 1D array containing the extracted slice values.

    Raises:
        ValueError: If the lengths of 'rows' and 'cols' are different.
    '''

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
    '''
    Normalize a NumPy array to the [0, 1] range.

    Parameters:
        arr (numpy.ndarray): The input array to be normalized.
        zero_padding (bool, optional): If True, normalize without zero-padding.
        offset_coef (float, optional): Coefficient to calculate the offset when zero_padding is False.

    Returns:
        numpy.ndarray: The normalized array.

    Notes:
        If zero_padding is True, the normalization is performed in the range [0, 1] without zero-padding.
        If zero_padding is False, an offset is added to ensure positive values and normalize to [0, 1].

    Examples:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> normalize(arr)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> normalize(arr, zero_padding=True)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    '''

    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if zero_padding:
        new_arr = (arr - min_val) / (max_val - min_val)
    else:
        offset = offset_coef * abs(min_val)
        new_arr = (arr - min_val + offset) / (max_val - min_val + offset)

    return new_arr


def normalize_img(arr):
    '''
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
    '''
    
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    if arr_min >= 0 and arr_max <= 255: 
        return arr
    else:
        arr_norm = 255 * (arr - arr_min) / (arr_max - arr_min)
        return arr_norm


def threshold_arr_1d(arr, th_val, use_abs=False):
    '''
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
    '''
	
    th_val = abs(th_val) if use_abs else th_val
    for i in range(len(arr)):
        if arr[i] < th_val:
            arr[i] = th_val
    return arr
    
    
def threshold_arr_2d(arr, th_val, use_abs=False):
    '''
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
    '''
	
    for i in range(arr.shape[0]):
        arr[i, :] = threshold_arr_1d(arr[i, :], th_val, use_abs)
    return arr


def clip_graph(x_arr, y_arr, x_min=0, x_max=100, y_min=0, y_max=100):  
    '''
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
    '''
	
    x_mask = (x_arr >= x_min) & (x_arr <= x_max)
    y_mask = (y_arr >= y_min) & (y_arr <= y_max)

    # Apply clipping to both x and y arrays
    x_arr = x_arr[x_mask & y_mask]
    y_arr = y_arr[x_mask & y_mask]
	
    return x_arr, y_arr


###################################################################################################################
#                                                    FFT utils                                                    #
###################################################################################################################


def find_ft_1d(arr):
    ft = np.fft.fft(arr)
    return np.fft.fftshift(ft)


def find_ift_1d(arr):
    ift = np.fft.ifftshift(arr)
    return np.fft.ifft(ift).real


def find_ft_2d(arr):
    ft = np.fft.fft2(arr)
    return np.fft.fftshift(ft)


def find_ift_2d(arr):
    ift = np.fft.ifftshift(arr)
    return np.fft.ifft2(ift).real


def freq_numbers_1d(size):
    if size % 2:
        return np.arange(-(size // 2), size // 2 + 1, 1) 
    else:
        return np.arange(-(size // 2), size // 2, 1) 
    

def freq_arr_1d(size, step=1):
    freq = freq_numbers_1d(size) 
    return freq / step / size


def freq_numbers_2d(shape):
    y_size, x_size = shape
    y_freq_numbers = freq_numbers_1d(y_size)
    x_freq_numbers = freq_numbers_1d(x_size)
    return x_freq_numbers, y_freq_numbers
    

def freq_arr_2d(shape, x_step=1, y_step=1):
    y_size, x_size = shape
    y_freq_numbers, x_freq_numbers = freq_numbers_2d(shape) 
    x_freq = x_freq_numbers / x_step / x_size
    y_freq = y_freq_numbers / y_step / y_size
    return y_freq, x_freq
	
	
def freq_mesh_2d(shape):
    x_freq_numbers, y_freq_numbers = freq_numbers_2d(shape) 
    x_mesh, y_mesh = np.meshgrid(x_freq_numbers, y_freq_numbers)
    return x_mesh, y_mesh


###################################################################################################################
#                                                Frequency domain filters                                         #
###################################################################################################################


def freq_filter_2d(x_freq, y_freq):
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y)
    return f


# should be deprecated or changed
"""
def freq_filter(x_freq, y_freq, factor=2.4):
    eps = 10 ** -8
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y)
    f = f ** factor + eps
    return normalize(1 / f)
"""


def freq_pink_filter_2d(x_freq, y_freq, factor=1, x_stretch=1, y_stretch=1, no_mean=False):
    x_freq, y_freq = np.abs(x_freq), np.abs(y_freq)
    fr = freq_filter_2d(x_freq / x_stretch, y_freq / y_stretch)
    f = 1 / np.where(fr <= 1, 1, fr)
    f = f ** factor
    if no_mean:
        f_mask = np.abs(fr) < 1
        f[f_mask] = 0
    return f


def freq_pink_filter_1d(x_freq, factor=0.5, no_mean=False):
    x_freq = np.abs(x_freq)
    f = 1 / np.where(x_freq <= 1, 1, x_freq)
    f = f ** factor
    if no_mean:
        f_mask = np.abs(x_freq) < 1
        f[f_mask] = 0
    return f


def freq_pink_filter_1d_old(x_freq, factor=0.5, no_mean=False):
    x_freq = np.abs(x_freq)
    f = 1 / (1 + np.abs(x_freq))
    f = f ** factor
    if no_mean:
        f_mask = np.abs(x_freq) < 1
        f[f_mask] = 0
    return f


def freq_pink_filter_2d_old(x_freq, y_freq, factor=1, x_stretch=1, y_stretch=1, no_mean=False):
    f = freq_filter_2d(x_freq / x_stretch, y_freq / y_stretch)
    f = 1 / (1 + f)
    f = f ** factor
    f = np.where(f==1, 0, f) if no_mean else f
    return f


# replaced by no_mean flag in freq_filter_1d
"""
def freq_filter_1d_a(x_freq, factor=1):
    f = 1 / np.where(x_freq == 0, 1, np.abs(x_freq))
    f = f ** factor
    f[len(x_freq) // 2] = 0
    return f
"""


def freq_sharp_round_filter_2d(x_freq, y_freq, radius, low_pass_filter=True):
    check = np.less if low_pass_filter else np.greater
    f = freq_filter_2d(x_freq, y_freq)
    f = np.where(check(f, radius), 1, 0)
    return f


def freq_sharp_square_filter_2d(x_freq, y_freq, width, angle=0, low_pass_filter=True):
    check = np.less if low_pass_filter else np.greater
    angle_radians = math.radians(angle)    
    x, y = np.meshgrid(x_freq, y_freq)
    rotated_x = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    rotated_x = np.where(check(np.abs(rotated_x), width), 1, 0)
    
    rotated_y = x * np.sin(angle_radians) + y * np.cos(angle_radians)  
    rotated_y = np.where(check(np.abs(rotated_y), width), 1, 0)

    f = rotated_x * rotated_y 
    return f


###################################################################################################################
#                                           Spatial domain utils (NOT REWORKED)                                   #
###################################################################################################################


def spatial_smooth_filter(x_size, y_size, depth, horiz=True):
    values = np.linspace(0, 1, depth)
    values = 6 * values ** 5 - 15 * values ** 4 + 10 * values ** 3
    values = 1 - values
    if horiz:
        kernel = np.tile(values, (y_size, 1))
    else:
        kernel = values[:, np.newaxis] * np.ones((1, x_size))   
    return kernel


def make_img_transition_x(img, depth, is_dx_pos=True, outter_smooth=False):
    y_size, x_size = img.shape
    add_img = gen_cloud(x_size + depth, y_size)   
    transition_kernel = spatial_smooth_filter(x_size, y_size, depth)     
    
    img_copy = np.copy(img)
    if is_dx_pos:
        img_copy[:, -depth:] = img[:, -depth:] * transition_kernel + \
                               add_img[:, :depth] * (1 - transition_kernel)
        img_copy = np.concatenate((img_copy, add_img[:, depth:]), axis=1)
    else:
        transition_kernel = np.fliplr(transition_kernel)
        img_copy[:, :depth] = img[:, :depth] * transition_kernel + \
                              add_img[:, -depth:] * (1 - transition_kernel)  
        img_copy = np.concatenate((add_img[:, 0:-depth], img_copy), axis=1) 
    
    if outter_smooth:
        add_img = gen_cloud(2 * depth, y_size)   
        transition_kernel = spatial_smooth_filter(x_size, y_size, depth) 
        transition_kernel = np.fliplr(transition_kernel)
        img_copy[:, :depth] = img_copy[:, :depth] * transition_kernel + \
                              add_img[:, :depth] * (1 - transition_kernel)  
        transition_kernel = np.fliplr(transition_kernel)
        img_copy[:, -depth:] = img_copy[:, -depth:] * transition_kernel + \
                               add_img[:, -depth:] * (1 - transition_kernel)
    return img_copy
    

def make_img_transition_y(img, depth, is_dy_pos=True, outter_smooth=False):
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
        img_copy[:depth, :] = img[:depth, :] * transition_kernel + \
                              add_img[-depth:, :] * (1 - transition_kernel)  
        img_copy = np.concatenate((add_img[:-depth, :], img_copy), axis=0)
    
    if outter_smooth:
        add_img = gen_cloud(x_size, 2 * depth)   
        transition_kernel = spatial_smooth_filter(x_size, y_size, depth, horiz=False) 
        transition_kernel = np.flipud(transition_kernel)
        img_copy[:depth, :] = img_copy[:depth, :] * transition_kernel + \
                              add_img[:depth, :] * (1 - transition_kernel)   
        transition_kernel = np.flipud(transition_kernel)
        img_copy[-depth:, :] = img_copy[-depth:, :] * transition_kernel + \
                               add_img[-depth:, :] * (1 - transition_kernel)  
    return img_copy


def make_img_transition_xy(img, depth, is_dx_pos=True, is_dy_pos=True, outter_smooth=False):
    new_img = make_img_transition_x(img, depth, is_dx_pos=is_dx_pos, outter_smooth=outter_smooth)
    new_img = make_img_transition_y(new_img, depth, is_dy_pos=is_dy_pos, outter_smooth=outter_smooth)
    return new_img


def shift_img_x(img, dx, is_dx_pos=True): 
    _, width = img.shape
    
    # Wind blows in negative X dirrection
    if is_dx_pos:
        c1 = img[:, -dx:]
        c2 = img[:, :-dx]
    else:
        c1 = img[:, dx:]
        c2 = img[:, :dx]
    return np.concatenate((c1, c2), axis=1)


def shift_img_y(img, dy, is_dy_pos=True):
    height, _ = img.shape
    
    # Wind blows in negative Y dirrection
    if is_dy_pos:
        c1 = img[-dy:, :]
        c2 = img[:-dy, :]
    else:
        c1 = img[dy:, :]
        c2 = img[:dy, :]
    return np.concatenate((c1, c2), axis=0)

    
def shift_img_xy(img, dx, dy, is_dx_pos=True, is_dy_pos=True):
    height, width = img.shape
    shifted_img = shift_img_x(img, dx, is_dx_pos=is_dx_pos)
    shifted_img = shift_img_y(shifted_img, dy, is_dy_pos=is_dy_pos)
    return shifted_img


###################################################################################################################
#                                                Other (NOT REWORKED)                                             #
###################################################################################################################


def normalize_psd(original_magn, modified_magn):
    original_psd = original_magn ** 2
    modified_psd = modified_magn ** 2
    psd_coef = np.sum(original_psd) / np.sum(modified_psd)
    return psd_coef ** 0.5


def fit_clement_new(x_freq, y_freq, alpha1, eta=0.1, angle=1):
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y)
    fp = abs(y - angle * x)
    f = np.sqrt((1 - eta) * f ** 2 + eta * fp ** 2)    
    f = 1 / ((1 + abs(f)) ** alpha1)
    return f


def fit_clement(x_freq, y_freq, alpha1, alpha2, eta=0.1, angle=3):
    x, y = np.meshgrid(x_freq, y_freq)
    f = np.hypot(x, y )
    fp = abs(angle * x  + y)
    f = np.sqrt((1 - eta) * f ** 2 + eta * fp ** 2)    
    f = (1 + abs(f)) ** (-alpha1) + 0.02 * (1 + abs(f)) ** (-alpha2)
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


def adjust_freq_1d(img):
    return np.append(img, abs(img[0]))


def adjust_img_1d(img):
    return np.append(img, img[0])
        
        
def gen_cloud(x_size, y_size, factor=2.4):
    xx = np.linspace(-x_size / 2, x_size / 2, x_size)
    yy = np.linspace(-y_size / 2, y_size / 2, y_size)
    whitenoise = np.random.normal(0, 1, (y_size, x_size))
    cloud_freq = find_ft_2d(whitenoise)  
    kernel = freq_pink_filter_2d(xx, yy, factor=factor)
    cloud_freq_filtered = cloud_freq * kernel
    cloud_spatial = find_ift_2d(cloud_freq_filtered).real
    return normalize_img(cloud_spatial)


# Reworked
def show_images(*images, vrange=None, x_fig_size=10, y_fig_size=10, cmap='gray', graphs_per_row=2):
    row_num = len(images) // graphs_per_row + 1 if len(images) % graphs_per_row else len(images) // graphs_per_row
    col_num = len(images) // row_num + 1 if len(images) % row_num else len(images) // row_num
    images_len = len(images)
    
    f, axes = plt.subplots(row_num, col_num, figsize=(x_fig_size, y_fig_size))  
    if row_num == 1 and col_num == 1:
        axes = np.array([axes])
    
    for index, ax in enumerate(axes.flatten()):
        if index < images_len:
            if vrange:
                vmin, vmax = vrange
                im = ax.imshow(images[index], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                
            else:
                im = ax.imshow(images[index], cmap=cmap, aspect='equal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            ax.axis('off')

    plt.tight_layout()
    return f, axes

    
def show_surfaces(*surfaces, axes=None, cmap=None, colorscale='Thermal', showscale=True):  
    if cmap:
        cmin, cmax = cmap
    else:
        if axes:
            z_data = surfaces
        else:
            z_data = [surf[-1] for surf in surfaces]
        cmin, cmax = np.min(z_data), np.max(z_data)
    
    fig = go.Figure()
    for surf in surfaces:
        if axes: 
            x, y = axes
            z = surf
        else:
            x, y, z = surf
        fig.add_trace(go.Surface(x=x, y=y, z=z, cmin=cmin, cmax=cmax, colorscale=colorscale, showscale=showscale))
        showscale = False
    return fig
    
	
def plot_graphs(*graphs, ax=None, figsize=None, grid=False, xlabel='', ylabel='', title=''):   
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    for graph in graphs:        
        if isinstance(graph, (list, tuple)):
            ax.plot(*graph)
        else:
            ax.plot(graph)
    
    ax.grid(grid)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
        
    return ax


def lin_regression(x, y):
    # y - original img
    # x - restored img
    num = np.mean(x * y) - np.mean(x) * np.mean(y)
    denum = np.mean(x ** 2) - np.mean(x) ** 2
    a = num / denum
    b = np.mean(y) - a * np.mean(x)
    return a, b


def lin_phase(start, end, size):
    pos_freq = np.linspace(start, end, size // 2)
    neg_freq = -pos_freq[::-1]
    if size % 2: 
        neg_freq = np.append(neg_freq, [0])
        freq = np.append(neg_freq, pos_freq)
    else: 
        freq = np.append(neg_freq, pos_freq)
    return freq


def open_img(filename, no_mean=True, grayscale=True):
    img = Image.open(filename)
    
    if grayscale:
        img = img.convert('L')
    
    if no_mean:
        img -= np.mean(img)
    
    return np.array(img)
	

###################################################################################################################
#                                          Latice noise utils                                                     #
###################################################################################################################


def graph_mesh(x_min, x_max, x_step):
    mesh = np.arange(x_min, x_max + 1, x_step)
    return mesh


def qubic_interp(x):
    return x * x * (3 - 2 * x)


def fifth_order_interp(x):
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3


def gen_value_noise_1d(outp_noise_size, iterm_mesh_size, octaves_num, persistence, func=None):
    random_values = []
    harmonics = []
    amplitude = persistence

    for _ in range(octaves_num):
        rv = amplitude * np.random.rand(outp_noise_size // iterm_mesh_size + 1)
        octave = gen_single_octave_1d(rv, outp_noise_size, iterm_mesh_size, func=func)

        random_values.append(rv)
        harmonics.append(octave)
        iterm_mesh_size = iterm_mesh_size // 2
        amplitude *= persistence

    return harmonics, random_values


def gen_single_octave_1d(random_values, size, grid_size, func=None):
    num_cells_x = size // grid_size
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


def gen_single_octave_2d(random_values, outp_noise_shape, iterm_mesh_shape, func=None):
    h, w = outp_noise_shape
    gr_h, gr_w = iterm_mesh_shape
    num_cells_x = w // gr_w
    num_cells_y = h // gr_h
    noise_map = np.zeros((h + 1, w + 1))
    
    for y in range(h):
        for x in range(w):
            cell_x = x // gr_w
            cell_y = y // gr_h

            local_x = (x % gr_w) / gr_w
            local_y = (y % gr_h) / gr_h

            top_left = random_values[cell_y, cell_x]
            top_right = random_values[cell_y, cell_x + 1]
            bottom_left = random_values[cell_y + 1, cell_x]
            bottom_right = random_values[cell_y + 1, cell_x + 1]

            if func is None:
                smooth_x = local_x
                smooth_y = local_y
            else:
                smooth_x = func(local_x)
                smooth_y = func(local_y)

            interp_top = top_left * (1 - smooth_x) + top_right * smooth_x
            interp_bottom = bottom_left * (1 - smooth_x) + bottom_right * smooth_x
            noise_map[y, x] += interp_top * (1 - smooth_y) + interp_bottom * smooth_y          
            noise_map[-1, x] = bottom_left * (1 - smooth_x) + bottom_right * smooth_x
        noise_map[y, -1] = top_right * (1 - smooth_y) + bottom_right * smooth_y
    noise_map[-1, -1] = random_values[-1, -1]
    return noise_map


def gen_value_noise_2d(outp_noise_shape, iterm_mesh_shape, octave_nums, persistence, func=None):
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


###################################################################################################################
#                                                    FFT utils                                                    #
###################################################################################################################


def rmse(arr1, arr2, weight_arr=None, normalize=False):
    if arr1.shape != arr2.shape:
        raise ValueError("Shape mismatch between input arrays.")
    
    if weight_arr is None:
        weight_arr = np.ones(arr1.shape)
    elif arr1.shape != weight_arr.shape:
        raise ValueError("Shape mismatch between input arrays and weight array.")
    
    weight_mean = np.mean(weight_arr) 
    diff = weight_arr * (arr1 - arr2) ** 2
    mse = np.sum(diff) / arr1.size
    mse = mse / weight_mean if normalize else mse
    return np.sqrt(mse)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals = 1, 
                       length = 100, fill = '█', print_end='\r'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
        
def gaussian(x_vals, y_vals, std):
    arg = x_vals + y_vals
    exp = np.exp(-(arg ** 2) / 2 / std)
    return exp


def find_index_by_val(arr, target_val, find_last=True):
    idx = 0
    for index, val in enumerate(arr):
        if val == target_val:
            idx = index
    return idx
