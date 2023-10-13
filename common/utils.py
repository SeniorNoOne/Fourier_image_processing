import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import scipy
import math
import matplotlib.animation as animation

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


###################################################################################################################
#                                              Array utils                                                        #
###################################################################################################################


def normalize(arr, zero_padding=False, offset_coef=0.001):
    min_val = np.min(arr)
    max_val = np.max(arr)
    offset = offset_coef * abs(min_val)
    new_arr = arr - min_val if zero_padding else arr - min_val + offset
    d = (max_val - min_val) if zero_padding else (max_val - min_val + offset)
    new_arr = new_arr / d
    return new_arr


def normalize_img(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_min > 0 and arr_max < 255: 
        return arr
    else:
        arr_norm = 255 * (arr - arr_min) / (arr_max - arr_min)
        return arr_norm


def threshold_arr_1d(arr, th_val, use_abs=False):
    th_val = abs(th_val) if use_abs else th_val
    for i in range(len(arr)):
        if arr[i] < th_val:
            arr[i] = th_val
    return arr
    
    
def threshold_arr_2d(arr, th_val, use_abs=False):
    for i in range(arr.shape[0]):
        arr[i, :] = threshold_arr_1d(arr[i, :], th_val, use_abs)
    return arr


def clip_graph(x_arr, y_arr, x_min=0, x_max=100, y_min=0, y_max=100):  
    x_mask = (x_arr < x_max) & (x_arr > x_min)
    x_arr = x_arr[x_mask]
    y_arr = y_arr[x_mask]
    
    y_mask = (y_arr < y_max) & (y_arr > y_min)
    y_arr = y_arr[y_mask] 
    x_arr = x_arr[y_mask]
    
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
    return y_freq_numbers, x_freq_numbers
    

def freq_arr_2d(shape, x_step=1, y_step=1):
    y_size, x_size = shape
    y_freq_numbers, x_freq_numbers = freq_numbers_2d(shape) 
    x_freq = x_freq_numbers / x_step / x_size
    y_freq = y_freq_numbers / y_step / y_size
    return y_freq, x_freq


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


def freq_pink_filter_2d(x_freq, y_freq, factor=1, no_mean=False):
    f = freq_filter_2d(x_freq, y_freq)
    f = 1 / (1 + f)
    f = f ** factor
    f = np.where(f==1, 0, f) if no_mean else f
    return f


def freq_pink_filter_1d(x_freq, factor=1, no_mean=False):
    f = 1 / (1 + np.abs(x_freq))
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


def make_img_transition_x(img, depth, is_dx_pos=True):
    y_size, x_size = img.shape
    additional_img = gen_cloud(x_size + depth, y_size)   
    transition_kernel = spatial_smooth_filter(x_size, y_size, depth)     
    
    new_img = np.copy(img)
    if is_dx_pos:
        new_img[:, -depth:x_size] = img[:, -depth:x_size] * transition_kernel + \
                                additional_img[:, 0:depth] * (1 - transition_kernel)
        return new_img, additional_img[:, depth:]    
    else:
        transition_kernel = np.fliplr(transition_kernel)
        new_img[:, 0:depth] = img[:, 0:depth] * transition_kernel + \
                          additional_img[:, -depth:] * (1 - transition_kernel)  
        return new_img, additional_img[:, 0:-depth]    


def make_img_transition_y(img, depth, is_dy_pos=True):
    y_size, x_size = img.shape
    additional_img = gen_cloud(x_size, y_size + depth)   
    transition_kernel = spatial_smooth_filter(x_size, y_size, depth, horiz=False)
        
    new_img = np.copy(img)
    if is_dy_pos:
        new_img[-depth:x_size, :] = img[-depth:x_size, :] * transition_kernel + \
                                additional_img[0:depth, :] * (1 - transition_kernel)
        return new_img, additional_img[depth:, :]    
    else:
        transition_kernel = np.flipud(transition_kernel)
        new_img[0:depth, :] = img[0:depth, :] * transition_kernel + \
                          additional_img[-depth:, :] * (1 - transition_kernel)  
        return new_img, additional_img[0:-depth:1, :]
		
		
def make_img_transition_xy(img, depth, is_dx_pos=True, is_dy_pos=True):
    new_img, add_img = make_img_transition_x(img, depth, is_dx_pos=is_dx_pos)
    new_img = np.concatenate((new_img, add_img), axis=1)
    
    new_img, add_img = make_img_transition_y(new_img, depth, is_dy_pos=is_dy_pos)
    new_img = np.concatenate((new_img, add_img), axis=0)
    
    return new_img


def shift_img_x(img, width, dx, is_dx_pos=True): 
    # Wind blows in negative X dirrection
    if is_dx_pos:
        return img[:, dx:width + dx]
    else:
        return img[:, width - dx:-dx]


def shift_img_y(img, height, dy, is_dy_pos=True):
    # Wind blows in negative Y dirrection
    if is_dy_pos:
        return img[dy:height + dy, :]
    else:
        return img[height - dy:-dy, :]

    
def shift_img_xy(img, window_shape, dx, dy, is_dx_pos=True, is_dy_pos=True):
    height, width = window_shape
    
    shifted_img = shift_img_x(img, width, dx, is_dx_pos=is_dx_pos)
    shifted_img = shift_img_y(shifted_img, height, dy, is_dy_pos=is_dy_pos)
    
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
