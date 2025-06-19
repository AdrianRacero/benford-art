# imports
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import os
import pywt

# torch configuration
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)

# Constants
EPSILON = 1e-6

#########################
# FFT
#########################

def compute_fft2d(block):
    """
    Compute 2D Fourier Transform using FFT.
    
    Parameters:
    - block (torch.Tensor): input block
    
    Returns:
    - torch.Tensor: Fourier coefficients
    """
    return torch.fft.fft2(block)
    
#########################
# DCT
#########################

def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def dct(x):
    """
    Compute 1D Discrete Cosine Transform (DCT-II) using FFT.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V[:, 0] /= np.sqrt(N) * 2 # norm='ortho'
    V[:, 1:] /= np.sqrt(N / 2) * 2 # norm='ortho'

    V = 2 * V.view(*x_shape)

    return V

def compute_dct2d(block):
    """
    Compute 2D Discrete Cosine Transform (DCT-II).
    
    Parameters:
    - block (torch.Tensor): input block
    
    Returns:
    - torch.Tensor: DCT coefficients
    """
    X1 = dct(block)
    X2 = dct(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

#########################
# DST
#########################

def dst_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def dst(x):
    """
    Compute 1D Discrete Sine Transform (DST) using FFT.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    
    v = torch.cat([torch.zeros_like(x[:, :1]), x, -x.flip([1])[:, :-1]], dim=1)
    
    Vc = dst_fft_impl(v)
    
    k = torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (N + 1)
    W_r = torch.sin(k)
    W_i = torch.cos(k)
    
    V = Vc[:, 1:N+1, 0] * W_r + Vc[:, 1:N+1, 1] * W_i
    
    V = 2 * V.view(*x_shape)
    
    return V

def compute_dst2d(block):
    """
    Compute 2D Discrete Sine Transform (DST-II).
    
    Parameters:
    - block (torch.Tensor): input block
    
    Returns:
    - torch.Tensor: DST coefficients
    """
    X1 = dst(block)
    X2 = dst(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

#########################
# WT
#########################

def compute_wt2d(image):
    """
    Compute Wavelet Transform to an 2D image (WT).
    
    Parameters:
    - image (np.ndarray): 2D grayscale image
    
    Returns:
    - torch.Tensor: WT coefficients
    """
    wavelet = 'bior4.4'
    
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=7)
    
    detail_coeffs = coeffs[1:]

    detail_coeffs_torch = torch.cat([
        torch.tensor(c, dtype=torch.float, device=device).view(-1)  
        for level in detail_coeffs
        for c in level  # cH, cV, cD
    ])

    return detail_coeffs_torch

#########################
# LP
#########################

def compute_lp2d(image, levels=3):
    """
    Compute Laplacian Pyramid to an 2D image (LP).
    
    Parameters:
    - image (np.ndarray): 2D grayscale image
    
    Returns:
    - torch.Tensor: LP coefficients
    """

    image_float = image.astype(np.float32)
    gaussian_pyramid = [image_float.copy()]
    
    # gaussian pyramid
    for _ in range(levels):
        image_float = cv2.pyrDown(image_float)
        gaussian_pyramid.append(image_float)

    laplacian_pyramid = []
    
    # laplacian pyramid
    for i in range(levels):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        
        # resize to match the size of the first level for stacking
        laplacian_resized = cv2.resize(laplacian, (gaussian_pyramid[0].shape[1], gaussian_pyramid[0].shape[0]))
        laplacian_pyramid.append(laplacian_resized)

    # Convert to tensor
    lp_tensor = torch.from_numpy(np.stack(laplacian_pyramid, axis=0)).float().to(device)

    return lp_tensor

def first_significant_digit(values, base):
    """
    Computes the first significant digit (FD) of a given values in a specified base.
    
    Parameters:
    - values (torch.Tensor): coefficient values.
    - base (int): The numerical base.
    
    Returns:
    - torch.Tensor: The first significant digit of the input values in the given base.
    """
    values = torch.flatten(torch.abs(values))

    base = torch.tensor(base, dtype=torch.int32, device=device)
    
    fd = torch.zeros_like(values, dtype=torch.uint8, device=device)
    
    # Select non zero values
    mask = values != 0
    nonzero_values = values[mask]
    
    # Apply the formula only to non zero values
    exponent = torch.floor(torch.log(nonzero_values) / torch.log(base))
    fd_nonzero = torch.floor(nonzero_values / (base ** exponent)).to(torch.uint8)
    
    fd[mask] = fd_nonzero
    
    return fd

def get_pdf_est(digits, base):
    """
    Compute the probability density function of the first significant digits of DCT coefficients.
    
    Parameters:
    - digits (torch.Tensor): First significant digits of DCT coefficients.
    - base (int): Numerical base.
    
    Returns:
    - p_est (dict): Probability density function of the first significant digits of DCT coefficients.
    """
    
    p_est = {i: 0 for i in range(1, base)}

    K = digits.shape[0]

    # Count occurrences of each digit using torch's bincount (efficient counting)
    counts = torch.bincount(digits[digits > 0], minlength=base)[1:]

    # Update p_est with counts
    p_est.update({i: counts[i-1].item() for i in range(1, base)})

    # Compute total non-zero elements (avoid division by zero)
    for i in range(1,base):
        p_est[i] /= K
        if p_est[i] == 0:
            p_est[i] = EPSILON
    
    # Normalize the probabilities
    total = sum(p_est.values())
    for i in range(1, base):
        p_est[i] /= total

    return p_est

def benford(d, beta, gamma, delta, base):
    """Generalized Benford's Law function."""
    denom = np.clip(gamma + d**delta, EPSILON, None)  # Avoid division by zero
    return beta * (np.log(1 + 1 / denom) / np.log(base))

def residuals(params, d, p):
    """Compute residuals for curve fitting."""
    return benford(d, *params, len(d) + 1) - p

def jacobian(params, d, p):
    beta, gamma, delta = params
    base = len(d) + 1

    J_beta = (np.log(1 - 1 / (gamma - d**delta)) / np.log(base))
    J_gamma = beta / (np.log(base) * (d**delta - gamma) * (d**delta - gamma + 1))
    J_delta = (base * d**delta ** np.log(d)) / (np.log(base) * (d**delta - gamma) * (d**delta - gamma + 1))

    return np.vstack((J_beta, J_gamma, J_delta)).T

def get_pdf_fit(p_est):
    """
    Fit Benford's Law to the estimated probability density function.
    
    Parameters:
    - p_est (dict): Estimated probability density function (keys: digits, values: probabilities).
    
    Returns:
    - p_fit (dict): Fitted probability density function.
    """
    
    p_fit = {}
    
    d_values = np.array(list(p_est.keys()))
    p_values = np.array(list(p_est.values()))

    base = len(d_values) + 1

    initial_guess = [1, 1, 1]

    # popt = opt.least_squares(residuals, initial_guess, jac=jacobian, args=(d_values, p_values))
    popt = opt.least_squares(residuals, initial_guess, args=(d_values, p_values))

    for i in d_values:
        p_fit[i] = benford(i, *popt.x, base)
        if p_fit[i] == 0:
            p_fit[i] = EPSILON  # Avoid division by zero

    # Normalize PDF
    p_fit_sum = sum(p_fit.values())
    for i in d_values:
        p_fit[i] /= p_fit_sum

    return p_fit

def js_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    
    Parameters:
    - p (dict): First probability distribution.
    - q (dict): Second probability distribution.
    
    Returns:
    - float: Jensen-Shannon divergence between p and q.
    """
    
    return kl_divergence(p, q) + kl_divergence(q, p)

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions.
    
    Parameters:
    - p (dict): First probability distribution.
    - q (dict): Second probability distribution.
    
    Returns:
    - float: Kullback-Leibler divergence between p and q.
    """
    
    kl = 0
    
    for key in p.keys():
        kl += p[key] * np.log(p[key] / q[key])
    
    return kl

def r_divergence(p, q, alpha):
    """
    Compute the Renyi divergence between two probability distributions.
    
    Parameters:
    - p (dict): First probability distribution.
    - q (dict): Second probability distribution.
    
    Returns:
    - float: Renyi divergence between p and q.
    """
    
    r = 1 / (1 - alpha) * (np.log(s_function(p, q, alpha)) + np.log(s_function(q, p, alpha)))
    
    return r

def t_divergence(p, q, alpha):
    """
    Compute the Tsallis divergence between two probability distributions.
    
    Parameters:
    - p (dict): First probability distribution.
    - q (dict): Second probability distribution.
    
    Returns:
    - float: Tsallis divergence between p and q.
    """
    
    t = 1 / (1 - alpha) * (2 - s_function(p, q, alpha) - s_function(q, p, alpha))
    
    return t

def s_function(q, p, alpha):
    """
    Compute weighted sum that combines two probability distributions

    Parameters:
    - q (dict): First probability distribution.
    - p (dict): Second probability distribution.
    - alpha (float): Weighting factor.

    Returns:
    - float: Weighted sum of the two probability distributions.
    """

    s = 0

    for key in q.keys():
        s += (q[key] ** alpha) / (p[key] ** (alpha - 1))

    return s

def get_feature_vector(image_np, bases):
    """
    Compute feature vectors of an image
    
    Parameters:
    - image (torch.Tensor): input image
    - bases (list): list of numerical bases
    
    Returns:
    - torch.Tensor: feature vectors
    """

    image = torch.tensor(image_np, dtype=torch.float, device=device)
    
    fft = compute_fft2d(image)
    dst = compute_dst2d(image)
    dct = compute_dct2d(image)
    wt = compute_wt2d(image_np)
    lp = compute_lp2d(image_np)

    feature_vectors = torch.zeros((len(bases), 15), dtype=torch.float32, device=device)

    for b, base in enumerate(bases):
        fd_fft = first_significant_digit(fft, base)
        fd_dst = first_significant_digit(dst, base)
        fd_dct = first_significant_digit(dct, base)
        fd_wt = first_significant_digit(wt, base)
        fd_lp = first_significant_digit(lp, base)
        
        p_est_fft = get_pdf_est(fd_fft, base)
        p_fit_fft = get_pdf_fit(p_est_fft)

        p_est_dst = get_pdf_est(fd_dst, base)
        p_fit_dst = get_pdf_fit(p_est_dst)

        p_est_dct = get_pdf_est(fd_dct, base)
        p_fit_dct = get_pdf_fit(p_est_dct)

        p_est_wt = get_pdf_est(fd_wt, base)
        p_fit_wt = get_pdf_fit(p_est_wt)

        p_est_lp = get_pdf_est(fd_lp, base)
        p_fit_lp = get_pdf_fit(p_est_lp)
        
        js_fft = js_divergence(p_est_fft, p_fit_fft)
        r_fft = r_divergence(p_est_fft, p_fit_fft, 0.5)
        t_fft = t_divergence(p_est_fft, p_fit_fft, 0.5)
        js_dst = js_divergence(p_est_dst, p_fit_dst)
        r_dst = r_divergence(p_est_dst, p_fit_dst, 0.5)
        t_dst = t_divergence(p_est_dst, p_fit_dst, 0.5)
        js_dct = js_divergence(p_est_dct, p_fit_dct)
        r_dct = r_divergence(p_est_dct, p_fit_dct, 0.5)
        t_dct = t_divergence(p_est_dct, p_fit_dct, 0.5)
        js_wt = js_divergence(p_est_wt, p_fit_wt)
        r_wt = r_divergence(p_est_wt, p_fit_wt, 0.5)
        t_wt = t_divergence(p_est_wt, p_fit_wt, 0.5)
        js_lp = js_divergence(p_est_lp, p_fit_lp)
        r_lp = r_divergence(p_est_lp, p_fit_lp, 0.5)
        t_lp = t_divergence(p_est_lp, p_fit_lp, 0.5)
        feature_vectors[b] = torch.tensor([js_fft,r_fft,t_fft,js_dst,r_dst,t_dst,js_dct,r_dct,t_dct,js_wt,r_wt,t_wt,js_lp,r_lp,t_lp], device=device)
    
    return feature_vectors 

def process_and_save_images(folder_path, n_images, label, bases, filename):
    """
    Process and save feature vectors of a set of images to a CSV file one by one.

    Parameters:
    - folder_path (str): path to the folder containing the images.
    - n_images (int): number of images to process.
    - label (int): label of the images (1 if AI generated, 0 if not).
    - bases (list): list of numerical bases.
    - filename (str): name of the CSV file.
    """
    # Get the image filenames
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    n_images = min(n_images, len(image_files))

    if n_images == 0:
        print("No images found in the specified folder.")
        return

    for i in range(n_images):
        image_path = os.path.join(folder_path, image_files[i])
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image_np is None:
            print(f"Error loading image: {image_path}")
            continue

        try:
            # Extract features
            features = get_feature_vector(image_np, bases)
        except Exception as e:
            print(f"Error processing image: {image_path}, Error: {e}")
            continue

        # Convert to NumPy and flatten
        flattened_features = features.cpu().numpy().flatten()
        row = np.append(flattened_features, label)

        # Convert to DataFrame and save immediately
        df = pd.DataFrame([row])
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    print(f"{n_images} images have been processed and saved to {filename}")


def generate_csv_header(transformations, divergences, bases, filename):
    """
    Generate and save CSV header based on provided lists.
    """
    
    header = []

    for b in bases:
        for t in transformations:
            for d in divergences:
                    header.append(f"{b}_{t}_{d}")

    header.append("label")

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)

        if list(df_existing.columns) != header:
            df_existing.columns = header
            df_existing.to_csv(filename, index=False, header=True)  # Sobrescribir con nuevo encabezado
    else:
        df = pd.DataFrame(columns=header)
        df.to_csv(filename, index=False, header=True)
        
    return header
    