import numpy as np
import math

def HXconv(X, H):
    """Add gaussian blur H to X

    This functions add the gaussian blur H to the image X.
    Uses a fourier transform to perform an Hadammard product.

    Args:
        X: Original Image
        H: The gaussian blur kernel

    Returns:
        H_F: The gaussian blur kernel in the fourier space
        H_Fconf: The conjugate of the gaussian blur kernel in the fourier space
        H_Fcarre: absolute value (term wise) of the gaussian blur kernel in the fourier space
        image_floutee : The blurred image

    """
    m, n = X.shape
    m0, n0 = H.shape
    Hpad = np.pad(H, (math.floor((m-m0+1)/2), math.floor((n-n0+1)/2)), 'constant')
    Hpad = np.pad(H, (round((m-m0+1)/2), round((n-n0-1)/2)), 'constant') 
    Hpad=np.fft.fftshift(Hpad)
    H_F = np.fft.fft2(Hpad)
    H_Fconj = np.conj(H_F)
    H_Fcarre = np.power(np.absolute(H_F), 2)
    image_floutee = np.real(np.fft.ifft2(H_F * np.fft.fft2(X)))
    return H_F, H_Fconj, H_Fcarre, image_floutee