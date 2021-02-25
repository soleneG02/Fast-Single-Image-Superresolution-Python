import numpy as np
from PIL import Image
from HXconv import HXconv
import f_special as f_special

def create_imageLR(imageHR, BSNRdb, down_sampling_factor):
    """
    Args:
        imageHR: The original HR
        BSNRdb:The signal to noise ratio in decibel
        down_sampling_factor: The down sampling factor

    Returns:
        imageLR_resized: The interpolated (bicubic interpolation) imageLR
        imageLR: The downsampled and blurred image
        nr_up: The upsampling vertical dimension
        nc_up: The upsampling horizontal dimension
        sigma_carre: Squared value of the variance of the noise
        H_args: The H (blurring) matrix, it's conjugate and term-wise square
        FSR_args: The following information as a list
            d:The down-sampling factor
            nr: the vertical dimension of LR image
            nc: the horizontal dimension of HR image
            m: the size of the LR image
            N: the size of the HR image. 
    """
    ### Blurring operation => H
    H = f_special.fspecial_python(shape=(9, 9), sigma=3)

    H_F, H_Fconj, H_Fcarre, image_withH = HXconv(imageHR,H)

    N_im, M_im = imageHR.shape
    N = N_im * M_im

    Psig = (np.linalg.norm(image_withH, ord='fro')**2)/N
    BSNRdb = BSNRdb
    sigma = np.linalg.norm(image_withH-np.mean(np.mean(image_withH)),ord='fro')/np.sqrt(N*10**(BSNRdb/10))
    
    # White gaussian noise
    n = sigma*np.random.randn(N_im, M_im)
    imageLR = image_withH + n # size = (512, 512)

    ## Down-sampling operation => S
    sigma_carre = sigma**2
    dr = down_sampling_factor # number of discarded rows
    dc = down_sampling_factor # number of discarded columns
    
    imageLR =  imageLR[::dr, ::dc] # size = (128, 128)
    nr,nc = imageLR.shape
    nr_up = nr*down_sampling_factor
    nc_up = nc*down_sampling_factor

    d = dr*dc # integer factor d of the down-sampling
    m = nr*nc
    
    imageLR_resized = Image.fromarray(imageLR).resize((nr_up, nc_up), Image.BICUBIC)
    H_args = [H_F, H_Fconj, H_Fcarre]
    FSR_args = [d, nr, nc, m, N]
    return imageLR_resized, imageLR, nr_up, nc_up, sigma_carre, H_args, FSR_args