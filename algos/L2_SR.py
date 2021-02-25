import numpy as np
from PIL import Image
import utils.HXconv as utils_HXconv
import utils.f_special as utils_fspecial
import utils.INVLS as utils_INVLS

def L2_SR(imageHR):
    """Performs the FSR algorithm on an image

    This algorithms performs the Fast Super-Resolution algorithm on an HR image.
    It will degrade it according to the model and then reconstruct it. 

    Args:
        imageHR : The original image

    Returns:
        Xest_analytic: estimated original image from the algorithm.
        yinp: Blurred image, interpolated from the low resolution image.
    """
    #Building blurred image
    H = utils_fspecial.fspecial_python(shape=(9, 9), sigma=3)
    H_F, H_Fconj, H_Fcarre, image_H = utils_HXconv.HXconv(imageHR, H)

    #Building gaussian noise
    N = imageHR.shape[0] * imageHR.shape[1]
    BSNRdb = 40
    sigma = np.linalg.norm(image_H - np.mean(np.mean(image_H)), ord='fro') / np.sqrt(N * 10 ** (BSNRdb / 10))
    n = sigma * np.random.randn(imageHR.shape[0], imageHR.shape[1])
    #Adding gaussian noise
    imageLR = image_H + n

    #Building Low_resolution image
    sigma_carre = sigma ** 2
    d = 2
    dr = d
    dc = d
    imageLR = imageLR[::dr, ::dc]

    #Building SR image based on Bicubic interpolation
    imageLR_resized = Image.fromarray(imageLR).resize((imageLR.shape[0] * dr, imageLR.shape[1] * dc), Image.BICUBIC)

    #Building Fr matrix
    taup = 2 * 10 ** (-3)
    tau = taup * sigma_carre
    (nr, nc) = imageLR.shape
    nr_up = nr * d
    nc_up = nc * d
    Nb = dr * dc
    m = nr * nc

    xp = imageLR_resized

    SH_y = np.zeros((nr_up, nc_up))
    SH_y[::d, ::d] = imageLR

    FR = H_Fconj * np.fft.fft2(SH_y) + np.fft.fft2(2 * tau * xp)

    #Finding estimated SR image from FSR algorithm.
    Xest_analytic, _ = utils_INVLS.INVLS(H_F, H_Fconj, H_Fcarre, FR, 2 * tau, Nb, nr, nc, m, 1)
    
    return Xest_analytic, imageLR_resized , imageLR
