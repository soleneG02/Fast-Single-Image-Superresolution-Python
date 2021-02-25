import numpy as np
def create_SH_y_matrix(imageLR, nr_up, nc_up, down_sampling_factor):
    """Allows to get upsampled image

    This function returns the up sampled image according to the model

    Args: 
        imageLR: The degraded image (blurred, but of original dimension because of bicubic interpolation)
        nr_up: The vertical dimension of the upsized image
        nc_up: The horizontal dimension of the downsized image
        down_sampling_factor: The down sampling factor. 

    Returns: 
        Sh_y: The upsampled image.
        ind1_SH_y: The indices where we lost the information compared to the interpolation.
        ind2_SH_y: The indices where we know the information.
       
    """
    SH_y = np.zeros((nr_up,nc_up))
    dr = down_sampling_factor
    dc = down_sampling_factor
    SH_y[::dr,::dc] = imageLR
    SH_y_temporary = np.ones((nr_up,nc_up))
    SH_y_temporary[::dr, ::dc] = imageLR
    ind1_SH_y = np.where((SH_y_temporary-SH_y)==1)
    ind2_SH_y = np.where((SH_y_temporary-SH_y)==0)
    return SH_y, ind1_SH_y, ind2_SH_y