import numpy as np

def create_filtering_matrix(nr_up, nc_up, c):
    """ This functions allows to get the regularization

    This function returns the regularization matrix used for the problem with TV regularization.

    Args: 
        nr_up: The vertcial dimension of the upsampled image
        nc_up: The horizontal dimension of the upsampled 
        c: a small number
    
    Returns:
        Dh_F: Horizontal Derivation Matrix in fourier
        Dh_Fconj: Conjugate of Dh_F
        Dh_Fcarre: Square (term-wise) of Dh_F
        Dv_F: Vertical Derivation matrix in fourier
        Dv_Fconj: Conjuguate of Dv_F
        Dv_Fcarre: Square (term-wise) of Dv_F
        Dh_Dv_regularisation: The regularisation matrix
    """
    #define the difference operator kernel
    Dh = np.zeros((nr_up,nc_up))
    Dh[0,0] = 1
    Dh[0,1] = -1

    Dv = np.zeros((nr_up,nc_up))
    Dv[0,0] = 1
    Dv[1,0] = -1

    # compute FFTs for filtering
    Dh_F = np.fft.fft2(Dh)
    Dh_Fconj = np.conj(Dh_F)
    Dh_Fcarre = np.power(np.absolute(Dh_F), 2)
    Dv_F = np.fft.fft2(Dv)
    Dv_Fconj = np.conj(Dv_F);
    Dv_Fcarre = np.power(np.absolute(Dv_F), 2)
    Dh_Dv_regularisation = Dh_Fcarre + Dv_Fcarre + c # regularisation
    return Dh_F, Dh_Fconj, Dh_Fcarre, Dv_F, Dv_Fconj, Dv_Fcarre, Dh_Dv_regularisation