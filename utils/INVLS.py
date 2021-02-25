import numpy as np

def BlockMM(nr,nc,d,m,x1):
    """Function used to compute the analytical solution.

    This functions takes an input matrix and reduces it's dimension to observation matrix doing some reshaping and sums.

    Args:
        nr: the vertical size of the observation
        nc: the horizontal size of the observation
        d: the scale factor
        m: No. of the pixels of the observation m = nr*nc 
        x1: A matrix

    Returns:
        Modified matrix
    """

    nr_x1,nc_x1 = x1.shape
    result=np.empty((int(nc_x1/nc)*m,int(nr_x1/nr)*1),dtype=np.ndarray)
    for i in range(0,nr_x1,nr):
        for j in range(0, nc_x1,nc):
            result[int((i/nr)*m):int((i/nr)*m+m),int(j/nc)]=x1[i:i+nr, j:j+nc].reshape((m,),order="F")    
    result=result.reshape((m,d),order="F")
    result=np.sum(result, axis=1)
    return result.reshape(nr,nc,order="F")

def INVLS(FB,FBC,F2B,FR,mu,d,nr,nc,m,regularization):
    """Function to get the SR image.

    This functions allows one to get the super resolution image with the FSR algorithm.
    It gives the analytical solution as below :
        x = (B^H S^H SH + mu I )^(-1) R
        
    Args:
        FB: Fourier transform of the blurring kernel B
        FBC: conj(FB)
        F2B: abs(FB)**2
        FR: Fourier transform of R
        d: scale factor d = dr*dc
        nr,nc: size of the observation
        m: No. of the pixels of the observation m = nr*nc 
        regularization : Regularization used in the model 

    Returns:
        Xest->Analytical solution
        FX->Fourier transform of the analytical solution
    """ 
    x1=np.divide(FB*FR,regularization)
    FBR=BlockMM(nr,nc,d,m,x1)
    invW=BlockMM(nr,nc,d,m,np.divide(F2B,regularization))
    invWBR=np.divide(FBR,(invW+mu*d))

    nr_FBC, nc_FBC=FBC.shape

    FBCinvWBR=np.zeros((nr_FBC,nc_FBC),dtype=np.complex64)
    for i in range(0,nr_FBC,nr):
        for j in range(0, nc_FBC,nc):
            FBCinvWBR[i:i+nr,j:j+nc]=FBC[i:i+nr,j:j+nc]*invWBR

    FX=np.divide(FR-FBCinvWBR,regularization)/mu
    Xest=np.real(np.fft.ifft2(FX))

    return Xest,FX
