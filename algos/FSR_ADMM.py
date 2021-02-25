import numpy as np
import utils.INVLS as utils_INVLS

def FSR_ADMM(imageHR, imageLR_resized, imageLR, tolA, init_args, H_args, Dh_Dv_args, SH_y_args, FSR_args): 
    """Implements the ADMM method with FSR.

    Args:
        imageHR: the original image
        imageLR_resized: the observed image, interpolated to be of the same dimension
        imageLR: the observed image (blurred and downsized)
        tolA: convergence criterion
        init_args: constants used by the algorithms, see detail in code
        H_args: the blurring matrix, it's conjugate and it's square
        Dh_Dv_args: derivation matrix (in both axis h: horizontal and v: : vertical), conjugate and square
        SH_y_args: The upsampled image.
        FSR_args: Arguments for the FSR ADMM, see init_params docstring.

    Returns: 
        X: reconstructed image
        ISNR : array of errors computed with the ISNR metric (improved signal-to-noise ratio), for each value in muSet
        RMSE : array of errors computed with the RMSE metric (root mean square error), for each value in muSet
        PSNR : array of errors computed with the PSNR metric (peak signal-tonoise ratio), for each value in muSet
        Iter : array of the number of iterations before convergence for each value in muSet

    """

    [muSet, tau, dr, dc, N, maxiter, objective] = init_args # constants
    [H_F, H_Fconj, H_Fcarre] = H_args # blurring matrix
    [Dh_F, Dh_Fconj, _, Dv_F, Dv_Fconj, _] = Dh_Dv_args # derivation matrix
    [SH_y, _, _] = SH_y_args # SH_y matrix and indexes
    [d, nr, nc, m, Dh_Dv_regularisation] = FSR_args # args for FSR-ADMM

    nt = len(muSet)
    times = np.zeros((1,maxiter))
    distance = np.zeros((2,maxiter))
    distance_max = np.zeros((1,maxiter))
    criterion = np.zeros((1,maxiter))
    mses = np.zeros((1,maxiter))
    gamSet = np.divide(tau, muSet)
    ISNR = []
    RMSE = []
    PSNR = []
    Iter = []

    for t in range(0,nt):
        X = np.array(imageLR_resized)
        mu = muSet[t]
        U1 = X
        U2 = X
        D1 = 0*X
        D2 = D1
        
        for i in range(0, maxiter):
        #%%%%%%%%%%%%%%%%%%%%%%%%%% update X %%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% argmin_x  .5*||y-SHx||^2 + 
        #%           mu/2*||DhX - U1 + D1||^2 + 
        #%           mu/2*||DvX - U2 + D2||^2
            V1 = U1-D1  #rhoh
            V2 = U2-D2  #rhov
            # F(Dh) = FDHC
            # F(Dv) = FDVC
            FV1 = mu*Dh_Fconj*np.fft.fft2(V1) # mu * F(Dh) * F(rhoh)
            FV2 = mu*Dv_Fconj*np.fft.fft2(V2) #mu * F(Dv) * F(rhov)
            # F_HT_ST_y = F H^H S^H y
            F_HT_ST_y = H_Fconj*np.fft.fft2(SH_y)
            FR = F_HT_ST_y + FV1 + FV2
            # FX = x_f
            # X = x_k+1
            [X,FX] = utils_INVLS.INVLS(H_F,H_Fconj,H_Fcarre,FR,mu,d,nr,nc,m,Dh_Dv_regularisation)
        #%%%%%%%%%%%%%%%%%%%%%%%%%% update U %%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% argmin_u tau*||sqrt(U1^2 + U2^2)||_1 + 
        #%          mu/2*||DhX - U1 + D1||^2 +
        #%          mu/2*||DvX - U2 + D2||^2   
            DhX = np.fft.ifft2(FX*Dh_F) # F-1(Dh * x_k+1)
            DvX = np.fft.ifft2(FX*Dv_F) # Dv x_k+1
            NU1 = DhX + D1  #vecteur nu, 1er coeff 
            # D1 = dh
            NU2 = DvX + D2 #vecteur nu, 2eme coeff
            # D2 = dv
            NU = np.sqrt(np.power(NU1, 2)+np.power(NU2, 2))  #norme de nu
            a = np.maximum(0, NU-gamSet[t]) 
            # gamset = tau/mu
            # A = NU - gamset[t] => NU = A + gamSet[t]
            a = np.divide(a, (a + gamSet[t])) # A = np.divide(A, NU)
            U1 = a*NU1 #uk+1
            U2 = a*NU2  #uk+1
        #%%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #A = [Dh, Dv]
            # D1 = Dhk
            # D2 = Dvk
            D1 = D1 + (DhX-U1)  
            D2 = D2 + (DvX-U2)
        
            BX = np.fft.ifft2(H_F*FX)
            resid =  imageLR - BX[::dr,::dc]
            TVpenalty = np.sum(np.sum(np.sqrt(np.power(np.absolute(np.fft.ifft2(Dh_F*FX)), 2)+np.power(np.absolute(np.fft.ifft2(Dv_F*FX)), 2))));
            objective[t,i+1] = 0.5*np.real(np.dot(resid.flatten('F').T,resid.flatten('F'))) + tau*TVpenalty
            distance[0,i] = np.linalg.norm((DhX.flatten('F')-U1.flatten('F')),2)**2; # **2 new 
            distance[1,i] = np.linalg.norm((DvX.flatten('F')-U2.flatten('F')),2)**2; # **2 new
            distance_max[:,i]=distance[0,i]+distance[1,i];

            err = X-imageHR;
            mses[:,i] = err.flatten('F').T@err.flatten('F')/N
            
            criterion[:,i] = np.absolute(objective[t,i+1]-objective[t,i])/objective[t,i]
            if criterion[:,i] < tolA:
                break
        ISNR.append(10*np.log10(np.power(np.linalg.norm(imageHR-imageLR_resized, ord='fro'), 2))/np.power(np.linalg.norm(imageHR-X, ord='fro'), 2))
        RMSE_value = np.sqrt(np.power(np.linalg.norm(imageHR-X, ord='fro'), 2))
        RMSE.append(RMSE_value)
        PSNR.append(20*np.real(np.log10(np.amax(np.maximum(imageHR, X))/RMSE_value)))
        Iter.append(i)
        print('FSR ADMM: mu = ', muSet[t], ', Iter = ', Iter[-1], ', RMSE = ', RMSE[-1], ', PSNR = ', PSNR[-1], ', ISNR = ', ISNR[-1])
    return X, ISNR, RMSE, PSNR, Iter