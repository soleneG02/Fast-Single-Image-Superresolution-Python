import numpy as np

def direct_ADMM(imageHR, imageLR_resized, imageLR, tolA, init_args, H_args, Dh_Dv_args, SH_y_args):
    """Implements the ADMM method to find the SR image.

    Args:
        imageHR: the original image
        imageLR_resized: the observed image, interpolated to be of the same dimension
        imageLR: the observed image (blurred and downsized)
        tolA: convergence criterion
        init_args: constants used by the algorithms, see detail in code
        H_args: The blurring matrix, it's conjugate and it's square
        Dh_Dv_args: Derivation matrix (in both axis h: horizontal and v: : vertical), conjugate and square
        SH_y_args: The upsampled image.
        
    Returns:
        X: reconstructed image
        ISNR : array of errors computed with the ISNR metric (improved signal-to-noise ratio), for each value in muSet
        RMSE : array of errors computed with the RMSE metric (root mean square error), for each value in muSet
        PSNR : array of errors computed with the PSNR metric (peak signal-tonoise ratio), for each value in muSet
        Iter : array of the number of iterations before convergence for each value in muSet
    
    """
    #muSet : values of mu
    [muSet, tau, dr, dc, N, maxiter, objective] = init_args # constants
    [H_F, H_Fconj, H_Fcarre] = H_args # blurring matrix
    [Dh_F, Dh_Fconj, Dh_Fcarre, Dv_F, Dv_Fconj, Dv_Fcarre] = Dh_Dv_args # derivation matrix
    [SH_y, ind1_SH_y, ind2_SH_y] = SH_y_args # SH_y matrix and indexes

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

    for t in range(0, nt):
        mu = muSet[t]
        X = np.array(imageLR_resized)
        U1 = X
        U2 = X
        U3 = X
        D1 = 0*X
        D2 = D1
        D3 = D1
        
        I_DH = np.divide(Dh_Fconj, (Dh_Fcarre + Dv_Fcarre + H_Fcarre))
        I_DV = np.divide(Dv_Fconj, (Dh_Fcarre + Dv_Fcarre + H_Fcarre))
        I_BB = np.divide(H_Fconj, (Dh_Fcarre + Dv_Fcarre + H_Fcarre))

        for i in range(0, maxiter):
        # %%%%%%%%%%%%%%%%%%%%%%%%%% update X %%%%%%%%%%%%%%%%%%%%%%%%%%%
        # argmin_x  mu/2*||HX  - U1 + D1||^2 + 
        #           mu/2*||DhX - U2 + D2||^2 + 
        #           mu/2*||DvX - U3 + D3||^2
            V1 = U1-D1
            V2 = U2-D2
            V3 = U3-D3
            FX = I_BB*np.fft.fft2(V1) + I_DH*np.fft.fft2(V2) + I_DV*np.fft.fft2(V3)
            X = np.fft.ifft2(FX)

        #%%%%%%%%%%%%%%%%%%%%%%%%%% update U %%%%%%%%%%%%%%%%%%%%%%%%%%%
        #     argmin_u1 0.5||Su1 - y||_1 + mu/2*||HX - U1 + D1||^2 
            HX = np.fft.ifft2(H_F*FX)
            rr = mu*(HX + D1)
            temp1 = np.divide(rr[ind1_SH_y], mu)
            temp2 = np.divide((rr[ind2_SH_y] + SH_y[ind2_SH_y]),(1+mu))
            U1[ind1_SH_y] = np.real(temp1)
            U1[ind2_SH_y] = np.real(temp2)
        #   argmin_u(2,3) tau*||sqrt(U2^2 + U3^2)||_1 + 
        #            mu/2*||DhX - U2 + D2||^2 +
        #            mu/2*||DvX - U3 + D3||^2
            DhX = np.fft.ifft2(FX*Dh_F)
            DvX = np.fft.ifft2(FX*Dv_F)
            NU1 = DhX + D2
            NU2 = DvX + D3
            NU = np.sqrt(NU1**2+NU2**2)
            A = np.maximum(0, NU-gamSet[t])
            A = np.divide(A,(A + gamSet[t]))
            U2 = A*NU1
            U3 = A*NU2

        #%%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            D1 = D1 + (HX-U1) # DIFF : do not exist in FSR
            D2 = D2 + (DhX-U2)
            D3 = D3 + (DvX-U3) 
        
            resid =  imageLR - HX[::dr,::dc]
            TVpenalty = np.sum(np.sum(np.sqrt(np.power(np.absolute(np.fft.ifft2(Dh_F*FX)), 2)+np.power(np.absolute(np.fft.ifft2(Dv_F*FX)), 2))))
            objective[t,i+1] = 0.5*(np.real(np.dot(resid.flatten('F').T, resid.flatten('F')))) + tau*TVpenalty
            distance[0,i] = np.linalg.norm(DhX.flatten('F')-U1.flatten('F'),2)
            distance[1,i] = np.linalg.norm(DvX.flatten('F')-U2.flatten('F'),2)

            distance_max[:, i] = distance[0,i] + distance[1,i]
            err = X-imageHR
            mses[:, i] =  np.real(np.dot(err.flatten('F').T, err.flatten('F')))/N
            
            criterion[:, i] = np.absolute(objective[t,i+1]-objective[t,i])/objective[t,i]

            if criterion[:, i] < tolA:
                break

        ISNR.append(10*np.log10(np.power(np.linalg.norm(imageHR-imageLR_resized, ord='fro'), 2))/np.power(np.linalg.norm(imageHR-X, ord='fro'), 2))
        RMSE_value = np.sqrt(np.power(np.linalg.norm(imageHR-X, ord='fro'), 2))
        RMSE.append(RMSE_value)
        PSNR.append(20*np.real(np.log10(np.amax(np.maximum(imageHR, X))/RMSE_value)))
        Iter.append(i)
        print('direct ADMM: mu = ', muSet[t], ', Iter = ', Iter[-1], ', RMSE = ', RMSE[-1], ', PSNR = ', PSNR[-1], ', ISNR = ', ISNR[-1])
    return X, ISNR, RMSE, PSNR, Iter