import DiffuserCam_settings
from DiffuserCam_settings import solverSettings 
from scipy.io import loadmat
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
    return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def PsiT(P1,P2,P3):
    return   np.concatenate((np.expand_dims(P1[0,:,:],axis = 0),np.diff(P1,1,axis = 0),np.expand_dims(-P1[-1,:,:], axis = 0)),axis=0)+ \
                        np.concatenate((np.expand_dims(P2[:,0,:], axis =1),np.diff(P2,1,axis = 1),np.expand_dims(-P2[:,-1,:],axis =1)),axis=1) + np.concatenate((np.expand_dims(P3[:,:,0],axis = 2),np.diff(P3,1,axis = 2),np.expand_dims(-P3[:,:,-1], axis =2)),axis=2)

def Psi(X):
    k1 = -np.diff(X,1,0)
    k2 = -np.diff(X,1,1)
    k3 = -np.diff(X,1,2)
    return k1,k2,k3

def generate_laplacian(lapl):
    lapl[0] = 6
    lapl[0,1,0] = -1
    lapl[1,0,0] = -1
    lapl[0,0,1] = -1
    lapl[0,-1,0] = -1
    lapl[-1,0,0] = -1
    lapl[0,0,-1] = -1
    PsiTpsi = np.abs(np.fft.fftn(lapl))
    return PsiTpsi

def ADMM3D_update_param(mu,resid_tol,mu_inc,mu_dec,r,s):
    if (r > resid_tol*s):
        mu_out = mu*mu_inc
        mu_update = 1
    elif (r*resid_tol < s):
        mu_out = mu/mu_dec
        mu_update = -1
    else:
        mu_out = mu
        mu_update = 0
    return mu_out,mu_update

def DiffuserCam_soft(x,tau):

    threshed = np.maximum(np.abs(x)-tau,0)
    threshed = np.multiply(threshed,np.sign(x))
    return threshed

def DiffuserCam_soft_3d(v,h,d,tau,varargin):
    
    if np.shape(v)[0]!= 0: 
        mag = np.sqrt(np.square(np.concatenate((v,np.zeros((1,np.shape(v)[1],np.shape(v)[2]))),axis=0)+ 
                    np.concatenate((h,np.zeros((np.shape(h)[0],1,np.shape(h)[2]))),axis=1)+np.concatenate((d,np.zeros((np.shape(d)[0],np.shape(d)[1],1))),axis=2)))

        magt = DiffuserCam_soft(mag,tau)
        mmult = np.divide(magt, mag)
        mmult[mag == 0] = 0

        varargout = [0] * 3
        varargout[0]= v * mmult[0:-1,:,:]
        varargout[1] = h * mmult[:, 0:-1, :]
        varargout[2] = d * mmult[:,:,0:-1]

        if varargin is not None:
            varargout[4] = DiffuserCam_soft(varargin[0], tau)
    else:
        varargout = []
        varargout.append(DiffuserCam_soft(varargin[0], tau))
    return varargout #should be python list of numpy arrays? hopefully

def draw_figures(xk):
    print('Plotting Now')
    #plt.imshow(image, cmap=solverSettings.cmap)
    fig, (ax1, ax2,ax3)  = plt.subplots(1,3)
    im1 = np.squeeze(np.sum(xk, axis = 2))
    ax1.imshow(im1,cmap = solverSettings.cmap)
    ax1.set_title('XY')

    im2 = np.squeeze(np.amax(xk, axis = 0))
    ax2.imshow(im2,cmap = solverSettings.cmap)
    ax2.set_title('XZ')

    im3 = np.squeeze(np.amax(xk, axis = 1))
    ax3.imshow(im3,cmap = solverSettings.cmap)
    ax3.set_title('XZ')
    plt.show()

    return None


def ADMM3D_solver(psf,b):
    mu1 = solverSettings.mu1
    mu2 = solverSettings.mu2
    mu3 = solverSettings.mu3
    [Ny, Nx, Nz] = np.shape(psf)
    print(Ny, Nx, Nz)

    p1 = np.floor(Ny/2)    
    p2 = np.floor(Nx/2)

    def pad2d(x, p1, p2):
        p1 = int(p1)
        p2 = int(p2)
        return np.pad(x,((p1,),(p2,),(0,)))

    def pad3d(x, Nz):
        h = Nz-1
        return np.pad(x, ((0,0),(0,0),(0,h)))

    def crop2d(x,p1,p2):
        p1 = int(p1)
        p2 = int(p2)
        x = np.expand_dims(x, axis =2)
        return x[p1: (-p1),p2:(-p2),:]

    def crop3d(x):
        return crop2d(x[:,:,0],p1,p2)

    psf = np.roll(np.flip(psf,2),int(np.ceil(Nz/2)+1),axis = 2)/ np.linalg.norm(psf)
    Hs = np.fft.fftn(np.fft.ifftshift(pad2d(psf,p1,p2)))
    Hs_conj = np.conj(Hs)

    def vec(X):
        return np.reshape(X,(np.prod(np.shape(X)),1))

    def Hfor(x):
        return np.real(np.fft.ifftn(np.multiply(Hs,np.fft.fftn(x))))

    def Hadj(x):
        return np.real(np.fft.ifftn(np.multiply(Hs_conj,np.fft.fftn(x))))

    HtH = np.abs(np.multiply(Hs,Hs_conj)) 
    
    vk = 0*np.real(Hs)
    xi = vk
    rho = vk
    b = np.expand_dims(b, axis =2)
    Dtb= pad3d(pad2d(b,p1,p2),Nz)

    if np.char.lower(solverSettings.regularizer) =='tv':
        PsiTPsi = generate_laplacian(vk)
        eta_1 = vk[0:-1,:,:]
        eta_2 = vk[:,0:-1,:]
        eta_3 = vk[:,:,0:-1]
        uk1,uk2,uk3 = Psi(vk)
        Lvk1 = uk1
        Lvk2 = uk2
        Lvk3 = uk3

    v_mult = 1./(mu1*HtH+mu2*PsiTPsi+mu3)
    DtD = pad3d(pad2d(np.ones_like(b),p1,p2),Nz)
    nu_mult = 1./(DtD+mu1)

    n = 0 
    class f:
        dual_resid_s = np.full((1,solverSettings.maxIter), np.nan)
        primal_resid_s = np.full((1,solverSettings.maxIter), np.nan)
        dual_resid_u = dual_resid_s
        primal_resid_u = dual_resid_u
        dual_resid_w = dual_resid_s
        primal_resid_w = dual_resid_s
        objective = primal_resid_u 
        data_fidelity = primal_resid_u
        regularizer_penalty = primal_resid_u
        
    tic()
    Hvkp = vk
    while (n<solverSettings.maxIter): 
        Hvk = Hvkp 
        nukp = np.multiply(nu_mult,(mu1 *(xi/mu1+Hvk)+Dtb))
        wkp = np.maximum(rho/mu3+vk, 0)
        if np.char.lower(solverSettings.regularizer) =='tv':
            uk1,uk2,uk3 = DiffuserCam_soft_3d(Lvk1+eta_1/mu2, Lvk2+eta_2/mu2, Lvk3+eta_3/mu2,solverSettings.tau/mu2,None)
            vkp_numerator = mu3*(wkp-rho/mu3) + mu2*PsiT(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2) + mu1*Hadj(nukp - xi/mu1)

        vkp = np.real(np.multiply(np.fft.ifftn(v_mult),np.fft.fftn(vkp_numerator)))

        Hvkp = Hfor(vkp)
        r_sv = Hvkp-nukp
        xi = xi + mu1*r_sv
        f.dual_resid_s[0,n] = mu1*np.linalg.norm(vec(Hvk - Hvkp))
        f.primal_resid_s[0,n] = np.linalg.norm(vec(r_sv))
        mu1, mu1_update = ADMM3D_update_param(mu1,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_s[0,n],f.dual_resid_s[0,n])
        f.data_fidelity[0,n] = .5*np.square(np.linalg.norm(crop3d(Hvkp)-b))
        if np.char.lower(solverSettings.regularizer) =='tv':
            Lvk1_ = Lvk1
            Lvk2_ = Lvk2
            Lvk3_ = Lvk3
            Lvk1, Lvk2, Lvk3 = Psi(vkp)
            r_su_1 = Lvk1 - uk1
            r_su_2 = Lvk2 - uk2
            r_su_3 = Lvk3 - uk3
            eta_1 = eta_1 + mu2*r_su_1
            eta_2 = eta_2 + mu2*r_su_2
            eta_3 = eta_3 + mu2*r_su_3
            f.dual_resid_u[0,n] = mu2*np.sqrt(np.square(np.linalg.norm(vec(Lvk1_ - Lvk1))) + np.square(np.linalg.norm(vec(Lvk2_ - Lvk2))) + np.square(np.linalg.norm(vec(Lvk3_ - Lvk3))))
            f.primal_resid_u[0,n] = np.sqrt(np.square(np.linalg.norm(vec(r_su_1))) + np.square(np.linalg.norm(vec(r_su_2))) + np.square(np.linalg.norm(vec(r_su_3))))
            f.regularizer_penalty[0,n] = solverSettings.tau*(np.sum(vec(np.abs(Lvk1))) + np.sum(vec(np.abs(Lvk2))) + sum(vec(np.abs(Lvk3))))
        

        f.objective[0,n] = f.data_fidelity[0,n] + f.regularizer_penalty[0,n]
    
        mu2, mu2_update = ADMM3D_update_param(mu2,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_u[0,n],f.dual_resid_u[0,n])

        r_sw = vkp-wkp
        rho = rho + mu3*r_sw
        f.dual_resid_w[0,n] = mu3*np.linalg.norm(vec(vk - vkp))
        f.primal_resid_w[0,n] = np.linalg.norm(vec(r_sw))
        mu3, mu3_update = ADMM3D_update_param(mu3,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_w[0,n],f.dual_resid_w[0,n])
        
        if (mu1_update or mu2_update or mu3_update):
            mu_update = 1
        else:
            mu_update = 0
        if mu_update:
            v_mult = 1./(mu1*HtH + mu2*PsiTPsi + mu3)
            nu_mult = 1./(DtD + mu1)
         
        vk = vkp
        '''
        if np.mod(n,solverSettings.save_every) == 0:
            print('saving state %i...\n',n)
            #out_file = save_state(solverSettings,n)
            #save(out_file,solverSettings.save_vars{:})
            print('done saving\n')
        '''
        print(n)
        if n!=0:
            if np.mod(n,solverSettings.print_interval) == 0:
                t_iter = toc()/solverSettings.print_interval
        
                print('iter: ', n)
                print('t: ',t_iter)

                tic()
        if np.mod(n,solverSettings.disp_figs) == 0 :
            draw_figures(vk)
        
        n = n+1
    return vk, f