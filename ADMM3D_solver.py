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

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def PsiT(P1,P2,P3):
    return  np.concatenate((P1[1,:,:],np.diff(P1,1,0),-P1[-1,:,:]),axis=0)+ \
                        np.concatenate((P2[:,0,:],np.diff(P2,0,1),-P2[:,-1,:]),axis=1) + np.concatenate((P3[:,:,0],np.diff(P3,0,2),-P3[:,:,-1]),axis=2)

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
    '''
    if np.shape(v)[0]!= 0: 
        mag = np.sqrt(np.square(np.concatenate((v,np.zeros(1,np.shape(v)[1],np.shape(v)[2])),axis=0)+ \
                    np.concatenate((h,np.zeros(np.shape(h)[0],1,np.shape(h)[2])),axis=1)+np.concatenate((d,np.zeros(np.shape(d)[0],np.shape(d)[1],1)),axis=2)
        )
        #magt = DiffuserCam_soft(mag,tau)
        #mmult = magt./mag
       #mmult(mag==0) = 0
    '''
    return varagout

def draw_figures(xk):
    #plt.imshow(image, cmap=solverSettings.cmap)


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
        return x[(p1+1):(-1-p1),(p2+1):(-1-p2),:]

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
    Dtb= pad3d(pad2d(np.expand_dims(b,axis=2),p1,p2),Nz)

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
    DtD = pad3d(pad2d(np.expand_dims(np.ones_like(b),axis=2),p1,p2),Nz)
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
        n = n+1
        Hvk = Hvkp 
        nukp = np.multiply(nu_mult,(mu1 *(xi/mu1+Hvk)+Dtb))
        wkp = np.maximum(rho/mu3+vk, 0)
        if np.char.lower(solverSettings.regularizer) =='tv':
            print(np.shape(Lvk1))
            print(np.shape(eta_1))
            uk1,uk2,uk3 = DiffuserCam_soft_3d(Lvk1+eta_1/mu2, Lvk2+eta_2/mu2, Lvk3+eta_3/mu2,solverSettings.tau/mu2)
            vkp_numerator = mu3*(wkp-rho/mu3) + mu2*PsiT(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2) + mu1*Hadj(nukp - xi/mu1)

        vkp = np.multiply(np.real(np.fft.ifftn(v_mult, np.fft.fftn(vkp_numerator))))

        Hvkp = Hfor(vkp)
        r_sv = Hvkp-nukp
        xi = xi + mu1*r_sv
        f.dual_resid_s[n] = mu1*np.linalg.norm(vec(Hvk - Hvkp))
        f.primal_resid_s[n] = np.linalg.norm(vec(r_sv))
        mu1, mu1_update = ADMM3D_update_param(mu1,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_s[n],f.dual_resid_s[n])
        f.data_fidelity[n] = .5*np.linalg.norm(crop3d(Hvkp)-b,'fro')^2
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
            f.dual_resid_u[n] = mu2*np.sqrt(np.linalg.norm(vec(Lvk1_ - Lvk1))^2 + np.linalg.norm(vec(Lvk2_ - Lvk2))^2 + np.linalg.norm(vec(Lvk3_ - Lvk3))^2)
            f.primal_resid_u[n] = np.sqrt(np.linalg.norm(vec(r_su_1))^2 + np.linalg.norm(vec(r_su_2))^2 + np.linalg.norm(vec(r_su_3))^2)
            f.regularizer_penalty[n] = solverSettings.tau*(np.sum(vec(np.abs(Lvk1))) + np.sum(vec(np.bs(Lvk2))) + sum(vec(np.abs(Lvk3))))
        

        f.objective[n] = f.data_fidelity[n] + f.regularizer_penalty[n]
    
        mu2, mu2_update = ADMM3D_update_param(mu2,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_u[n],f.dual_resid_u[n])

        r_sw = vkp-wkp
        rho = rho + mu3*r_sw
        f.dual_resid_w[n] = mu3*np.linalg.norm(vec(vk - vkp))
        f.primal_resid_w[n] = np.linalg.norm(vec(r_sw))
        mu3, mu3_update = ADMM3D_update_param(mu3,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_w[n],f.dual_resid_w[n])
        
        if (mu1_update or mu2_update or mu3_update):
            mu_update = 1
        else:
            mu_update = 0
        if mu_update:
            v_mult = 1./(mu1*HtH + mu2*PsiTPsi + mu3)
            nu_mult = 1./(DtD + mu1)
         
        vk = vkp

        if np.mod(n,solverSettings.save_every) == 0:
            print('saving state %i...\n',n)
            #out_file = save_state(solverSettings,n)
            #save(out_file,solverSettings.save_vars{:})
            print('done saving\n')
        
        if np.mod(n,solverSettings.print_interval) == 0:
            t_iter = toc()/solverSettings.print_interval
            '''
            fprintf('iter: %i \t t: %.2g \t cost: %.2g \t data_fidelity: %.2g \t norm: %.2g \t Primal v: %.2g \t Dual v: %.2g \t Primal u: %.2g \t Dual u: %.2g \t Primal w: %.2g \t Dual w: %.2g \t mu1: %.2g \t mu2: %.2g \t mu3: %.2g \n',...
                n,t_iter,f.objective(n),f.data_fidelity(n),f.regularizer_penalty(n),f.primal_resid_s(n), f.dual_resid_s(n),f.primal_resid_u(n), f.dual_resid_u(n),f.primal_resid_w(n), f.dual_resid_w(n),mu1,mu2,mu3)
                %disp([n,f.objective(n),f.data_fidelity(n),f.regularizer_penalty(n),f.primal_resid_s(n), f.dual_resid_s(n),f.primal_resid_u(n), f.dual_resid_u(n),f.primal_resid_w(n), f.dual_resid_w(n),mu1,mu2,mu3])
            '''
            tic()
        if np.mod(n,solverSettings.disp_figs) == 0 :
            draw_figures(vk,solverSettings)
    return vk, f