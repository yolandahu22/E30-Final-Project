import DiffuserCam_settings
from DiffuserCam_settings import solverSettings 
from scipy.io import loadmat,savemat 
import numpy as np
from PIL import Image
from ADMM3D_solver import *
import os

def DiffuserCam_main():
    #Solve for image from DiffuserCam. First rev: 3D ADMM only. 
    xhat = 0
    f = 0
    if solverSettings.save_dir[-1]=='/':
        solverSettings.save_dir = solverSettings.save_dir[1:-2]
    solverSettings.save_dir = [solverSettings.save_dir,'/',solverSettings.dtstamp]

    if (DiffuserCam_settings.lateral_downsample < 1):
        print('lateral_downsample must be >= 1')
        pass
    if (DiffuserCam_settings.axial_downsample < 1 ):
        print('axial_downsample must be >= 1')
        pass
    
    psf = loadmat(DiffuserCam_settings.impulse_mat_file_name)
    psf = psf[DiffuserCam_settings.impulse_var_name]

    Nz_in=psf.shape[2]

    if (DiffuserCam_settings.end_z == 0 or DiffuserCam_settings.end_z > Nz_in):
        DiffuserCam_settings.end_z = Nz_in
    
    psf=np.array(psf)
    psf = psf[:, :, DiffuserCam_settings.start_z:DiffuserCam_settings.end_z]-DiffuserCam_settings.psf_bias
    for n in range(int(np.log2(DiffuserCam_settings.lateral_downsample))):
        psf = 0.25*(psf[::2,::2,:]+psf[::2,1::2,:]+psf[1::2,::2,:]+psf[1::2,1::2,:])

    for n in range(int(np.log2(DiffuserCam_settings.axial_downsample))):
        psf = 0.5*(psf[:,:,::2]+psf[:,:,1::2])
    
    Ny = np.shape(psf)[0]
    Nx = np.shape(psf)[1]

    #assume image is preprocesssed into monochromic
    imc = Image.open(DiffuserCam_settings.image_file)
    b = np.array(imc) - DiffuserCam_settings.image_bias
    b = b/np.amax(b[:])

    [xhat, f] = ADMM3D_solver(np.single(psf),np.single(b))
    if solverSettings.save_results: 
        print('saving final results. Please wait. \n')
        path = './DiffuserCamResults/'+solverSettings.dtstamp+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        outfile = solverSettings.dtstamp+'.mat'
        print(outfile)
        xhat_out = np.copy(xhat)
        mdic = {"xhat_out": xhat_out}
        savemat(outfile,mdic)

    return xhat,f

xhat,f=DiffuserCam_main()