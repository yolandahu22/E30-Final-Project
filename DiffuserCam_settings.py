
import numpy as np 
from datetime import datetime

#MAT File with PSF data
impulse_mat_file_name = './3d_example_data/example_psfs.mat'

#Variable name in mat file
impulse_var_name = 'psf'

#Measurement
image_file = './3d_example_data/example_raw.png'
color_to_process = 'mono'  #'red','green','blue', or 'mono'. If raw file is mono, this is ignored
image_bias = 100   #If camera has bias, subtract from measurement file. 
psf_bias = 102   #if PSF needs sensor bias removed, put that here.
lateral_downsample = 1  #factor to downsample impulse stack laterally. Must be multiple of 2 and >= 1.
axial_downsample = 1  #Axial averageing of impulse stack. Must be multiple of 2 and >= 1.
 
#Allow user to use subset of Z. This is computed BEFORE downsampling by a factor of AXIAL_DOWNSAMPLE
start_z = 0  #First plane to reconstruct. 1 indexed, as is tradition.
end_z = 0   #Last plane to reconstruct. If set to 0, use last plane in file.
 
 
# Populate solver options
 
# Solver parameters
class solverSettings: 
    tau = .000600    #sparsity parameter for TV
    tau_n = .0400     #sparsity param for native sparsity
    mu1 = 1    #Initialize ADMM tuning params to 1. These will be updated automatically by the autotune. 
    mu2 = 1    # To use fixed parameters, you'll have to hand tune. We recommend using autotune.
    mu3 = 1
    
    # if set to 1, auto-find mu1, mu2, mu3 every step. If set to 0, use user defined values. If set to N>1, tune for N steps then stop.
    autotune = 1    # default: 1
    mu_inc = 1.1 
    mu_dec = 1.1  #Inrement and decrement values for mu during autotune. Turn to 1 to have no tuning. The algorithm is very insensitive to these parameters.
    resid_tol = 1.5   #Primal/dual gap tolerance. Lower means more frequent tuning
    maxIter = 300 #Maximum iteration count  Default: 200
    regularizer = 'tv'   #'TV' for 3D TV, 'native' for native. Default: TV
    cmap = 'gray'


    #Figures and display
    disp_percentile = 99.9 #Percentile of max to set image scaling
    save_every = 0   #Save image stack as .mat every N iterations. Use 0 to never save (except for at the end);


    #Folder for saving state. If it doesn't exist, create it. 
    save_dir = '../DiffuserCamResults/'

    def disp_crop(x):
        return x[np.floor(np.size(x)[0]/4):np.floor(np.size(x)[0]*3/4),np.floor(np.size(x)[1]/4):np.floor(np.size(x)[1]*3/4),:]
    def disp_func(x):
        return x
    disp_figs = 50 #If set to 0, never display. If set to N>=1, show every N.
    print_interval = 50 #Print cost every N iterations. Default 1. If set to 0, don't print.
    fig_num = 1 #Figure number to display in
    save_results = 1
    save_vars = {'vk'} #List of variable names. If empty or not included in settings file, defaults to just vk, the main volume

    useGpu = 1 #Use GPU or not. 
    if useGpu:  #Check if GPU is present. if not, disable GPU mode
        try:
            useGpu = 1
        except:
            useGpu = 0
    now = datetime.now()
    dtstamp = now.strftime("%d%m%Y_%H%M%S")
