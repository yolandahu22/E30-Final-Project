# E30-Final-Project
## Developing a Mirrorless Camera Attachment for Lensless Imaging

For our project, we want to implement a variation of the DiffuserCam using an off-the-shelf mirrorless camera. Since most of the existing research in the field approaches lensless imaging with image sensors, we want to adapt the existing designs and algorithms to make a commercial camera capable of doing 3D image reconstruction. 

### Contents

- The "3D Files" folder contains the reproduced original MATLAB code for DiffuserCam 3D reconstruction. Key files include `DiffuserCam_settings.m` and `DiffuserCam_main.m`. The "3d_example_data" folder contains Waller Lab's test data for these programs. 

- The "DiffuserCam" folder contains our own `processImage.m` file. We used this to prepare the `.mat` file for 3D reconstruction. This program also contains cleaning data and 3D plotting functionality post-reconstruction to better visualize our reconstruction result.

- "Images" contains all of the Raw data and PSFs from the camera attachment. While .JPG files are included here, we primarily used .RAF (due to JPEG nonlinearity complexities) and uploaded these on our Google Drive. 

- The "rpi", "test_images", and "tutorial" folders were all directly pulled from Waller Lab. "tutorial" contains some useful Jupyter notebooks (`ADMM.ipynb` and `GD.ipynb`) for a thorough overview of the algorithms. 

- "CAD files" contain schematics of our diffuser and attachment hardware components.

- `ADMM.py`, `GD.py`, and their respecitve configuration files (`admm_config.yml` and `gd_config.yml`) were also reproduced from Waller Lab; we used these for our 2D reconstructions. 

- `ADMM3D_solver.py`, `DiffuserCam_main.py`, and `DiffuserCam_settings.py` are our own files - we configured Waller Lab's 3D files into Python for integration with 2D files. 

- `Code_summary.pdf` serves as a high-level overview of our code for ease of understanding. 

- `process_raw.py` is our own file to prepare the Images for use in ADMM/GD algorithms.

- `psf_stack.py` is our own file to prepare the PSF stacks and test image for 3D reconstruction. It will first convert raw PSFs into TIF files, crop them, and then stack them into a 3D matrix. The matrix will be saved to a `.mat` file at the end. 

### Reconstructing 3D images in Python

The workflow for doing 3D images in Python that we wrote is heavily adapted from the Waller Lab's MATLAB code and follows a similar procedure. Both take the same inputs. A `.mat` file containing all of the point spread functions to be used is specified as `impulse_mat_file_name`, and a corresponding image is specified as `image_file`. There are also options to use a subset of the PSFs, to change the bias, solver parameters, and the number of iterations, although the defaults that are set now work well. Note that paralellization on the GPU is not yet implemented in Python as it is in MATLAB.

Once the settings are established, simply running `DiffuserCam_main.py` from a terminal or IDE will allow for the reconstruction. Plots of slices in XY, XZ, and YZ are plotted every five iterations. When done, the DiffuserCam file returns the reconstructed image, and saves it to a `.mat` file.

So far, the best way we have found to view 3D reconstructed images is still in MATLAB, using the Volume Viewer app. Future improvements could involve finding or creating a similar viewer in Python.
