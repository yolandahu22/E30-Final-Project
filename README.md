# E30-Final-Project
## Developing a Mirrorless Camera Attachment for Lensless Imaging

For our project, we want to implement a variation of the DiffuserCam using an off-the-shelf mirrorless camera. Since most of the existing research in the field approaches lensless imaging with image sensors, we want to adapt the existing designs and algorithms to make a commercial camera capable of doing 3D image reconstruction. 

### Contents

- The "3D Files" folder contains the reproduced original MATLAB code for DiffuserCam 3D reconstruction. Key files include `DiffuserCam_settings.m` and `DiffuserCam_main.m`. The "3d_example_data" folder contains Waller Lab's test data for these programs. 

- The "DiffuserCam" folder contains our own `processImage.m` file. We used this to prepare the `.mat` file for 3D reconstruction.

- "Images" contains all of the Raw data and PSFs from the camera attachment. While both .JPG and .RAF are included, we primarily used .RAF (due to JPEG nonlinearity complexities). 

- The "rpi", "test_images", and "tutorial" folders were all directly pulled from Waller Lab. "tutorial" contains some useful Jupyter notebooks (`ADMM.ipynb` and `GD.ipynb`) for a thorough overview of the algorithms. 

- "CAD files" contain schematics of our diffuser and attachment hardware components.

- `ADMM.py`, `GD.py`, and their respecitve configuration files (`admm_config.yml` and `gd_config.yml`) were also reproduced from Waller Lab; we used these for our 2D reconstructions. 

- `ADMM3D_solver.py`, `DiffuserCam_main.py`, and `DiffuserCam_settings.py` are our own files - we configured Waller Lab's 3D files into Python for integration with 2D files. 

- `Code_summary.pdf` serves as a high-level overview of our code for ease of understanding. 

- `process_raw.py` is our own file to prepare the Images for use in ADMM/GD algorithms.
