# E30-Final-Project
## Developing a Mirrorless Camera Attachment for Lensless Imaging

For our project, we want to implement a variation of the DiffuserCam using an off-the-shelf mirrorless camera. Since most of the existing research in the field approaches lensless imaging with image sensors, we want to adapt the existing designs and algorithms to make a commercial camera capable of doing 3D image reconstruction. 

We used the Waller Lab's DiffuserCam code to get started - see their intro below.

=======

# DiffuserCam-Tutorial
#### See our [full tutorial](https://waller-lab.github.io/DiffuserCam/tutorial) for complete guides on setting up the DiffuserCam hardware and installing and running the software.
Below is an overview of the organization of this repo.
<br><br>

#### Home Directory
The base directory contains python code for processing DiffuserCam raw data with two algorithms, gradient descent (`GD.py`) and alternating direction method of multipliers (`ADMM.py`). The corresponding `.yml` files should be modified to include the file path of the raw data that is to be processed. 

#### Rpi Folder
This folder contains python code for previewing and capturing raw images using a Raspberry Pi camera.

#### Tutorial Folder
This folder contains iPython notebooks that walk the user step-by-step through the two algorithms, gradient descent (`GD.ipynb`) and alternating direction method of multipliers (`ADMM.ipynb`). Sample test data is included.

#### Test_Images Folder
This folder contains sample images that you can place on a phone or laptop screen for testing your Raspberry Pi DiffuserCam. We recommend you start with `sprial_bw.gif`.



 

