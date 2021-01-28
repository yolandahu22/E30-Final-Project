import rawpy
import imageio
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from scipy.io import loadmat,savemat 

#process psfs for 3d Image from raw fiile 
psf_stack = np.zeros([300,400,40])

for i in range(94,133):
    if i<100:
        path = './210126/DSCF80'+str(i)+'.RAF'
    else:
        path = './210126/DSCF8'+str(i)+'.RAF'
    name='./DiffuserCam/'+str(i)+'.tif'
    print(name)
    print(path)
    with rawpy.imread(path) as raw:
        bw = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    imageio.imsave('mid.tif', bw)
    image_file=Image.open('mid.tif')
    print(image_file.size)
    image_file = image_file.convert('L') # convert image to black and white
    cover = resizeimage.resize_width(image_file, 300)
    cover = resizeimage.resize_cover(image_file, [400, 300])
    psf_stack[:,:,i-94] = np.array(cover)
    cover.save(name, image_file.format)


outfile = 'testpsf_python.mat'
mdic = {"psf": psf_stack}
savemat(outfile,mdic)

'''
# processing the test image 
path = 'DSCF8038.RAF'
with rawpy.imread(path) as raw:
    bw = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
imageio.imsave('DSCF8038-pro.tif', bw)

image_file=Image.open('DSCF8038-pro.tif')
print(image_file.size)
image_file = image_file.convert('L') # convert image to black and white
#print(image_file.size)
cover = resizeimage.resize_width(image_file, 1200)
cover = resizeimage.resize_cover(image_file, [1600, 1200])
cover.save('./DiffuserCam/testimage_38_large.tif', image_file.format)
'''