import rawpy
import imageio
from PIL import Image
from resizeimage import resizeimage


path = 'DSCF8065.RAF'
with rawpy.imread(path) as raw:
    bw = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
imageio.imsave('DSCF8065-pro.tif', bw)

image_file=Image.open('DSCF8065-pro.tif')
print(image_file.size)
image_file = image_file.convert('L') # convert image to black and white
image_file.save('DSCF8065-pro-bw.tif')
#print(image_file.size)
with open('DSCF8065-pro-bw.tif', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_width(image, 3000)
        cover = resizeimage.resize_cover(image, [3600, 3000])
        cover.save('DSCF8065_cover.tif', image.format)

path = 'DSCF8074.RAF'
with rawpy.imread(path) as raw:
    bw = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
imageio.imsave('DSCF8074-pro.tif', bw)

image_file=Image.open('DSCF8074-pro.tif')
image_file = image_file.convert('L') # convert image to black and white
image_file.save('DSCF8074-pro-bw.tif')
print(image_file.size)

with open('DSCF8074-pro-bw.tif', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_width(image, 3000)
        cover = resizeimage.resize_cover(image, [3600, 3000])
        cover.save('DSCF8074_cover.tif', image.format)