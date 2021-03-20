import PIL.Image as Image
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.ImageOps
import matplotlib.pyplot as plt


def lin_interp(shape, xyd):
  # taken from https://github.com/hunse/kitti
  m, n = shape
  ij, d = xyd[:, 1::-1], xyd[:, 2]
  f = LinearNDInterpolator(ij, d, fill_value=0)
  J, I = np.meshgrid(np.arange(n), np.arange(m))
  IJ = np.vstack([I.flatten(), J.flatten()]).T
  disparity = f(IJ).reshape(shape)
  return disparity


image_path = "0000000006.png" # You image_path
# Load image, convert to numpy array and divide by 256
depth_map = np.asarray(Image.open(image_path)) / 256

# Get location (xy) for valid pixeles
y, x = np.where(depth_map > 0)
# Get depth values for valid pixeles
d = depth_map[depth_map != 0]

# Generate an array Nx3
xyd = np.stack((x,y,d)).T

gt = lin_interp(depth_map.shape, xyd)


print('---------------------depth_map')
print(type(depth_map))
print(depth_map.shape)
print(depth_map.dtype)

print('---------------------gt')
print(type(gt))
print(gt.shape)
print(gt.dtype)
print(gt.min())
print(gt.max())

# if you want inverse depth map, use following code:
# inv=1/(gt+0.0001)
# inv[inv == 10000] = 0;

# img = inv
img = gt
# img = np.load('imgdepth.npy')

vmax = np.percentile(img, 95)
normalizer = mpl.colors.Normalize(vmin=img.min(), vmax=vmax)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
colormapped_im = ((mapper.to_rgba(img)[:, :, :3]) * 255).astype(np.uint8)
im = Image.fromarray(colormapped_im)


im.show()
im.save('depth_result.png')

print('done')

cv2.imshow("depth-image", colormapped_im)
cv2.waitKey(0)




