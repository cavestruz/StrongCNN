import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import roberts, sobel, scharr, prewitt
from StrongCNN.IO.read_fits import *


image =get_fits_obj('../../data/lensed/0_all_imgs.fits')
edge_roberts = roberts(image)
edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')

plt.tight_layout()
plt.savefig('lensed')
