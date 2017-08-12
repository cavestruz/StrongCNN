from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
lensed_image = pyfits.getdata("../data/lensed/0_all_imgs.fits")
unlensed_image = pyfits.getdata("../data/unlensed/0_all_imgs.fits")

test_images = [lensed_image, unlensed_image]

gray_images = [rgb2gray(image) for image in test_images]

blobs_log = [blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1) for image_gray in gray_images]

# Compute radii in the 3rd column.
# for bl in blobs_log : 
#     bl[:, 2] = bl[:, 2] * sqrt(2)

# blobs_dog = [blob_dog(image_gray, max_sigma=30, threshold=.1) for image_gray in gray_images]
# for bd in blobs_dog :
#     bd[:, 2] = bd[:, 2] * sqrt(2)

blobs_doh = [blob_doh(image_gray, max_sigma=30, threshold=.01) for image_gray in gray_images]
print('Finished blob detection with Determinant of Hessian')

blobs_list = [#blobs_log, blobs_dog, 
              blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']

sequence = zip(blobs_list, colors, titles)

for i, image in enumerate(['lensed','unlensed']) :
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    plt.tight_layout()

    axes = axes.ravel()
    for blobs, color, title in sequence:
        ax = axes[0]
        axes = axes[1:]
        ax.set_title(title)
        ax.imshow(image, interpolation='nearest')
        ax.set_axis_off()
        # for blob in blobs:
        #     y, x, r = blob[i]
        #     c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        #     ax.add_patch(c)
    plt.savefig('blob_finder_test_'+image)
