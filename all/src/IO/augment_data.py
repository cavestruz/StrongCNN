'''image rotations'''
from scipy.ndimage import rotate

def _rotate_images(images_X, images_y, degrees=[0], axes=(1,0)) :

    rotated_images_X = []
    rotated_y = []
    for d in degrees :
        rotated_images_X += [ rotate(image, int(d), axes=axes) for image in images_X ]
        rotated_y += images_y
    return rotated_images_X, rotated_y

augment_methods = { 'rotate_images': _rotate_images,
                    }

def augment_data( X, y, method_label, **method_kwargs) :
    try :
        assert(method_label in augment_methods.keys())
    except KeyError :
        print 'method_label must be one of ', augment_methods.keys()
    return augment_methods[method_label](X, y, **method_kwargs)
