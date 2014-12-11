import argparse
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import (filter, io, feature, img_as_float, measure, restoration, exposure, morphology)
from scipy import ndimage
from scipy.ndimage.filters import (convolve1d, convolve)
from scipy.ndimage.morphology import binary_closing, grey_dilation

MASKS = {
    'sobel': np.array([-1., 0., 1.]) * 0.5,
    'gauss': np.array([1., 4., 6., 4., 1.]) / 16.
}


class Rectangle(object):
    def __init__(self, corners):
        self.corners = np.array([np.min(corners, 0), np.max(corners, 0)])
        print 'Given coords: %s' % str(corners).replace('\n', '')
        print 'yielded rect: %s' % (self)

    def __repr__(self):
        return "< Rect { %s } >" % (str(self.corners).replace('\n', ''))

class Staff(object):
    def __init__(self, lines):
        temp = sorted(lines)
        self.lines = [lines[5*i:5*(i+1)] for i in xrange(len(lines) / 5)]
        self.bounds = (temp[0], temp[-1])

        print "min: %i max: %i" % (self.bounds[0], self.bounds[1])

    def get_idx(self, y_coord):
        return None



def histogram(img, thresh=0.9):
    hist = np.zeros((img.shape[0],))

    for y in xrange(0, img.shape[0]):
        chain = 0
        for x in xrange(1, img.shape[1]):
            chain = 0 if img[y, x] != img[y, x-1] or img[y, x] < 0.5 else chain + 1
            hist[y] += chain
    return hist


def normalize(img):
    imin, imax = np.min(img), np.max(img)
    if imin == imax: imax = imin + 1E-3
    return (img - imin) / (imax - imin)


def generate_midi(src):
    """
    Generate a midi from sheet music.

    :param src: Either a path to the image location, or a numpy array representing the image.
    :return:    The midi file corresponding to the input.
    """
    if isinstance(src, str):
        img = img_as_float(io.imread(src,True))
    elif isinstance(src, np.ndarray):
        img = src
    else:
        raise TypeError("src must be a numpy array or string.")
    img = normalize(restoration.denoise_bilateral(img))
    original = img.copy()

    # Get staff lines
    img = grey_dilation(img, structure=np.ones((1, 15)), mode='nearest')
    img = 1 - normalize(img)

    for y in xrange(img.shape[0]):
        img[y, :] = [np.average(img[y, :])]*img.shape[1]

    img = img > filter.threshold_otsu(img)

    lines = [y for y in xrange(1, img.shape[0]) if not img[y-1, 0] and img[y, 0]]
    # print len(lines), repr(lines)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.imshow(img, cmap=plt.cm.gray)
    # for l in lines: plt.plot([0, img.shape[1]], [l, l])
    # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    # plt.show()

    # now extract notes
    staff = 1-img
    img = original.copy()
    img = grey_dilation(img, structure=np.ones((3, 1)), mode='nearest')
    img = 1-normalize(img)
    # img = filter.vsobel(img)
    # img = exposure.equalize_hist(img)
    img = (img > filter.threshold_otsu(img))

    # blobs = feature.blob_log(img, 3)
    # blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    # # img = binary_closing(img, structure=np.ones((3, 3)))
    fig, ax = plt.subplots(figsize=((8,8)))
    ax.imshow(img, cmap=plt.cm.gray)
    # for blob in blobs:
    #     y, x, r = blob
    #     c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
    #     ax.add_patch(c)
    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()

    exit()

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HearMe: Sheet Music to MIDI converter.')
    parser.add_argument('src', type=str, nargs='?', help='Path to source image (sheet music) to be parsed.')
    parser.add_argument('dest', type=str, nargs='?', help='Name to save midi as, if not supplied will save output in same directory as source.')
    # parser.set_defaults(dest=None, src='../BoleroofFire.jpg')
    parser.set_defaults(dest=None, src='../Frere_Jacques.png')
    args = parser.parse_args()

    # Initialize
    src = path.abspath(args.src)
    if args.dest is None:
        dest = path.basename(src)
        dest = path.abspath(dest.replace(dest.split('.')[-1], 'midi'))
    else:
        dest = path.abspath(args.dest)

    generate_midi(src)


    exit(0)
