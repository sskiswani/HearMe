import argparse
from os import path
import numpy as np
import matplotlib.pyplot as plt
from skimage import (filter, io, feature, img_as_float)
from scipy.ndimage.filters import convolve1d

MASKS = {
    'sobel': np.array([-1., 0., 1.]) * 0.5,
    'gauss': np.array([1., 4., 6., 4., 1.]) / 16.
}

def histogram(img, thresh=0.9):
    hist = np.zeros((img.shape[0],))

    for y in xrange(1, img.shape[0]-1):
        for x in xrange(1, img.shape[1]-1):
            if img[y, x] < thresh and img[y, x-1] < thresh and img[y, x+1] < thresh:
                hist[y] += 1

    return hist

def normalize(img):
    imin, imax = np.min(img), np.max(img)
    if imin == imax: imax = imin + 1E-3
    return (img - imin) / (imax - imin)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HearMe: Sheet Music to MIDI converter.')
    parser.add_argument('src', type=str, nargs='?', help='Path to source image (sheet music) to be parsed.')
    parser.add_argument('dest', type=str, nargs='?', help='Name to save midi as, if not supplied will save output in same directory as source.')
    parser.set_defaults(dest=None, src='../BoleroofFire.jpg')
    args = parser.parse_args()

    # Initialize
    src = path.abspath(args.src)
    if args.dest is None:
        dest = path.basename(src)
        dest = path.abspath(dest.replace(dest.split('.')[-1], 'midi'))
    else:
        dest = path.abspath(args.dest)

    img = img_as_float(io.imread(src, True))
    img = convolve1d(filter.vsobel(img), np.array([1., 2., 2., 6., 2., 2., 1.]) / 16., 1, mode='nearest')
    img = 1 - img_as_float(io.imread(src, True)) - normalize(img)
    print 'min: %.3f max: %.3f' % (np.min(img), np.max(img))
    io.imshow(img)
    hist = np.array([np.sum(img[y, :] > 0.4) for y in xrange(img.shape[0])])
    plt.scatter(hist, range(img.shape[0]), c='r')
    avg = 2*np.average(hist)
    lines = np.array([hist[y] - hist[y-1] > avg for y in xrange(len(hist))]) * img.shape[0] * 0.75

    plt.scatter(lines, range(img.shape[0]), c='b')
    plt.show()

    exit(0)
