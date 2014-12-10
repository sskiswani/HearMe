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

def generate_midi(src_path):
    img = img_as_float(io.imread(src_path, True))
    img = normalize(filter.canny(img))
    img = convolve1d(img, MASKS['gauss'], 1, mode='nearest')
    img = normalize(img)

    hist = histogram(img)
    hist = normalize(hist) * img.shape[1]
    avg = (np.average([y for y in hist]))

    # pluck out staff lines
    lines = [i for i in xrange(len(hist)) if hist[i] > avg]
    lines = [(lines[i], lines[i+1]) for i in xrange(len(lines)-1)]
    if not len(lines) % 5: print 'ERROR: Have %i staff lines. (should be a multiple of 5)' % len(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HearMe: Sheet Music to MIDI converter.')
    parser.add_argument('src', type=str, nargs='?', help='Path to source image (sheet music) to be parsed.')
    parser.add_argument('dest', type=str, nargs='?', help='Name to save midi as, if not supplied will save output in same directory as source.')
    parser.set_defaults(dest=None, src='../BoleroofFire.jpg')
    # parser.set_defaults(dest=None, src='../Frere_Jacques.png')
    args = parser.parse_args()

    # Initialize
    src = path.abspath(args.src)
    if args.dest is None:
        dest = path.basename(src)
        dest = path.abspath(dest.replace(dest.split('.')[-1], 'midi'))
    else:
        dest = path.abspath(args.dest)

    generate_midi(src)

    # io.imshow(img)
    # plt.scatter(hist, range(img.shape[0]), c='r')
    # plt.plot([avg, avg], [0, img.shape[0]], c='g')
    # plt.show()



    exit(0)
