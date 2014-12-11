import argparse
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import (filter, io, feature, img_as_float, measure, restoration, color)
from scipy.ndimage.filters import (convolve1d, convolve)
from scipy.ndimage.morphology import binary_closing, grey_dilation
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

MASKS = {
    'sobel': np.array([-1., 0., 1.]) * 0.5,
    'gauss': np.array([1., 4., 6., 4., 1.]) / 16.
}


class Staff(object):
    def __init__(self, lines):
        self.lines = lines
        self.pairwise_lines = [[(l[i], l[i+1]) for i in xrange(len(l)-1)] for l in lines]
        self.bounds = (lines[0][0], lines[-1][-1])

        print "min: %i max: %i" % (self.bounds[0], self.bounds[1])

    def get_idx(self, y_coord):
        """
        Get the pairwise index for the given coordinate (line number, note idx)

        :param y_coord: y-coordinate to check
        :return: a tuple corresponding to: (line number, note idx)
        """
        if y_coord < self.bounds[0]:
            return (0, -np.inf)
        elif y_coord > self.bounds[1]:
            return (len(self.lines), np.inf)

        for (i, line) in enumerate(self.lines):
            if line[-1] < y_coord: continue
            elif y_coord < line[0]: return (i, -np.inf)

            for (j, l) in enumerate(line):
                if line[j] <= y_coord <= line[j+1]:
                    return (i, j)

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

    # canny = filter.canny(img)
    # img = convolve1d(normalize(canny), MASKS['gauss'], 1, mode='nearest')
    # img = normalize(img)
    #
    # hist = histogram(img)
    # hist = normalize(hist) * img.shape[1]
    # avg = (np.average(hist))
    #
    # # pluck out staff lines
    # lines = sorted([i for i in xrange(len(hist)) if hist[i] > avg])
    # staff = Staff([lines[10*i:10*(i+1)] for i in xrange(len(lines) / 10)])


    # now eliminate the staff lines
    img = original.copy()

    img = grey_dilation(img, structure=np.ones((3, 1)))
    img = normalize(restoration.denoise_bilateral(img))
    img = filter.canny(img, sigma=2)

    hradii = np.arange(1, 15)
    print repr(hradii)

    hres = hough_circle(img, hradii)
    centers, accum, radii = [], [], []

    for radius, h in zip(hradii, hres):
        # for each radius, extract two circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        print repr(peaks)
        centers.extend(peaks)
        accum.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)
    print repr(centers)
    # Draw the most prominent 5 circles
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
    image = color.gray2rgb(original)
    for idx in np.argsort(accum)[::-1][:5]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)
        plt.scatter(cx, cy, c='r')
        image[cy, cx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)

    print np.min(img), ' ', np.max(img)

    # img = filter.gaussian_filter(img, 0.5)
    # img = filter.threshold_adaptive(img, 5)
    # img = filter.canny(img)
    # img = binary_closing(img>0.1, np.ones((3, 3)))
    io.imshow(img)
    # for line in lines: plt.plot((0, img.shape[1]), (line, line), '-r')
    plt.show()

    return None

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


    exit(0)
