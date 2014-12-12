import argparse
from os import path, listdir
import numpy as np
import matplotlib.pyplot as plt
from skimage import (filter, io, img_as_float, restoration, transform)
from scipy import ndimage
from scipy.ndimage import convolve1d
from scipy.ndimage.morphology import (grey_dilation, binary_erosion, binary_dilation)
from sklearn import svm, feature_selection
from sklearn.externals import joblib
import music21 as m21

MATCHING_SIZE = np.array([62, 29])

CLASSES = {
    'BASS_CLEF': 0,
    'C': 1,
    'FLAT': 2,
    'HALF_NOTE': 3,
    'NATURAL': 4,
    'QUARTER_NOTE': 5,
    'QUARTER_REST': 6,
    'SHARP': 7,
    'TREBLE_CLEF': 8,
    'WHOLE': 9
}

MASKS = {
    'sobel': np.array([-1., 0., 1.]) * 0.5,
    'gauss': np.array([1., 4., 6., 4., 1.]) / 16.
}


def feature_extractor(i, sel=None, visualize=False, save=False, shape=None):
    if shape is None: shape = np.array([30, 30])
    img = transform.resize(i, shape, mode='nearest')
    img = grey_dilation(img, structure=np.ones((3, 1)), mode='nearest')

    # if save and False:
    # global accum
    #     io.imsave(path.abspath('../res/' + str(accum) + '.png'), img)
    #     accum+=1

    if visualize:
        fig = plt.figure(frameon=False)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.axis('off')
        fig.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0, left=0, right=1)
        plt.show()
    # result = feature.hog(img, 5, (3, 3),(1, 1), normalise=True)
    result = img.flatten()

    return sel.transform(result).flatten() if sel is not None else result


class Staff(object):
    def __init__(self, lines):
        temp = sorted(lines)
        self.lines = [lines[5 * i:5 * (i + 1)] for i in xrange(len(lines) / 5)]
        self.bounds = (temp[0], temp[-1])

    def __getitem__(self, item):
        return self.lines[item]

    def __len__(self):
        return len(self.lines)

    def __itr__(self):
        return iter(self.lines)


def histogram(img, thresh=0.9):
    hist = np.zeros((img.shape[0],))

    for y in xrange(0, img.shape[0]):
        chain = 0
        for x in xrange(1, img.shape[1]):
            chain = 0 if img[y, x] != img[y, x - 1] or img[y, x] < 0.5 else chain + 1
            hist[y] += chain
    return hist


def normalize(img):
    imin, imax = np.min(img), np.max(img)
    if imin == imax: imax = imin + 1E-3
    return (img - imin) / (imax - imin)


def load_classifier(fn='./svm.pkl', save_if_not_exists=True):
    fn = path.abspath(fn)
    # if path.exists(fn): return joblib.load(fn)

    # TODO: Hardcode the test images
    training_dir = path.abspath('./training')
    clean = [path.join(training_dir, f) for f in listdir(training_dir) if path.isfile(path.join(training_dir, f))]
    imgs = [io.imread(f, True) for f in clean]

    # max_shape = np.max([i.shape for i in imgs], 0)
    # print repr(max_shape)

    hogs = [feature_extractor(i) for i in imgs]
    X = hogs
    Y = [path.basename(f).split('_')[0].replace('-', '_').upper() for f in clean]

    sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit(X, Y)
    X = [sel.transform(x).flatten() for x in X]

    clf = svm.LinearSVC()
    clf.fit(X, Y)

    # clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform')
    # clf.fit(X, Y)

    if save_if_not_exists: joblib.dump(clf, fn)
    return (clf, sel)


def generate_midi(src):
    """
    Generate a midi from sheet music.

    :param src: Either a path to the image location, or a numpy array representing the image.
    :return:    The midi file corresponding to the input.
    """
    if isinstance(src, str):
        img = img_as_float(io.imread(src, True))
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
        img[y, :] = [np.average(img[y, :])] * img.shape[1]

    img = img > filter.threshold_otsu(img)

    lines = [y for y in xrange(1, img.shape[0]-1) if not img[y-1, 0] and img[y, 0]]
    staff = Staff(lines)
    staff_img = 1 - img
    print len(lines)


    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.imshow(original, cmap=plt.cm.gray)
    # for l in lines: plt.plot([0, img.shape[1]], [l, l])
    # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    # plt.show()


    # now separate notes from staff
    img = 1 - original.copy()
    img = img > filter.threshold_otsu(img)
    img = binary_erosion(img, np.ones((2, 1)))
    img = binary_dilation(img, np.ones((2, 1)))
    for y in lines:
        for x in xrange(img.shape[1]):
            img[y, x] = (img[y, x] & img[y+1, x] & img[y-1, x])
    nostaff = 1 - normalize(img)
    img = grey_dilation(1 - nostaff, structure=np.ones((2, 3)), mode='nearest')
    img = normalize(img)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.imshow(img, cmap=plt.cm.gray)
    # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    # plt.show()

    # now find the notes in the staff
    labels, num_labels = ndimage.label(img, np.ones((3, 3)))
    slices = ndimage.find_objects(labels)
    # clf, sel = load_classifier()
    #
    # matcher = original
    # matches = []
    # for i, loc in enumerate(slices):
    #     patch = transform.resize(matcher[loc], MATCHING_SIZE, mode='nearest')
    #     hog = feature_extractor(patch, sel, True)
    #     print "patch[%i] is %s " % (i, clf.predict([hog]))
    #     matches.append((clf.predict([hog]), loc))


    # Plot the first match (should be a treble clef)
    # loc = slices[0]
    original = (1-original)
    original = original > 0.5
    original = binary_erosion(original, np.ones((3, 1)))
    fig, ax = plt.subplots(nrows=4, figsize=((8,8)))
    ax[0].imshow(labels, cmap=plt.cm.spectral)
    ax[1].imshow(img, cmap=plt.cm.gray)
    ax[2].imshow(original, cmap=plt.cm.gray)
    ax[3].imshow((original)*img, cmap=plt.cm.gray)
    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()

    exit()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HearMe: Sheet Music to MIDI converter.')
    parser.add_argument('src', type=str, nargs='?', help='Path to source image (sheet music) to be parsed.')
    parser.add_argument('dest', type=str, nargs='?',
                        help='Name to save midi as, if not supplied will save output in same directory as source.')
    # parser.set_defaults(dest=None, src='../BoleroofFire.jpg')
    parser.set_defaults(dest=None, src='../Frere_Jacques.png')
    # parser.set_defaults(dest=None, src='../YB4001Canon_Frere_Jacques.png')
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
