HearMe
======

Parse sheet music into a playable MIDI file. The proper phrasing is _Optical Music Recognition_.

To use, simply run `hearme.py` with the path to an image of sheet music supplied as an argument, e.g. `> python hearme.py source.png`.

Prerequisites
-------------

- [scikit-image](http://scikit-image.org/) for image analysis
- [scikit-learn](http://scikit-learn.org) for SVM
- [music21](http://web.mit.edu/music21/) for MIDI creation
- [numpy](http://www.numpy.org/) misc.
- scipy
- matplotlib 

Beware!
---------------

- HearMe currently requires a set of training images that aren't on this repo (trying to move away/improve from this method anyways)
- Only *very nice* images work at this time. (TODO) ergo...
 - Images _should not_ have any text in them. Text currently ruins the whole entire process.
 - Noisy images cause problems (anti-aliasing, for example). In this case, *very nice* means perfectly black-and-white images.
- Images can only contain the following symbols (e.g. anything else breaks the 'warranty')
 - Quarter-, half-, and whole- notes and rests
 - Eigthth notes in singles, pairs, tuples, and quadruples
 - The common time symbol (the small 'c')
 - Time signatures are variable (e.g. 2s, 3s, and 4s are recognized, albeit poorly in some instances)
 - Single and double bars
 - Flats and sharps
 - Key signatures are spotty (e.g. works if there are only a few sharps/flats)
 - Treble and bass clefs.