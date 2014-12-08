HearMe
======

Parse sheet music into a playable MIDI file. The proper phrasing is _Optical Music Recognition_.

To use, simply run `hearme.py` with the path to an image of sheet music supplied as an argument, e.g. `> python hearme.py source.png`.

Prerequisites
-------------

- [scikit-image](http://scikit-image.org/) for image analysis
- [music21](http://web.mit.edu/music21/) for MIDI creation
- [numpy](http://www.numpy.org/) misc.
- scipy
- matplotlib 