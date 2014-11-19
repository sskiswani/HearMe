import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HearMe: Sheet Music to MIDI converter.')
    parser.add_argument('src', '--s', type=str, help='path to source.')
    parser.add_argument('dest', '--d', type=str, nargs='?', help='name to save midi as.')
    parser.set_defaults(dest=None)
    args = parser.parse_args()

    print 'hi'



    exit(0)