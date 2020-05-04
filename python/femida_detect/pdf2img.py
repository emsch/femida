import tempfile
import cv2
import os
from pathlib import Path

command = ("gs -dBATCH -dNOPAUSE -dNOPROMPT -dMaxBitmap=500000000 -dAlignToPixels=0 "
           "-dGridFitTT=2 -sDEVICE=pngalpha -dTextAlphaBits=4 -dGraphicsAlphaBits=4 "
           "-r500x500 -sOutputFile={out} -f{path} > /dev/null 2> /dev/null")


def pdf_to_images(path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = Path(tmp, 'img%03d.png')
        code = os.system(command.format(out=out, path=path))
        if code != 0:
            raise RuntimeError('gs failed')
        for image in tmp.iterdir():
            yield cv2.imread(str(image))


def pdf_to_images_silent(path, outfmt):
    code = os.system(command.format(out=outfmt, path=path))
    if code != 0:
        raise RuntimeError('gs failed')
    i = 1
    files = []
    while True:
        if os.path.exists(outfmt % (i, )):
            files.append(outfmt % (i, ))
            i += 1
        else:
            break
    return files
