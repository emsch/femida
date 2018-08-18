import tempfile
import PIL.Image
import os
from pathlib import Path

command = ("gs  -dBATCH -dNOPAUSE -dNOPROMPT -dMaxBitmap=500000000 -dAlignToPixels=0 "
           "-dGridFitTT=2 -sDEVICE=pngalpha -dTextAlphaBits=4 -dGraphicsAlphaBits=4 "
           "-r500x500 -sOutputFile={out} -f{path}")


def pdf_to_images(path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = Path(tmp, 'img%03d.png')
        code = os.system(command.format(out=out, path=path))
        if code != 0:
            raise ValueError('gs failed')
        return [PIL.Image.open(image).convert('RGB') for image in tmp.iterdir()]
