import tempfile
import PIL.Image
import os
from pathlib import Path


def pdf_to_images(path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = Path(tmp, 'img%03d.png')
        command = f"gs -dNOPAUSE -q -sDEVICE=png256 -r400 -dBATCH -o {out} {path}"
        code = os.system(command)
        if code != 0:
            raise ValueError('gs failed')
        return [PIL.Image.open(image).convert('RGB') for image in tmp.iterdir()]
