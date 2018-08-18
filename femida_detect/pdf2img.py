import tempfile
import PIL.Image
import os


def pdf_to_image_gs(path, out):
    command = f"gs -dNOPAUSE -q -sDEVICE=png256 -r400 -dBATCH -sOutputFile={out} {path}"
    code = os.system(command)
    if code != 0:
        raise ValueError('gs failed')


def pdf_to_image(path, impl='gs'):
    with tempfile.NamedTemporaryFile(delete=False) as out:
        _select[impl](path, out.name)
        return PIL.Image.open(out.name).convert('RGB')


_select = {'gs': pdf_to_image_gs}
