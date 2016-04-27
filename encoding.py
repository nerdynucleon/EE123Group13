import zipfile
import StringIO

test = False

def encode(raw_str):
    if test: return raw_str
    comp_file = StringIO.StringIO()
    with zipfile.ZipFile(comp_file, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('comp', raw_str)
    return comp_file.getvalue()

def decode(enc_str):
    if test: return enc_str
    comp_file = StringIO.StringIO(enc_str)
    with zipfile.ZipFile(comp_file, mode='r') as z:
        return z.read('comp')

