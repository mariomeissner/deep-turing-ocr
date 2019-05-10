import pickle
import subprocess
from bs4 import BeautifulSoup

with open("data/numpy_arrays/alphabet.pickle", "rb") as handle:
    alphabet = pickle.load(handle)
with open("data/numpy_arrays/ixchar.pickle", "rb") as handle:
    ix_to_char = pickle.load(handle)
print("Loaded alphabet and ix_to_char from pickles.")


def label_to_text(ixes):
    ret = []
    for c in ixes:
        if c == len(alphabet) or c == -1:  # CTC Blank
            ret.append("")
        else:
            ret.append(ix_to_char[c])
    return "".join(ret)


def get_boxes(path):

    # Load image and get data
    output = subprocess.run(
        ["tesseract", "--psm", "11", path, "-", "hocr"], stdout=subprocess.PIPE
    )
    soup = BeautifulSoup(output.stdout.decode("utf-8"), features="html.parser")
    lines = soup.findAll("span", {"class": "ocr_line"})

    # Extract line box data
    line_coords = []
    for line in lines:
        coords = line["title"].split(" ")[1:5]
        for i, coord in enumerate(coords):
            coords[i] = int(coord.replace(";", ""))
        line_coords.append(coords)

    return line_coords
