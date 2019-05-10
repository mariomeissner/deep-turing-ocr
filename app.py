import json
import os
from flask import Flask, request
from flask_cors import CORS, cross_origin
from model.model import create_model
from model.tools import label_to_text, get_boxes
from PIL import Image
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
app = Flask(__name__, static_url_path="/client")
IMAGE_FOLDER = "evaluate/"
DATASET_FOLDER = "dataset/"
TURING_FILES_FOLDER = "turingFiles/"
LABELS_FILE = "labels.txt"
FILE_EXTENSION = ".png"

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

print("Loading the model...")
with open("model/params.json", "r") as file:
    params = json.load(file)
model, decoders = create_model(params, gpu=False)
decoder = decoders[0]
model.load_weights("model/weights.h5")

# A tensor where all values are the same, is required by ctc loss
ctc_input_length = (
    params["img_w"] // (params["pool_size"] ** params["num_convs"])
) - params["ctc_cut"]

# Create necessary folders if its the first time
for folder in ["dataset", "turingFiles", "evaluate"]:
    if not os.path.exists(folder):
        print(f"Creating folder {folder}.")
        os.mkdir(folder)

print("Ready!")


@app.route("/predict_boxes", methods=["GET"])
@cross_origin()
def predict_boxes():
    file = request.args.get("file")
    if not file:
        print("No file given.")
        return "No file given"
    box_coords = get_boxes(IMAGE_FOLDER + file)
    print(f"Sent boxes: {box_coords}")
    return json.dumps(box_coords)


@app.route("/predict_lines", methods=["POST"])
@cross_origin()
def predict_lines():

    box_coords = request.get_json()
    file = request.args.get("file")

    print(box_coords)

    if not file:
        print("No file given.")
        return "No file given"
    img = Image.open(IMAGE_FOLDER + file).convert("L")
    image_arrays = []

    for (x, y, x2, y2) in box_coords:
        crop = img.crop((x, y, x2, y2))
        crop = crop.resize((params["img_w"], params["img_h"]), Image.ANTIALIAS)
        crop = np.array(crop).T / 255  # Adapt to model input requirements
        image_arrays.append(np.array(crop))

    images = np.array(image_arrays)
    images = np.expand_dims(images, axis=3)  # Add shallow channel dimension
    input_lengths = np.expand_dims(np.array([ctc_input_length] * len(images)), 1)
    predictions = decoder([images, input_lengths])[0]
    prediction_strings = []

    for prediction in predictions:
        prediction = list(np.squeeze(prediction))
        pred_string = label_to_text(prediction)
        prediction_strings.append(pred_string)

    return json.dumps(prediction_strings)


@app.route("/save_data", methods=["POST"])
@cross_origin()
def save_data():

    # Get the POST data
    data = request.get_json()
    file = request.args.get("file")
    img = Image.open(IMAGE_FOLDER + file).convert("L")

    print(data)

    # Updatable dictionary of labels
    labels = {}
    labelsfile_path = DATASET_FOLDER + LABELS_FILE
    if os.path.isfile(labelsfile_path) and os.path.getsize(labelsfile_path) > 0:
        with open(labelsfile_path, "r") as labelsfile:
            for line in labelsfile:
                name, label = line.split(": ")
                labels[name] = label.strip()

    # Update of create all labels found in the json data
    i = 0
    for bunch in data:
        (x, y, x2, y2) = bunch["coords"]
        label = bunch["label"]
        crop = img.crop((x, y, x2, y2))
        filename = file.split(".")[0] + "_" + str(i)
        crop.save(DATASET_FOLDER + filename + FILE_EXTENSION)
        labels[filename] = label
        i += 1

    # Write new dictionary to file
    with open(DATASET_FOLDER + LABELS_FILE, "w") as labelsfile:
        for name, label in labels.items():
            labelsfile.write(name + ": " + label + "\n")

    return "Done"


@app.route("/append_turing_lines", methods=["POST"])
@cross_origin()
def append_turing_lines():
    # Get the POST data
    lines = request.get_json()
    filename = request.args.get("file")
    with open(TURING_FILES_FOLDER + filename, "a") as file:
        for line in lines:
            file.write(line + "\n")
    return "Done"
